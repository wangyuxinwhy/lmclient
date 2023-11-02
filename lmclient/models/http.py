from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, ClassVar, Generator, Literal, Mapping, Optional, Sequence, Union

import httpx
from httpx._types import ProxiesTypes
from httpx_sse import aconnect_sse, connect_sse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import Required, TypedDict, override

from lmclient.exceptions import ResponseFailedError
from lmclient.models.base import T_P, BaseChatModel
from lmclient.types import ChatModelOutput, ChatModelStreamOutput, Messages, ModelResponse, PrimitiveData, Stream, TextMessage

logger = logging.getLogger(__name__)
QueryParams = Mapping[str, Union[PrimitiveData, Sequence[PrimitiveData]]]
Headers = Mapping[str, str]


class RetryStrategy(BaseModel):
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpChatModelKwargs(TypedDict, total=False):
    timeout: Optional[int]
    retry: Union[bool, RetryStrategy]
    proxies: Union[ProxiesTypes, None]


class HttpxPostKwargs(TypedDict, total=False):
    url: Required[str]
    json: Required[Any]
    params: QueryParams
    headers: Headers
    timeout: Optional[int]


class HttpChatModel(BaseChatModel[T_P], ABC):
    model_type = 'http'
    stream_model: ClassVar[Literal['sse', 'basic']] = 'sse'

    def __init__(
        self,
        parameters: T_P,
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        proxies: ProxiesTypes | None = None,
    ):
        super().__init__(parameters=parameters)
        self.timeout = timeout or 60
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None
        self.proxies = proxies

    @abstractmethod
    def get_request_parameters(self, messages: Messages, parameters: T_P) -> HttpxPostKwargs:
        ...

    @abstractmethod
    def parse_reponse(self, response: ModelResponse) -> Messages:
        ...

    @abstractmethod
    def get_stream_request_parameters(self, messages: Messages, parameters: T_P) -> HttpxPostKwargs:
        ...

    @abstractmethod
    def parse_stream_response(self, response: ModelResponse) -> Stream:
        ...

    def _chat_completion_without_retry(self, messages: Messages, parameters: T_P) -> ChatModelOutput[T_P]:
        with httpx.Client(proxies=self.proxies) as client:
            http_parameters = self.get_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            logger.info(f'HTTP Request: {http_parameters}')
            http_response = client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_response = http_response.json()
        logger.info(f'HTTP Response: {model_response}')
        new_messages = self.parse_reponse(model_response)
        reply = new_messages[-1]['content']
        reply = reply if isinstance(reply, str) else ''
        return ChatModelOutput[T_P](
            model_id=self.model_id,
            parameters=parameters.model_copy(),
            messages=new_messages,
            extra_info={'http_response': model_response},
            reply=reply,
        )

    async def _async_chat_completion_without_retry(self, messages: Messages, parameters: T_P) -> ChatModelOutput[T_P]:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            http_parameters = self.get_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            logger.info(f'Async HTTP Request: {http_parameters}')
            http_response = await client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_response = http_response.json()
        logger.info(f'Async HTTP Response: {model_response}')
        new_messages = self.parse_reponse(model_response)
        reply = new_messages[-1]['content']
        reply = reply if isinstance(reply, str) else ''
        return ChatModelOutput[T_P](
            model_id=self.model_id,
            parameters=parameters.model_copy(),
            messages=new_messages,
            extra_info={'http_response': model_response},
            reply=reply,
        )

    @override
    def _chat_completion(self, messages: Messages, parameters: T_P) -> ChatModelOutput[T_P]:
        if self.retry_strategy is None:
            return self._chat_completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = retry(wait=wait, stop=stop)(self._chat_completion_without_retry)(messages, parameters)
        return output

    @override
    async def _async_chat_completion(self, messages: Messages, parameters: T_P) -> ChatModelOutput[T_P]:
        if self.retry_strategy is None:
            return await self._async_chat_completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = await retry(wait=wait, stop=stop)(self._async_chat_completion_without_retry)(messages, parameters)
        return output

    @override
    def _stream_chat_completion(self, messages: Messages, parameters: T_P) -> Generator[ChatModelStreamOutput[T_P], None, None]:
        if self.stream_model == 'sse':
            stream_data_generator = self._generate_data_from_sse_stream(messages, parameters)
        else:
            stream_data_generator = self._generate_data_from_basic_stream(messages, parameters)

        reply = ''
        start = False
        finish = False
        stream_response = {}

        for stream_data in stream_data_generator:
            stream_response = self._preprocess_stream_data(stream_data)

            try:
                stream = self.parse_stream_response(stream_response)
            except BaseException as e:
                logger.error(f'Parse stream response failed: {stream_response}')
                raise ResponseFailedError(f'{stream_response}') from e

            if not start:
                stream.control = 'start'
                start = True
            if stream.control == 'finish':
                finish = True
            if stream.delta:
                reply += stream.delta
            yield ChatModelStreamOutput(
                model_id=self.model_id,
                parameters=parameters.model_copy(deep=True),
                messages=[TextMessage(role='assistant', content=reply)],
                reply=reply,
                extra_info={'http_response': stream_response},
                stream=stream,
            )

        if not finish:
            stream = Stream(delta='', control='finish')
            yield ChatModelStreamOutput(
                model_id=self.model_id,
                parameters=parameters.model_copy(deep=True),
                messages=[TextMessage(role='assistant', content=reply)],
                reply=reply,
                extra_info={'http_response': stream_response},
                stream=stream,
            )

    def _preprocess_stream_data(self, stream_data: str) -> ModelResponse:
        try:
            stream_response = json.loads(stream_data)
        except json.JSONDecodeError:
            stream_response = {'data': stream_data}
        return stream_response

    def _generate_data_from_sse_stream(self, messages: Messages, parameters: T_P) -> Generator[str, None, None]:
        with httpx.Client(proxies=self.proxies) as client:
            http_parameters = self.get_stream_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            logger.info(f'HTTP Request: {http_parameters}')
            with connect_sse(client=client, method='POST', **http_parameters) as event_source:
                for sse in event_source.iter_sse():
                    yield sse.data

    def _generate_data_from_basic_stream(self, messages: Messages, parameters: T_P) -> Generator[str, None, None]:
        http_parameters = self.get_stream_request_parameters(messages, parameters)
        http_parameters.update({'timeout': self.timeout})
        with httpx.Client(proxies=self.proxies) as client:
            with client.stream('POST', **http_parameters) as source:
                for line in source.iter_lines():
                    yield line

    @override
    async def _async_stream_chat_completion(
        self, messages: Messages, parameters: T_P
    ) -> AsyncGenerator[ChatModelStreamOutput[T_P], None]:
        if self.stream_model == 'sse':
            stream_data_generator = self._async_generate_data_from_sse_stream(messages, parameters)
        else:
            stream_data_generator = self._async_generate_data_from_basic_stream(messages, parameters)

        reply = ''
        start = False
        finish = False

        async for data in stream_data_generator:
            try:
                stream_response = json.loads(data)
            except json.JSONDecodeError:
                stream_response = {'data': data}
            stream = self.parse_stream_response(stream_response)

            if not start:
                stream.control = 'start'
                start = True
            if stream.control == 'finish':
                finish = True
            if stream.delta:
                reply += stream.delta
            yield ChatModelStreamOutput(
                model_id=self.model_id,
                parameters=parameters.model_copy(deep=True),
                messages=[TextMessage(role='assistant', content=reply)],
                reply=reply,
                extra_info={'http_response': stream_response},
                stream=stream,
            )

        if not finish:
            stream = Stream(delta='', control='finish')
            yield ChatModelStreamOutput(
                model_id=self.model_id,
                parameters=parameters.model_copy(deep=True),
                messages=[TextMessage(role='assistant', content=reply)],
                reply=reply,
                stream=stream,
            )

    async def _async_generate_data_from_sse_stream(self, messages: Messages, parameters: T_P) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            http_parameters = self.get_stream_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            logger.info(f'HTTP Request: {http_parameters}')
            async with aconnect_sse(client=client, method='POST', **http_parameters) as event_source:
                async for sse in event_source.aiter_sse():
                    yield sse.data

    async def _async_generate_data_from_basic_stream(self, messages: Messages, parameters: T_P) -> AsyncGenerator[str, None]:
        http_parameters = self.get_stream_request_parameters(messages, parameters)
        http_parameters.update({'timeout': self.timeout})
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            async with client.stream('POST', **http_parameters) as source:
                async for line in source.aiter_lines():
                    yield line
