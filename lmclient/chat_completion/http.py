from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, ClassVar, Dict, Generator, Literal, Mapping, Optional, Sequence, TypeVar, Union

import httpx
from httpx._types import ProxiesTypes
from httpx_sse import aconnect_sse, connect_sse
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import Required, TypedDict, override

from lmclient.chat_completion.base import ChatCompletionModel
from lmclient.chat_completion.message import AssistantMessage, Messages
from lmclient.chat_completion.model_output import (
    ChatCompletionModelOutput,
    ChatCompletionModelStreamOutput,
    FinishStream,
    Stream,
)
from lmclient.chat_completion.model_parameters import ModelParameters
from lmclient.types import PrimitiveData

P = TypeVar('P', bound=ModelParameters)
HttpResponse = Dict[str, Any]
QueryParams = Mapping[str, Union[PrimitiveData, Sequence[PrimitiveData]]]
Headers = Mapping[str, str]
logger = logging.getLogger(__name__)


class RetryStrategy(BaseModel):
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpChatModelInitKwargs(TypedDict, total=False):
    timeout: Optional[int]
    retry: Union[bool, RetryStrategy]
    proxies: Union[ProxiesTypes, None]


class HttpxPostKwargs(TypedDict, total=False):
    url: Required[str]
    json: Required[Any]
    headers: Required[Headers]
    params: QueryParams
    timeout: Optional[int]


class UnexpectedResponseError(Exception):
    """
    Exception raised when an unexpected response is received from the server.

    Attributes:
        response (dict): The response from the server.
    """

    def __init__(self, response: dict, *args: Any) -> None:
        super().__init__(response, *args)


class HttpChatModel(ChatCompletionModel[P], ABC):
    model_type = 'http'
    parse_stream_strategy: ClassVar[Literal['sse', 'basic']] = 'sse'

    def __init__(
        self,
        parameters: P,
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        proxies: ProxiesTypes | None = None,
    ) -> None:
        super().__init__(parameters=parameters)
        self.timeout = timeout or 60
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None
        self.proxies = proxies

    @abstractmethod
    def _get_request_parameters(self, messages: Messages, parameters: P) -> HttpxPostKwargs:
        ...

    @abstractmethod
    def _parse_reponse(self, response: HttpResponse) -> ChatCompletionModelOutput:
        ...

    @abstractmethod
    def _get_stream_request_parameters(self, messages: Messages, parameters: P) -> HttpxPostKwargs:
        ...

    @abstractmethod
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        ...

    def _completion_without_retry(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        with httpx.Client(proxies=self.proxies) as client:
            http_parameters = self._get_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            http_response = client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_output = self._parse_reponse(http_response.json())
        model_output.extra['http_response'] = http_response.json()
        return model_output

    async def _async_completion_without_retry(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            http_parameters = self._get_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            http_response = await client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_output = self._parse_reponse(http_response.json())
        model_output.extra['http_response'] = http_response.json()
        return model_output

    @override
    def _completion(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        if self.retry_strategy is None:
            return self._completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return retry(wait=wait, stop=stop)(self._completion_without_retry)(messages, parameters)

    @override
    async def _async_completion(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        if self.retry_strategy is None:
            return await self._async_completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        return await retry(wait=wait, stop=stop)(self._async_completion_without_retry)(messages, parameters)

    @override
    def _stream_completion(self, messages: Messages, parameters: P) -> Generator[ChatCompletionModelStreamOutput, None, None]:
        if self.parse_stream_strategy == 'sse':
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
                stream = self._parse_stream_response(stream_response)
            except BaseException as e:
                raise UnexpectedResponseError(stream_response) from e

            if not start:
                start_stream = Stream(delta='', control='start')
                yield ChatCompletionModelStreamOutput(
                    chat_model_id=self.model_id,
                    messages=[AssistantMessage(content=reply)],
                    stream=start_stream,
                )
                if stream.control == 'start':
                    stream.control = 'continue'
                start = True
            if stream.delta:
                reply += stream.delta

            if isinstance(stream, FinishStream):
                finish = True
                yield ChatCompletionModelStreamOutput(
                    chat_model_id=self.model_id,
                    messages=[AssistantMessage(content=reply)],
                    extra={'http_response': stream_response},
                    stream=stream,
                    finish_reason=stream.finish_reason,
                    usage=stream.usage,
                    cost=stream.cost,
                )
                break
            else:
                yield ChatCompletionModelStreamOutput(
                    chat_model_id=self.model_id,
                    messages=[AssistantMessage(content=reply)],
                    extra={'http_response': stream_response},
                    stream=stream,
                )

        if not finish:
            raise UnexpectedResponseError(stream_response, 'Stream is not finished.')

    def _preprocess_stream_data(self, stream_data: str) -> HttpResponse:
        try:
            stream_response = json.loads(stream_data)
        except json.JSONDecodeError:
            stream_response = {'data': stream_data}
        return stream_response

    def _generate_data_from_sse_stream(self, messages: Messages, parameters: P) -> Generator[str, None, None]:
        with httpx.Client(proxies=self.proxies) as client:
            http_parameters = self._get_stream_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            with connect_sse(client=client, method='POST', **http_parameters) as event_source:
                for sse in event_source.iter_sse():
                    yield sse.data

    def _generate_data_from_basic_stream(self, messages: Messages, parameters: P) -> Generator[str, None, None]:
        http_parameters = self._get_stream_request_parameters(messages, parameters)
        http_parameters.update({'timeout': self.timeout})
        with httpx.Client(proxies=self.proxies) as client, client.stream('POST', **http_parameters) as source:
            for line in source.iter_lines():
                yield line

    @override
    async def _async_stream_completion(
        self, messages: Messages, parameters: P
    ) -> AsyncGenerator[ChatCompletionModelStreamOutput, None]:
        if self.parse_stream_strategy == 'sse':
            stream_data_generator = self._async_generate_data_from_sse_stream(messages, parameters)
        else:
            stream_data_generator = self._async_generate_data_from_basic_stream(messages, parameters)

        reply = ''
        start = False
        finish = False
        stream_response = {}

        async for data in stream_data_generator:
            try:
                stream_response = json.loads(data)
            except json.JSONDecodeError:
                stream_response = {'data': data}
            stream = self._parse_stream_response(stream_response)

            if not start:
                start_stream = Stream(delta='', control='start')
                yield ChatCompletionModelStreamOutput(
                    chat_model_id=self.model_id,
                    messages=[AssistantMessage(content=reply)],
                    stream=start_stream,
                )
                if stream.control == 'start':
                    stream.control = 'continue'
                start = True
            if stream.delta:
                reply += stream.delta

            if isinstance(stream, FinishStream):
                finish = True
                yield ChatCompletionModelStreamOutput(
                    chat_model_id=self.model_id,
                    messages=[AssistantMessage(content=reply)],
                    extra={'http_response': stream_response},
                    stream=stream,
                    finish_reason=stream.finish_reason,
                    usage=stream.usage,
                    cost=stream.cost,
                )
                break
            else:
                yield ChatCompletionModelStreamOutput(
                    chat_model_id=self.model_id,
                    messages=[AssistantMessage(content=reply)],
                    extra={'http_response': stream_response},
                    stream=stream,
                )

        if not finish:
            raise UnexpectedResponseError(stream_response, 'Stream is not finished.')

    async def _async_generate_data_from_sse_stream(self, messages: Messages, parameters: P) -> AsyncGenerator[str, None]:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            http_parameters = self._get_stream_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            async with aconnect_sse(client=client, method='POST', **http_parameters) as event_source:
                async for sse in event_source.aiter_sse():
                    yield sse.data

    async def _async_generate_data_from_basic_stream(self, messages: Messages, parameters: P) -> AsyncGenerator[str, None]:
        http_parameters = self._get_stream_request_parameters(messages, parameters)
        http_parameters.update({'timeout': self.timeout})
        async with httpx.AsyncClient(proxies=self.proxies) as client, client.stream('POST', **http_parameters) as source:
            async for line in source.aiter_lines():
                yield line
