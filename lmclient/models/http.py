from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, Mapping, Optional, Sequence, Union

import httpx
from httpx._types import ProxiesTypes
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import Required, TypedDict, override

from lmclient.cache import BaseCache
from lmclient.models.base import T_P, BaseChatModel, OverrideParameters
from lmclient.types import ChatModelOutput, Messages, ModelResponse, PrimitiveData, Prompt

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
    cache: Union[Path, str, bool, BaseCache[HttpChatModelOutput]]


class HttpxPostKwargs(TypedDict, total=False):
    url: Required[str]
    json: Any
    params: QueryParams
    headers: Headers
    timeout: Optional[int]


class HttpChatModelOutput(ChatModelOutput[T_P], Generic[T_P]):
    response: ModelResponse
    http_parameters: HttpxPostKwargs


class HttpChatModel(BaseChatModel[T_P, HttpChatModelOutput[T_P]], Generic[T_P], ABC):
    model_type = 'http'

    def __init__(
        self,
        parameters: T_P,
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        proxies: ProxiesTypes | None = None,
        cache: Path | str | bool | BaseCache[HttpChatModelOutput] = False,
    ):
        super().__init__(parameters=parameters, cache=cache)
        self.timeout = timeout
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None
        self.proxies = proxies

    @abstractmethod
    def get_request_parameters(self, messages: Messages, parameters: T_P) -> HttpxPostKwargs:
        ...

    @abstractmethod
    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        ...

    def chat_completion(
        self, prompt: Prompt, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> HttpChatModelOutput[T_P]:
        return super().chat_completion(prompt, override_parameters, **kwargs)

    def _chat_completion_without_retry(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput[T_P]:
        with httpx.Client(proxies=self.proxies) as client:
            http_parameters = self.get_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            logger.info(f'HTTP Request: {http_parameters}')
            http_response = client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_response = http_response.json()
        logger.info(f'HTTP Response: {model_response}')
        new_messages = self.parse_model_reponse(model_response)
        reply = new_messages[-1]['content']
        reply = reply if isinstance(reply, str) else ''
        return HttpChatModelOutput[T_P](
            model_id=self.model_id,
            parameters=parameters.model_copy(),
            http_parameters=http_parameters,
            messages=new_messages,
            response=model_response,
            reply=reply,
        )

    async def _async_chat_completion_without_retry(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput[T_P]:
        async with httpx.AsyncClient(proxies=self.proxies) as client:
            http_parameters = self.get_request_parameters(messages, parameters)
            http_parameters.update({'timeout': self.timeout})
            logger.info(f'Async HTTP Request: {http_parameters}')
            http_response = await client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_response = http_response.json()
        logger.info(f'Async HTTP Response: {model_response}')
        new_messages = self.parse_model_reponse(model_response)
        reply = new_messages[-1]['content']
        reply = reply if isinstance(reply, str) else ''
        return HttpChatModelOutput[T_P](
            model_id=self.model_id,
            parameters=parameters.model_copy(),
            http_parameters=http_parameters,
            messages=new_messages,
            response=model_response,
            reply=reply,
        )

    @override
    def _chat_completion(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput:
        if self.retry_strategy is None:
            return self._chat_completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = retry(wait=wait, stop=stop)(self._chat_completion_without_retry)(messages, parameters)
        return output

    @override
    async def _async_chat_completion(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput:
        if self.retry_strategy is None:
            return await self._async_chat_completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = await retry(wait=wait, stop=stop)(self._async_chat_completion_without_retry)(messages, parameters)
        return output
