from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential

from lmclient.models.base import BaseChatModel, T
from lmclient.types import BaseModel, ChatModelOutput, Messages, ModelResponse


class RetryStrategy(BaseModel):  # type: ignore
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpChatModel(BaseChatModel[T], ABC):
    def __init__(
        self,
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        default_parameters: T | None = None,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(default_parameters=default_parameters, use_cache=use_cache)
        self.timeout = timeout
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None

    @abstractmethod
    def get_post_parameters(self, messages: Messages, parameters: T | None = None) -> dict[str, Any]:
        ...

    @abstractmethod
    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        ...

    def _chat_completion(self, messages: Messages, override_parameters: T | None = None) -> ChatModelOutput:
        if self.default_parameters is not None and override_parameters is not None:
            override_parameters = self.default_parameters.model_copy(update=override_parameters.model_dump())

        http_parameters = self.get_post_parameters(messages, override_parameters)
        http_parameters = {'timeout': self.timeout, **http_parameters}
        http_response = httpx.post(**http_parameters)
        http_response.raise_for_status()
        model_response = http_response.json()
        return ChatModelOutput(
            messages=self.parse_model_reponse(model_response),
            response=model_response,
        )

    async def _async_chat_completion(self, messages: Messages, override_parameters: T | None = None) -> ChatModelOutput:
        if self.default_parameters is not None and override_parameters is not None:
            override_parameters = self.default_parameters.model_copy(update=override_parameters.model_dump())

        async with httpx.AsyncClient() as client:
            http_parameters = self.get_post_parameters(messages, override_parameters)
            http_parameters = {'timeout': self.timeout, **http_parameters}
            http_response = await client.post(**http_parameters)
        http_response.raise_for_status()
        model_response = http_response.json()
        return ChatModelOutput(
            messages=self.parse_model_reponse(model_response),
            response=model_response,
        )

    def chat_completion(self, messages: Messages, override_parameters: T | None = None) -> ChatModelOutput:
        if self.retry_strategy is None:
            return self._chat_completion(messages, override_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = retry(wait=wait, stop=stop)(self._chat_completion)(messages, override_parameters)
        return output

    async def async_chat_completion(self, messages: Messages, override_parameters: T | None = None) -> ChatModelOutput:
        if self.retry_strategy is None:
            return await self._async_chat_completion(messages, override_parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = await retry(wait=wait, stop=stop)(self._async_chat_completion)(messages, override_parameters)
        return output
