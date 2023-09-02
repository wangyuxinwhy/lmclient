from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_random_exponential

from lmclient.models.base import T_P, BaseChatModel
from lmclient.types import HttpChatModelOutput, Messages, ModelResponse, RetryStrategy

logger = logging.getLogger(__name__)


class HttpChatModel(BaseChatModel[T_P, HttpChatModelOutput], ABC):
    model_type = 'http'

    def __init__(
        self,
        parameters: T_P,
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(parameters=parameters, use_cache=use_cache)
        self.timeout = timeout
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None

    @abstractmethod
    def get_request_parameters(self, messages: Messages, parameters: T_P) -> dict[str, Any]:
        ...

    @abstractmethod
    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        ...

    def _chat_completion_without_retry(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput:
        http_parameters = self.get_request_parameters(messages, parameters)
        http_parameters = {'timeout': self.timeout, **http_parameters}
        logger.info(f'HTTP Request: {http_parameters}')
        http_response = httpx.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_response = http_response.json()
        logger.info(f'HTTP Response: {model_response}')
        new_messages = self.parse_model_reponse(model_response)
        reply = new_messages[-1].content
        reply = reply if isinstance(reply, str) else ''
        return HttpChatModelOutput(
            messages=new_messages,
            response=model_response,
            reply=reply,
        )

    async def _async_chat_completion_without_retry(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput:
        async with httpx.AsyncClient() as client:
            http_parameters = self.get_request_parameters(messages, parameters)
            http_parameters = {'timeout': self.timeout, **http_parameters}
            logger.info(f'ASYNC HTTP Request: {http_parameters}')
            http_response = await client.post(**http_parameters)  # type: ignore
        http_response.raise_for_status()
        model_response = http_response.json()
        new_messages = self.parse_model_reponse(model_response)
        reply = new_messages[-1].content
        reply = reply if isinstance(reply, str) else ''
        return HttpChatModelOutput(
            messages=new_messages,
            response=model_response,
            reply=reply,
        )

    def _chat_completion(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput:
        if self.retry_strategy is None:
            return self._chat_completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = retry(wait=wait, stop=stop)(self._chat_completion_without_retry)(messages, parameters)
        return output

    async def _async_chat_completion(self, messages: Messages, parameters: T_P) -> HttpChatModelOutput:
        if self.retry_strategy is None:
            return await self._async_chat_completion_without_retry(messages, parameters)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = await retry(wait=wait, stop=stop)(self._async_chat_completion_without_retry)(messages, parameters)
        return output
