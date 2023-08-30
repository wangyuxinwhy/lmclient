from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Generic, TypeVar, cast

import diskcache
import httpx

try:
    from pydantic.v1 import BaseModel
except ImportError:
    from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from lmclient.parser import ModelResponseParser
from lmclient.types import ChatModelOutput, Messages, ModelResponse, Prompt
from lmclient.utils import ensure_messages
from lmclient.version import __cache_version__

T = TypeVar('T')
DEFAULT_CACHE_DIR = Path(os.getenv('LMCLIENT_CACHE_DIR', '~/.cache/lmclient')).expanduser().resolve()


class BaseChatModel(Generic[T]):
    _cache: diskcache.Cache | None
    _cache_dir: Path | None

    def __init__(
        self,
        response_parser: ModelResponseParser[T] | None = None,
        use_cache: Path | str | bool = False,
    ) -> None:
        self.response_parser = response_parser

        if isinstance(use_cache, (str, Path)):
            self.cache_dir = Path(use_cache)
        elif use_cache:
            self.cache_dir = DEFAULT_CACHE_DIR
        else:
            self.cache_dir = None

    @property
    def identifier(self) -> str:
        raise NotImplementedError

    def call_model(self, messages: Messages, **kwargs) -> ModelResponse:
        raise NotImplementedError

    async def async_call_model(self, messages: Messages, **kwargs) -> ModelResponse:
        raise NotImplementedError

    def chat(self, prompt: Prompt, **kwargs) -> ChatModelOutput[T]:
        messages = ensure_messages(prompt)

        if self.use_cache:
            hash_key = self.generate_hash_key(prompt)
            model_response = self.try_load_response(hash_key)
            if model_response is None:
                model_response = self.call_model(messages, **kwargs)
                self.cache_response(hash_key, model_response)
        else:
            model_response = self.call_model(messages, **kwargs)

        if self.response_parser is None:
            parsed_result = None
        else:
            parsed_result = self.response_parser(model_response)

        return ChatModelOutput(
            parsed_result=parsed_result,
            response=model_response,
        )

    async def async_chat(self, prompt: Prompt, **kwargs) -> ChatModelOutput[T]:
        messages = ensure_messages(prompt)

        if self.use_cache:
            hash_key = self.generate_hash_key(prompt)
            model_response = self.try_load_response(hash_key)
            if model_response is None:
                model_response = await self.async_call_model(messages, **kwargs)
                self.cache_response(hash_key, model_response)
        else:
            model_response = await self.async_call_model(messages, **kwargs)

        if self.response_parser is None:
            parsed_result = None
        else:
            parsed_result = self.response_parser(model_response)

        return ChatModelOutput(
            parsed_result=parsed_result,
            response=model_response,
        )

    def cache_response(self, key: str, response: ModelResponse) -> None:
        if self._cache is not None:
            self._cache[key] = response
        else:
            raise RuntimeError('Cache is not enabled')

    def try_load_response(self, key: str):
        if self._cache is not None and key in self._cache:
            response = self._cache[key]
            response = cast(ModelResponse, response)
            return response

    def generate_hash_key(self, prompt: Prompt, **kwargs) -> str:
        if isinstance(prompt, str):
            hash_text = prompt
        else:
            hash_text = '---'.join([f'{k}={v}' for message in prompt for k, v in message.items()])
        items = sorted([f'{key}={value}' for key, value in kwargs.items()])
        items += [f'__cache_version__={__cache_version__}']
        items = [hash_text, self.identifier] + items
        task_string = '---'.join(items)
        return self.md5_hash(task_string)

    @staticmethod
    def md5_hash(string: str):
        return hashlib.md5(string.encode()).hexdigest()

    @property
    def use_cache(self) -> bool:
        return self._cache is not None

    @property
    def cache_dir(self) -> Path | None:
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: Path | None) -> None:
        if value is not None:
            if value.exists() and not value.is_dir():
                raise ValueError(f'Cache directory {value} is not a directory')
            value.mkdir(parents=True, exist_ok=True)
            self._cache = diskcache.Cache(value)
        else:
            self._cache = None


class RetryStrategy(BaseModel):  # type: ignore
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpChatModel(BaseChatModel[T]):
    def __init__(
        self,
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        response_parser: ModelResponseParser[T] | None = None,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(response_parser=response_parser, use_cache=use_cache)
        self.timeout = timeout
        if isinstance(retry, RetryStrategy):
            self.retry_strategy = retry
        else:
            self.retry_strategy = RetryStrategy() if retry else None

    def get_post_parameters(self, messages: Messages, **kwargs) -> dict[str, Any]:
        raise NotImplementedError

    def call_model(self, messages: Messages, **kwargs) -> ModelResponse:
        parameters = self.get_post_parameters(messages, **kwargs)
        parameters = {'timeout': self.timeout, **parameters}
        http_response = httpx.post(**parameters)
        http_response.raise_for_status()
        model_response = http_response.json()
        return model_response

    async def async_call_model(self, messages: Messages, **kwargs) -> ModelResponse:
        async with httpx.AsyncClient() as client:
            parameters = self.get_post_parameters(messages, **kwargs)
            parameters = {'timeout': self.timeout, **parameters}
            http_response = await client.post(**parameters)
        http_response.raise_for_status()
        model_response = http_response.json()
        return model_response

    def chat(self, prompt: Prompt, **kwargs) -> ChatModelOutput[T]:
        if self.retry_strategy is None:
            return super().chat(prompt, **kwargs)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = retry(wait=wait, stop=stop)(super().chat)(prompt=prompt, **kwargs)
        return output

    async def async_chat(self, prompt: Prompt, **kwargs) -> ChatModelOutput[T]:
        if self.retry_strategy is None:
            return await super().async_chat(prompt, **kwargs)

        wait = wait_random_exponential(min=self.retry_strategy.min_wait_seconds, max=self.retry_strategy.max_wait_seconds)
        stop = stop_after_attempt(self.retry_strategy.max_attempt)
        output = await retry(wait=wait, stop=stop)(super().async_chat)(prompt=prompt, **kwargs)
        return output
