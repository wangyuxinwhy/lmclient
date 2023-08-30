from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

from lmclient.models.base import HttpChatModel, RetryStrategy
from lmclient.models.openai import OpenAIContentParser
from lmclient.parser import ModelResponseParser
from lmclient.types import Messages

T = TypeVar('T')


class AzureChat(HttpChatModel[T]):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout: int | None = 60,
        response_parser: ModelResponseParser[T] | None = None,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
    ):
        response_parser = response_parser or OpenAIContentParser()
        super().__init__(timeout=timeout, response_parser=response_parser, retry=retry, use_cache=use_cache)
        self.model = model or os.environ['AZURE_CHAT_API_ENGINE'] or os.environ['AZURE_CHAT_MODEL_NAME']
        self.api_key = api_key or os.environ['AZURE_API_KEY']
        self.api_base = api_base or os.environ['AZURE_API_BASE']
        self.api_version = api_version or os.getenv('AZURE_API_VERSION')

    def get_post_parameters(self, messages: Messages, **kwargs) -> dict[str, Any]:
        headers = {
            'api-key': self.api_key,
        }
        params = {
            'model': self.model,
            'messages': messages,
            **kwargs,
        }
        return {
            'url': f'{self.api_base}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}',
            'headers': headers,
            'json': params,
        }

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
