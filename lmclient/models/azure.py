from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lmclient.models.http import HttpChatModel, RetryStrategy
from lmclient.models.openai import (
    OpenAIChatParameters,
    convert_lmclient_to_openai,
    parse_openai_model_reponse,
)
from lmclient.types import Messages, ModelResponse
from lmclient.utils import to_dict


class AzureChat(HttpChatModel[OpenAIChatParameters]):
    model_type = 'azure'

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        parameters: OpenAIChatParameters = OpenAIChatParameters(),
        use_cache: Path | str | bool = False,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache)
        self.model = model or os.environ['AZURE_CHAT_API_ENGINE'] or os.environ['AZURE_CHAT_MODEL_NAME']
        self.api_key = api_key or os.environ['AZURE_API_KEY']
        self.api_base = api_base or os.environ['AZURE_API_BASE']
        self.api_version = api_version or os.getenv('AZURE_API_VERSION')

    def get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> dict[str, Any]:
        headers = {
            'api-key': self.api_key,
        }
        parameters_dict = to_dict(parameters, exclude_defaults=True)
        openai_messages = [convert_lmclient_to_openai(message) for message in messages]
        params = {
            'model': self.model,
            'messages': openai_messages,
            **parameters_dict,
        }
        return {
            'url': f'{self.api_base}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}',
            'headers': headers,
            'json': params,
        }

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        return parse_openai_model_reponse(response)

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
