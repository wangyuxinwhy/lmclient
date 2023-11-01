from __future__ import annotations

import os
from typing import Any, ClassVar

from typing_extensions import Unpack, override

from lmclient.models.http import HttpChatModel, HttpChatModelKwargs, HttpxPostKwargs
from lmclient.models.openai import (
    OpenAIChatParameters,
    convert_to_openai_message,
    parse_openai_model_reponse,
)
from lmclient.types import Messages, ModelResponse


class AzureChat(HttpChatModel[OpenAIChatParameters]):
    model_type: ClassVar[str] = 'azure'

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        parameters: OpenAIChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ):
        parameters = parameters or OpenAIChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = model or os.environ['AZURE_CHAT_API_ENGINE'] or os.environ['AZURE_CHAT_MODEL_NAME']
        self.system_prompt = system_prompt
        self.api_key = api_key or os.environ['AZURE_API_KEY']
        self.api_base = api_base or os.environ['AZURE_API_BASE']
        self.api_version = api_version or os.getenv('AZURE_API_VERSION')

    @override
    def get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        openai_messages = [convert_to_openai_message(message) for message in messages]
        if self.system_prompt:
            openai_messages.insert(0, {'role': 'system', 'content': self.system_prompt})
        parameters_dict = parameters.model_dump(exclude_defaults=True)
        json_data = {
            'model': self.model,
            'messages': openai_messages,
            **parameters_dict,
        }
        headers = {
            'api-key': self.api_key,
        }
        return {
            'url': f'{self.api_base}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}',
            'headers': headers,
            'json': json_data,
        }

    @override
    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        return parse_openai_model_reponse(response)

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
