from __future__ import annotations

import os
from typing import cast

import openai
from openai.openai_object import OpenAIObject

from lmclient.types import ChatModel, Message, Messages, ModelResponse


class AzureChat(ChatModel):
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
    ):
        self.model = model or os.environ['AZURE_CHAT_API_ENGINE']

        openai.api_type = 'azure'
        openai.api_key = api_key or os.environ['AZURE_API_KEY']
        openai.api_base = api_base or os.environ['AZURE_API_BASE']
        openai.api_version = api_version or os.getenv('AZURE_API_VERSION') or '2023-05-15'
        self.timeout = None

    def chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = openai.ChatCompletion.create(engine=self.model, messages=prompt, **kwargs)
        response = cast(OpenAIObject, response)
        return response.to_dict_recursive()

    async def async_chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = await openai.ChatCompletion.acreate(engine=self.model, messages=prompt, **kwargs)
        response = cast(OpenAIObject, response)
        return response.to_dict_recursive()

    @staticmethod
    def default_postprocess_function(response: ModelResponse) -> ModelResponse:
        try:
            response['content'] = response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            response['content'] = 'Error Response'
        return response

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
