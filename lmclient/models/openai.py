from __future__ import annotations

import os
from typing import cast

import openai
from openai.openai_object import OpenAIObject

from lmclient.models.base import BaseChatModel
from lmclient.types import ModelResponse, Prompt
from lmclient.utils import ensure_messages


class OpenAIChat(BaseChatModel):
    def __init__(
        self,
        model_name: str = 'gpt-3.5-turbo',
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout: int | None = 60,
    ):
        self.model = model_name

        openai.api_type = 'open_ai'
        openai.api_base = api_base or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
        openai.api_key = api_key or os.environ['OPENAI_API_KEY']
        openai.api_version = api_version
        self.timeout = timeout

    def chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = openai.ChatCompletion.create(model=self.model, messages=messages, **kwargs)
        response = cast(OpenAIObject, response)
        return response.to_dict_recursive()

    async def async_chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = await openai.ChatCompletion.acreate(model=self.model, messages=messages, **kwargs)
        response = cast(OpenAIObject, response)
        return response.to_dict_recursive()

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
