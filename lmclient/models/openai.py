from __future__ import annotations

import os
from typing import cast

import openai
from openai.openai_object import OpenAIObject

from lmclient.exceptions import PostProcessError
from lmclient.types import ChatModel, Message, Messages, ModelResponse


class OpenAIChat(ChatModel):
    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
    ):
        self.model = model_name

        openai.api_type = 'open_ai'
        openai.api_base = 'https://api.openai.com/v1'
        openai.api_key = api_key or os.environ['OPENAI_API_KEY']
        openai.api_version = None
        self.timeout = None

    def chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = openai.ChatCompletion.create(model=self.model, messages=prompt, **kwargs)
        response = cast(OpenAIObject, response)
        return response.to_dict_recursive()

    async def async_chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = await openai.ChatCompletion.acreate(model=self.model, messages=prompt, **kwargs)
        response = cast(OpenAIObject, response)
        return response.to_dict_recursive()

    @staticmethod
    def default_postprocess_function(response: ModelResponse) -> str:
        try:
            output = response['choices'][0]['message']['content']  # type: ignore
        except (KeyError, IndexError) as e:
            raise PostProcessError('Parse response failed') from e
        return output

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
