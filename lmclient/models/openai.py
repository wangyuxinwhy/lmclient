from __future__ import annotations

import os

import openai

from lmclient.types import ChatModel, Message, Messages


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

    def chat(self, prompt: Messages | str, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = openai.ChatCompletion.create(model=self.model, messages=prompt, **kwargs)
        completion: str = response.choices[0]['message']['content']  # type: ignore
        return completion

    async def async_chat(self, prompt: Messages | str, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = await openai.ChatCompletion.acreate(model=self.model, messages=prompt, **kwargs)
        completion: str = response.choices[0]['message']['content']  # type: ignore
        return completion

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
