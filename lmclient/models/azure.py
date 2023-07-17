from __future__ import annotations

import os

import openai

from lmclient.types import ChatModel, Message, Messages


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

    def chat(self, prompt: Messages | str, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = openai.ChatCompletion.create(engine=self.model, messages=prompt, **kwargs)
        try:
            completion: str = response.choices[0]['message']['content']  # type: ignore
        except (KeyError, IndexError):
            raise ValueError(f'Invalid response: {response}')
        return completion

    async def async_chat(self, prompt: Messages | str, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]
        if self.timeout:
            kwargs['request_timeout'] = self.timeout

        response = await openai.ChatCompletion.acreate(engine=self.model, messages=prompt, **kwargs)
        try:
            completion: str = response.choices[0]['message']['content']  # type: ignore
        except (KeyError, IndexError):
            raise ValueError(f'Invalid response: {response}')
        return completion

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
