from __future__ import annotations

import os

import openai

from lmclient.protocols import CompletionModel


class OpenAICompletion(CompletionModel):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ):
        self.model = model

        openai.api_type = 'open_ai'
        openai.api_base = 'https://api.openai.com/v1'
        openai.api_key = api_key or os.environ['OPENAI_API_KEY']
        openai.api_version = None

    def complete(self, prompt: str, **kwargs) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(model=self.model, messages=messages, **kwargs)
        completion: str = response.choices[0]['message']['content']  # type: ignore
        return completion

    async def async_complete(self, prompt: str, **kwargs) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        response = await openai.ChatCompletion.acreate(model=self.model, messages=messages, **kwargs)
        completion: str = response.choices[0]['message']['content']  # type: ignore
        return completion

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
