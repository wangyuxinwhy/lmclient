from __future__ import annotations

import os

import openai

from lmclient.protocols import CompletionModel


class AzureCompletion(CompletionModel):
    def __init__(
        self,
        engine: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
    ):
        self.engine = engine or os.environ['AZURE_CHAT_API_ENGINE']

        openai.api_type = 'azure'
        openai.api_key = api_key or os.environ['AZURE_API_KEY']
        openai.api_base = api_base or os.environ['AZURE_API_BASE']
        openai.api_version = api_version or os.getenv('AZURE_API_VERSION') or '2023-05-15'

    def complete(self, prompt: str, **kwargs) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(engine=self.engine, messages=messages, **kwargs)
        completion: str = response.choices[0]['message']['content']  # type: ignore
        return completion

    async def async_complete(self, prompt: str, **kwargs) -> str:
        messages = [{'role': 'user', 'content': prompt}]
        response = await openai.ChatCompletion.acreate(engine=self.engine, messages=messages, **kwargs)
        completion: str = response.choices[0]['message']['content']  # type: ignore
        return completion
