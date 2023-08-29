from __future__ import annotations

import os

import httpx

from lmclient.models.base import BaseChatModel
from lmclient.types import ModelResponse, Prompt
from lmclient.utils import ensure_messages


class OpenAIChat(BaseChatModel):
    def __init__(
        self,
        model_name: str = 'gpt-3.5-turbo',
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int | None = 60,
    ):
        self.model = model_name
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
        self.api_key = api_key or os.environ['OPENAI_API_KEY']
        self.timeout = timeout

    def chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        params = {
            'model': self.model,
            'messages': messages,
            **kwargs,
        }
        response = httpx.post(
            url=f'{self.api_base}/chat/completions',
            headers=headers,
            json=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    async def async_chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        params = {
            'model': self.model,
            'messages': messages,
            **kwargs,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f'{self.api_base}/chat/completions',
                headers=headers,
                json=params,
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
