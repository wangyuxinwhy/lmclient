from __future__ import annotations

import os

import httpx

from lmclient.models.base import BaseChatModel
from lmclient.types import ModelResponse, Prompt
from lmclient.utils import ensure_messages


class AzureChat(BaseChatModel):
    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        timeout: int | None = 60,
    ):
        self.model_name = model_name or os.environ['AZURE_CHAT_API_ENGINE'] or os.environ['AZURE_CHAT_MODEL_NAME']
        self.api_key = api_key or os.environ['AZURE_API_KEY']
        self.api_base = api_base or os.environ['AZURE_API_BASE']
        self.api_version = api_version or os.getenv('AZURE_API_VERSION')
        self.timeout = timeout

    def chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)
        headers = {
            'api-key': self.api_key,
        }
        params = {
            'model': self.model_name,
            'messages': messages,
            **kwargs,
        }
        response = httpx.post(
            url=f'{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}',
            headers=headers,
            json=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    async def async_chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)
        headers = {
            'api-key': self.api_key,
        }
        params = {
            'model': self.model_name,
            'messages': messages,
            **kwargs,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f'{self.api_base}/openai/deployments/{self.model_name}/chat/completions?api-version={self.api_version}',
                headers=headers,
                json=params,
                timeout=self.timeout,
            )
        response.raise_for_status()
        return response.json()

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model_name})'
