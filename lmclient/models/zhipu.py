from __future__ import annotations

import os
import time

import cachetools.func  # type: ignore
import httpx
import jwt

from lmclient.models.base import BaseChatModel
from lmclient.types import ModelResponse, Prompt
from lmclient.utils import ensure_messages

API_TOKEN_TTL_SECONDS = 3 * 60
CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30


@cachetools.func.ttl_cache(maxsize=10, ttl=CACHE_TTL_SECONDS)
def generate_token(api_key: str):
    try:
        api_key, secret = api_key.split('.')
    except Exception as e:
        raise ValueError('invalid api_key') from e

    payload = {
        'api_key': api_key,
        'exp': int(round(time.time() * 1000)) + API_TOKEN_TTL_SECONDS * 1000,
        'timestamp': int(round(time.time() * 1000)),
    }

    return jwt.encode(
        payload,
        secret,
        algorithm='HS256',
        headers={'alg': 'HS256', 'sign_type': 'SIGN'},
    )


class ZhiPuChat(BaseChatModel):
    def __init__(
        self, model_name: str = 'chatglm_pro', api_base: str | None = None, api_key: str | None = None, timeout: int | None = 60
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ['ZHIPU_API_KEY']
        self.timeout = timeout
        self.api_base = api_base or os.getenv('ZHIPU_API_BASE') or 'https://open.bigmodel.cn/api/paas/v3/model-api'
        self.api_base.rstrip('/')

    def chat(
        self, prompt: Prompt, temperature: float | None = None, top_p: float | None = None, request_id: str | None = None
    ) -> ModelResponse:
        messages = ensure_messages(prompt)

        headers = {
            'Authorization': generate_token(self.api_key),
        }
        params = {'prompt': messages, 'temperature': temperature, 'top_p': top_p, 'request_id': request_id}
        reponse = httpx.post(
            url=f'{self.api_base}/{self.model_name}/invoke',
            headers=headers,
            json=params,
            timeout=self.timeout,
        )
        reponse.raise_for_status()
        return reponse.json()

    async def async_chat(
        self, prompt: Prompt, temperature: float | None = None, top_p: float | None = None, request_id: str | None = None
    ) -> ModelResponse:
        messages = ensure_messages(prompt)

        headers = {
            'Authorization': generate_token(self.api_key),
        }
        params = {'prompt': messages, 'temperature': temperature, 'top_p': top_p, 'request_id': request_id}
        async with httpx.AsyncClient() as client:
            reponse = await client.post(
                url=f'{self.api_base}/{self.model_name}/invoke',
                headers=headers,
                json=params,
                timeout=self.timeout,
            )
        reponse.raise_for_status()
        return reponse.json()

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model_name})'
