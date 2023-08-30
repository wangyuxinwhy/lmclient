from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, TypeVar

import cachetools.func  # type: ignore
import jwt

from lmclient.models.base import HttpChatModel, RetryStrategy
from lmclient.parser import ModelResponseParser, ParserError
from lmclient.types import Messages, ModelResponse

T = TypeVar('T')
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


class ZhiPuParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> str:
        try:
            output = response['data']['choices'][0]['content'].strip('"').strip()
        except (KeyError, IndexError) as e:
            raise ParserError(f'Parse response failed, reponse: {response}') from e
        return output


class ZhiPuChat(HttpChatModel[T]):
    def __init__(
        self,
        model: str = 'chatglm_pro',
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: int | None = 60,
        response_parser: ModelResponseParser[T] | None = None,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
    ) -> None:
        response_parser = response_parser or ZhiPuParser()
        super().__init__(timeout=timeout, response_parser=response_parser, retry=retry, use_cache=use_cache)
        self.model = model
        self.api_key = api_key or os.environ['ZHIPU_API_KEY']
        self.api_base = api_base or os.getenv('ZHIPU_API_BASE') or 'https://open.bigmodel.cn/api/paas/v3/model-api'
        self.api_base.rstrip('/')

    def get_post_parameters(self, messages: Messages, **kwargs) -> dict[str, Any]:
        for message in messages:
            if message['role'] not in ('user', 'assistant'):
                raise ValueError(f'Role of message must be user or assistant, but got {message["role"]}')

        headers = {
            'Authorization': generate_token(self.api_key),
        }
        params = {'prompt': messages, **kwargs}
        return {
            'url': f'{self.api_base}/{self.model}/invoke',
            'headers': headers,
            'json': params,
        }

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
