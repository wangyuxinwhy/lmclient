from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any, TypedDict, TypeVar

import cachetools.func  # type: ignore
import jwt

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, RetryStrategy
from lmclient.parser import ModelResponseParser, ParserError
from lmclient.types import GeneralParameters, Messages, ModelParameters, ModelResponse
from lmclient.utils import to_dict

T = TypeVar('T')
API_TOKEN_TTL_SECONDS = 3 * 60
CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30
logger = logging.getLogger(__name__)


class ZhiPuChatParameters(ModelParameters):
    temperature: float = 1
    top_p: float = 1

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
        )


class ZhiPuResponse(ModelResponse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content = self.content.replace('\n', ' ')


class ZhiPuModel(HttpChatModel):
    name = 'zhipu'

class ZhiPuMessageDict(TypedDict):
    role: str
    content: str


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


class ZhiPuChat(HttpChatModel[ZhiPuChatParameters]):
    def __init__(
        self,
        model: str = 'chatglm_pro',
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        default_parameters: ZhiPuChatParameters | None = None,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(default_parameters=default_parameters, timeout=timeout, retry=retry, use_cache=use_cache)
        self.model = model
        self.api_key = api_key or os.environ['ZHIPU_API_KEY']
        self.api_base = api_base or os.getenv('ZHIPU_API_BASE') or 'https://open.bigmodel.cn/api/paas/v3/model-api'
        self.api_base.rstrip('/')

    def get_post_parameters(self, messages: Messages, parameters: ZhiPuChatParameters | None = None) -> dict[str, Any]:
        for message in messages:
            if message.role not in ('user', 'assistant'):
                raise ValueError(f'Role of message must be user or assistant, but got {message.role}')

        zhipu_messages: list[ZhiPuMessageDict] = []
        for message in messages:
            if message.role not in ('user', 'assistant'):
                raise MessageError(f'Role of message must be user or assistant, but got {message.role}')
            if not isinstance(message.content, str):
                raise MessageError(f'Message content must be str, but got {type(message.content)}')
            zhipu_messages.append({
                'role': message.role,
                'content': message.content,
            })
        headers = {
            'Authorization': generate_token(self.api_key),
        }
        parameters_dict = {} if parameters is None else to_dict(parameters, exclude_defaults=True)
        params = {'prompt': messages, **parameters_dict}
        return {
            'url': f'{self.api_base}/{self.model}/invoke',
            'headers': headers,
            'json': params,
        }

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
