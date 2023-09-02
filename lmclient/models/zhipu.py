from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, TypedDict, TypeVar

import cachetools.func  # type: ignore
import jwt

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, RetryStrategy
from lmclient.parser import ParserError
from lmclient.types import GeneralParameters, Message, Messages, ModelParameters, ModelResponse

T = TypeVar('T')
API_TOKEN_TTL_SECONDS = 3 * 60
CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30


class ZhiPuChatParameters(ModelParameters):
    temperature: float = 1
    top_p: float = 1

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
        )


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

    return jwt.encode(  # type: ignore
        payload,
        secret,
        algorithm='HS256',
        headers={'alg': 'HS256', 'sign_type': 'SIGN'},
    )


class ZhiPuChat(HttpChatModel[ZhiPuChatParameters]):
    model_type = 'zhipu'

    def __init__(
        self,
        model: str = 'chatglm_pro',
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        parameters: ZhiPuChatParameters = ZhiPuChatParameters(),
        use_cache: Path | str | bool = False,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache)
        self.model = model
        self.api_key = api_key or os.environ['ZHIPU_API_KEY']
        self.api_base = api_base or os.getenv('ZHIPU_API_BASE') or 'https://open.bigmodel.cn/api/paas/v3/model-api'
        self.api_base.rstrip('/')

    def get_request_parameters(self, messages: Messages, parameters: ZhiPuChatParameters) -> dict[str, Any]:
        for message in messages:
            if message.role not in ('user', 'assistant'):
                raise ValueError(f'Role of message must be user or assistant, but got {message.role}')

        zhipu_messages: list[ZhiPuMessageDict] = []
        for message in messages:
            if message.role not in ('user', 'assistant'):
                raise MessageError(f'Role of message must be user or assistant, but got {message.role}')
            if not isinstance(message.content, str):
                raise MessageError(f'Message content must be str, but got {type(message.content)}')
            zhipu_messages.append(
                {
                    'role': message.role,
                    'content': message.content,
                }
            )
        headers = {
            'Authorization': generate_token(self.api_key),
        }
        parameters_dict = parameters.model_dump(exclude_defaults=True)
        params = {'prompt': zhipu_messages, **parameters_dict}
        return {
            'url': f'{self.api_base}/{self.model}/invoke',
            'headers': headers,
            'json': params,
        }

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        try:
            text = response['data']['choices'][0]['content'].strip('"').strip()
            return [Message(role='assistant', content=text)]
        except (KeyError, IndexError) as e:
            raise ParserError(f'Parse response failed, reponse: {response}') from e

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
