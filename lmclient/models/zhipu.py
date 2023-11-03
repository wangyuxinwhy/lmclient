from __future__ import annotations

import os
import time
from typing import Any, ClassVar, Literal, Optional

import cachetools.func  # type: ignore
import jwt
from typing_extensions import NotRequired, TypedDict, Unpack, override

from lmclient.exceptions import MessageError, UnexpectedResponseError
from lmclient.models.http import HttpChatModel, HttpChatModelKwargs, HttpxPostKwargs
from lmclient.types import T_P, Message, Messages, ModelParameters, ModelResponse, Probability, Stream, Temperature, TextMessage
from lmclient.utils import is_text_message

API_TOKEN_TTL_SECONDS = 3 * 60
CACHE_TTL_SECONDS = API_TOKEN_TTL_SECONDS - 30


class ZhiPuRef(TypedDict):
    enable: NotRequired[bool]
    search_query: NotRequired[str]


class ZhiPuChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    request_id: Optional[str] = None
    ref: Optional[ZhiPuRef] = None


class ZhiPuMeta(TypedDict):
    user_info: str
    bot_info: str
    bot_name: str
    user_name: str


class ZhiPuCharacterChatParameters(ModelParameters):
    meta: ZhiPuMeta = {
        'user_info': '我是陆星辰，是一个男性，是一位知名导演，也是苏梦远的合作导演。',
        'bot_info': '苏梦远，本名苏远心，是一位当红的国内女歌手及演员。',
        'bot_name': '苏梦远',
        'user_name': '陆星辰',
    }
    request_id: Optional[str] = None


class ZhiPuMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


def convert_to_zhipu_message(message: Message) -> ZhiPuMessage:
    if not is_text_message(message):
        raise MessageError(f'invalid message type: {type(message)}, only TextMessage is allowed')
    role = message['role']
    if role != 'assistant' and role != 'user':
        raise MessageError(f'invalid message role: {role}, only "user" and "assistant" are allowed')

    return {
        'role': role,
        'content': message['content'],
    }


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


class BaseZhiPuChat(HttpChatModel[T_P]):
    default_api_base: ClassVar[str] = 'https://open.bigmodel.cn/api/paas/v3/model-api'

    def __init__(
        self,
        model: str,
        parameters: T_P,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ):
        super().__init__(parameters=parameters, **kwargs)
        self.model = model
        self.api_key = api_key or os.environ['ZHIPU_API_KEY']
        self.api_base = (api_base or self.default_api_base).rstrip('/')

    @override
    def get_request_parameters(self, messages: Messages, parameters: T_P) -> HttpxPostKwargs:
        zhipu_messages = [convert_to_zhipu_message(message) for message in messages]
        headers = {
            'Authorization': generate_token(self.api_key),
        }
        parameters_dict = parameters.model_dump(exclude_none=True)
        params = {'prompt': zhipu_messages, **parameters_dict}
        return {
            'url': f'{self.api_base}/{self.model}/invoke',
            'headers': headers,
            'json': params,
        }

    @override
    def parse_reponse(self, response: ModelResponse) -> Messages:
        if response['success']:
            text = response['data']['choices'][0]['content']
            return [TextMessage(role='assistant', content=text)]
        else:
            raise UnexpectedResponseError(response)

    @override
    def get_stream_request_parameters(self, messages: Messages, parameters: T_P) -> HttpxPostKwargs:
        http_parameters = self.get_request_parameters(messages, parameters)
        http_parameters['url'] = f'{self.api_base}/{self.model}/sse-invoke'
        return http_parameters

    @override
    def parse_stream_response(self, response: ModelResponse) -> Stream:
        if response['data']:
            return Stream(delta=response['data'], control='continue')
        return Stream(delta='', control='finish')

    @override
    def _preprocess_stream_data(self, stream_data: str) -> ModelResponse:
        return {'data': stream_data}

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)


class ZhiPuChat(BaseZhiPuChat[ZhiPuChatParameters]):
    model_type: ClassVar[str] = 'zhipu'

    def __init__(
        self,
        model: str = 'characterglm',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: ZhiPuChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ):
        parameters = parameters or ZhiPuChatParameters()
        super().__init__(model=model, api_key=api_key, api_base=api_base, parameters=parameters, **kwargs)


class ZhiPuCharacterChat(BaseZhiPuChat[ZhiPuCharacterChatParameters]):
    model_type: ClassVar[str] = 'zhipu-character'

    def __init__(
        self,
        model: str = 'characterglm',
        api_key: str | None = None,
        api_base: str | None = None,
        parameters: ZhiPuCharacterChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ):
        parameters = parameters or ZhiPuCharacterChatParameters()
        super().__init__(model=model, api_key=api_key, api_base=api_base, parameters=parameters, **kwargs)
