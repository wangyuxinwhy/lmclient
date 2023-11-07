from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import uuid
from email.errors import MessageError
from typing import Any, ClassVar, Literal, Optional

from typing_extensions import Self, TypedDict, Unpack, override

from lmclient.exceptions import UnexpectedResponseError
from lmclient.models.http import HttpChatModel, HttpChatModelKwargs, HttpxPostKwargs, ModelResponse, Stream
from lmclient.types import Message, Messages, ModelParameters, Probability, Temperature, TextMessage
from lmclient.utils import is_text_message


class HunyuanMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class HunyuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None


def convert_to_hunyuan_message(message: Message) -> HunyuanMessage:
    if not is_text_message(message):
        raise MessageError(f'Invalid message type: {type(message)}, only TextMessage is allowed')
    role = message['role']
    if role not in ('assistant', 'user'):
        raise MessageError(f'Invalid message role: {role}, only "user" and "assistant" are allowed')

    return {
        'role': role,
        'content': message['content'],
    }


class HunyuanChat(HttpChatModel[HunyuanChatParameters]):
    model_type: ClassVar[str] = 'hunyuan'
    # stream_model = 'basic'
    default_api: ClassVar[str] = 'https://hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'
    default_sign_api: ClassVar[str] = 'hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'

    def __init__(
        self,
        app_id: int | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        api: str | None = None,
        sign_api: str | None = None,
        parameters: HunyuanChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ) -> None:
        parameters = parameters or HunyuanChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.app_id = app_id or int(os.environ['HUNYUAN_APP_ID'])
        self.secret_id = secret_id or os.environ['HUNYUAN_SECRET_ID']
        self.secret_key = secret_key or os.environ['HUNYUAN_SECRET_KEY']
        self.api = api or self.default_api
        self.sign_api = sign_api or self.default_sign_api

    @override
    def _get_request_parameters(self, messages: Messages, parameters: HunyuanChatParameters) -> HttpxPostKwargs:
        hunyuan_messages = [convert_to_hunyuan_message(message) for message in messages]
        json_dict = self.generate_json_dict(hunyuan_messages, parameters)
        signature = self.generate_signature(self.generate_sign_parameters(json_dict))
        headers = {
            'Content-Type': 'application/json',
            'Authorization': signature,
        }
        return {
            'url': self.api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: HunyuanChatParameters) -> HttpxPostKwargs:
        hunyuan_messages = [convert_to_hunyuan_message(message) for message in messages]
        json_dict = self.generate_json_dict(hunyuan_messages, parameters, stream=True)
        signature = self.generate_signature(self.generate_sign_parameters(json_dict))
        headers = {
            'Content-Type': 'application/json',
            'Authorization': signature,
        }
        return {
            'url': self.api,
            'headers': headers,
            'json': json_dict,
        }

    @override
    def _parse_stream_response(self, response: ModelResponse) -> Stream:
        message = response['choices'][0]
        if message['finish_reason']:
            return Stream(delta=message['delta']['content'], control='finish')
        return Stream(delta=message['delta']['content'], control='continue')

    @override
    def _parse_reponse(self, response: ModelResponse) -> Messages:
        if response.get('error'):
            raise UnexpectedResponseError(response)
        return [TextMessage(role='assistant', content=response['choices'][0]['messages']['content'])]

    def generate_json_dict(
        self, messages: list[HunyuanMessage], parameters: HunyuanChatParameters, stream: bool = False
    ) -> dict[str, Any]:
        timestamp = int(time.time()) + 10000
        json_dict: dict[str, Any] = {
            'app_id': self.app_id,
            'secret_id': self.secret_id,
            'query_id': 'query_id_' + str(uuid.uuid4()),
            'messages': messages,
            'timestamp': timestamp,
            'expired': timestamp + 24 * 60 * 60,
            'stream': int(stream),
        }
        json_dict.update(parameters.model_dump(exclude_none=True))
        return json_dict

    @staticmethod
    def generate_sign_parameters(json_dict: dict[str, Any]) -> dict[str, Any]:
        params = {
            'app_id': json_dict['app_id'],
            'secret_id': json_dict['secret_id'],
            'query_id': json_dict['query_id'],
            'stream': json_dict['stream'],
        }
        if 'temperature' in json_dict:
            params['temperature'] = f'{json_dict["temperature"]:g}'
        if 'top_p' in json_dict:
            params['top_p'] = f'{json_dict["top_p"]:g}'
        message_str = ','.join(
            ['{{"role":"{}","content":"{}"}}'.format(message['role'], message['content']) for message in json_dict['messages']]
        )
        message_str = '[{}]'.format(message_str)
        params['messages'] = message_str
        params['timestamp'] = str(json_dict['timestamp'])
        params['expired'] = str(json_dict['expired'])
        return params

    def generate_signature(self, sign_parameters: dict[str, Any]) -> str:
        sort_dict = sorted(sign_parameters.keys())
        sign_str = self.default_sign_api + '?'
        for key in sort_dict:
            sign_str = sign_str + key + '=' + str(sign_parameters[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha1).digest()
        signature = base64.b64encode(hmacstr)
        return signature.decode('utf-8')

    @property
    @override
    def name(self) -> str:
        return 'v1'

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name != 'v1':
            raise ValueError('Unknown name: {}, only support v1'.format(name))
        return cls(**kwargs)
