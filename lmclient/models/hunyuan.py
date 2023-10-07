from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import uuid
from email.errors import MessageError
from pathlib import Path
from typing import Any, Literal, Optional

from typing_extensions import Self, TypedDict

from lmclient.exceptions import ResponseError
from lmclient.models.http import HttpChatModel, ModelResponse, ProxiesTypes, RetryStrategy
from lmclient.types import GeneralParameters, Message, Messages, ModelParameters


class HunyuanMessageDict(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class HunyuanChatParameters(ModelParameters):
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
        )


class HunyuanChat(HttpChatModel[HunyuanChatParameters]):
    model_type = 'hunyuan'
    default_sign_api: str = 'hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'
    default_api: str = 'https://hunyuan.cloud.tencent.com/hyllm/v1/chat/completions'

    def __init__(
        self,
        app_id: int | None = None,
        secret_id: str | None = None,
        secret_key: str | None = None,
        api: str | None = None,
        sign_api: str | None = None,
        parameters: HunyuanChatParameters = HunyuanChatParameters(),
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        proxies: ProxiesTypes | None = None,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache, proxies=proxies)
        self.app_id = app_id or int(os.environ['HUNYUAN_APP_ID'])
        self.secret_id = secret_id or os.environ['HUNYUAN_SECRET_ID']
        self.secret_key = secret_key or os.environ['HUNYUAN_SECRET_KEY']
        self.api = api or self.default_api
        self.sign_api = sign_api or self.default_sign_api

    def get_request_parameters(self, messages: Messages, parameters: HunyuanChatParameters) -> dict[str, Any]:
        message_dicts: list[HunyuanMessageDict] = []
        for message in messages:
            role = message.role
            if role != 'assistant' and role != 'user':
                raise MessageError(f'Invalid message role: {role}, only "user" and "assistant" are allowed')
            if not isinstance(content := message.content, str):
                raise MessageError(f'Invalid message content: {content}, only string is allowed')
            message_dicts.append(HunyuanMessageDict(content=content, role=role))

        json_dict = self.generate_json_dict(message_dicts, parameters)
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

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        if response.get('error'):
            raise ResponseError(f'code: {response["error"]["code"]}, message: {response["error"]["message"]}')
        return [Message(role='assistant', content=response['choices'][0]['messages']['content'])]

    def generate_json_dict(self, messages: list[HunyuanMessageDict], parameters: HunyuanChatParameters):
        timestamp = int(time.time()) + 10000
        json_dict: dict[str, Any] = {
            'app_id': self.app_id,
            'secret_id': self.secret_id,
            'query_id': 'query_id_' + str(uuid.uuid4()),
            'messages': messages,
            'timestamp': timestamp,
            'expired': timestamp + 24 * 60 * 60,
            'stream': 0,
        }
        if self.parameters.temperature is not None:
            json_dict['temperature'] = parameters.temperature
        if self.parameters.top_p is not None:
            json_dict['top_p'] = parameters.top_p
        return json_dict

    @staticmethod
    def generate_sign_parameters(json_dict: dict[str, Any]):
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

    def generate_signature(self, sign_parameters: dict[str, Any]):
        sort_dict = sorted(sign_parameters.keys())
        sign_str = self.default_sign_api + '?'
        for key in sort_dict:
            sign_str = sign_str + key + '=' + str(sign_parameters[key]) + '&'
        sign_str = sign_str[:-1]
        hmacstr = hmac.new(self.secret_key.encode('utf-8'), sign_str.encode('utf-8'), hashlib.sha1).digest()
        signature = base64.b64encode(hmacstr)
        signature = signature.decode('utf-8')
        return signature

    @property
    def name(self) -> str:
        return 'v1'

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        if name != 'v1':
            raise ValueError('Unknown name: {}, only support v1'.format(name))
        return cls(**kwargs)
