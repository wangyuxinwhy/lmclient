from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, TypedDict

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, ProxiesTypes, RetryStrategy
from lmclient.parser import ParserError
from lmclient.types import GeneralParameters, Message, Messages, ModelParameters, ModelResponse


class BaichuanChatParameters(ModelParameters):
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    with_search_enhance: Optional[bool] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
        )


class BaichuanMessageDict(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class BaichuanChat(HttpChatModel[BaichuanChatParameters]):
    model_type = 'zhipu'
    default_api_base: ClassVar[str] = 'https://api.baichuan-ai.com/v1/chat'

    def __init__(
        self,
        model: str = 'Baichuan2-53B',
        api_key: str | None = None,
        secret_key: str | None = None,
        api_base: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        parameters: BaichuanChatParameters = BaichuanChatParameters(),
        use_cache: Path | str | bool = False,
        proxies: ProxiesTypes | None = None,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache, proxies=proxies)
        self.model = model
        self.api_key = api_key or os.environ['BAICHUAN_API_KEY']
        self.secret_key = secret_key or os.environ['BAICHUAN_SECRET_KEY']
        self.api_base = api_base or self.default_api_base
        self.api_base.rstrip('/')

    def get_request_parameters(self, messages: Messages, parameters: BaichuanChatParameters) -> dict[str, Any]:
        baichuan_messages: list[BaichuanMessageDict] = []
        for message in messages:
            role = message.role
            if role not in ('user', 'assistant'):
                raise ValueError(f'Role of message must be user or assistant, but got {message.role}')
            if not isinstance(message.content, str):
                raise MessageError(f'Message content must be str, but got {type(message.content)}')
            baichuan_messages.append(
                {
                    'role': role,
                    'content': message.content,
                }
            )

        data = {
            'model': self.model,
            'messages': baichuan_messages,
        }
        parameters_dict = parameters.model_dump(exclude_none=True)
        if parameters_dict:
            data['parameters'] = parameters_dict
        time_stamp = int(time.time())
        signature = self.calculate_md5(self.secret_key + json.dumps(data) + str(time_stamp))

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + self.api_key,
            'X-BC-Timestamp': str(time_stamp),
            'X-BC-Signature': signature,
            'X-BC-Sign-Algo': 'MD5',
            'X-BC-Request-Id': 'your requestId',
        }
        return {
            'url': self.api_base,
            'headers': headers,
            'json': data,
        }

    @staticmethod
    def calculate_md5(input_string: str):
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        encrypted = md5.hexdigest()
        return encrypted

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        try:
            text = response['data']['messages'][-1]['content']
            return [Message(role='assistant', content=text)]
        except (KeyError, IndexError) as e:
            raise ParserError(f'Parse response failed, reponse: {response}') from e

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
