from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, ClassVar, Literal, Optional, TypedDict

from pydantic import Field
from typing_extensions import Annotated, Unpack, override

from lmclient.exceptions import MessageError, ResponseFailedError
from lmclient.models.http import HttpChatModel, HttpChatModelKwargs, HttpxPostKwargs
from lmclient.types import Message, Messages, ModelParameters, ModelResponse, Probability, Temperature, TextMessage
from lmclient.utils import is_text_message


class BaichuanMessage(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class BaichuanChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_k: Optional[Annotated[int, Field(ge=0)]] = None
    top_p: Optional[Probability] = None
    with_search_enhance: Optional[bool] = None


class BaichuanChat(HttpChatModel[BaichuanChatParameters]):
    model_type: ClassVar[str] = 'zhipu'
    default_api_base: ClassVar[str] = 'https://api.baichuan-ai.com/v1/chat'

    def __init__(
        self,
        model: str = 'Baichuan2-53B',
        api_key: str | None = None,
        secret_key: str | None = None,
        api_base: str | None = None,
        parameters: BaichuanChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ):
        parameters = parameters or BaichuanChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = model
        self.api_key = api_key or os.environ['BAICHUAN_API_KEY']
        self.secret_key = secret_key or os.environ['BAICHUAN_SECRET_KEY']
        self.api_base = api_base or self.default_api_base
        self.api_base.rstrip('/')

    @override
    def get_request_parameters(self, messages: Messages, parameters: BaichuanChatParameters) -> HttpxPostKwargs:
        baichuan_messages: list[BaichuanMessage] = [self.convert_to_baichuan_message(message) for message in messages]
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
    def convert_to_baichuan_message(message: Message) -> BaichuanMessage:
        if not is_text_message(message):
            raise MessageError(f'Invalid message type: {type(message)}, only TextMessage is allowed')
        role = message['role']
        if role != 'assistant' and role != 'user':
            raise MessageError(f'Invalid message role: {role}, only "user" and "assistant" are allowed')

        return {
            'role': role,
            'content': message['content'],
        }

    @staticmethod
    def calculate_md5(input_string: str):
        md5 = hashlib.md5()
        md5.update(input_string.encode('utf-8'))
        encrypted = md5.hexdigest()
        return encrypted

    @override
    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        try:
            text = response['data']['messages'][-1]['content']
            return [TextMessage(role='assistant', content=text)]
        except (KeyError, IndexError) as e:
            raise ResponseFailedError(f'Response Failed, {response}') from e

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)