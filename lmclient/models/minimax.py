from __future__ import annotations

import os
from typing import Any, ClassVar, Literal, Optional

from pydantic import Field, PositiveInt, field_validator
from typing_extensions import Annotated, TypedDict, Unpack, override

from lmclient.exceptions import MessageError, ResponseFailedError
from lmclient.models.http import HttpChatModel, HttpChatModelKwargs, HttpxPostKwargs
from lmclient.types import (
    Message,
    Messages,
    ModelParameters,
    ModelResponse,
    Probability,
    Temperature,
    TextMessage,
)
from lmclient.utils import is_text_message


class MinimaxMessage(TypedDict):
    sender_type: Literal['USER', 'BOT']
    text: str


class RoleMeta(TypedDict):
    user_name: str
    bot_name: str


class MinimaxChatParameters(ModelParameters):
    prompt: str = 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'
    role_meta: RoleMeta = {'user_name': '用户', 'bot_name': 'MM智能助理'}
    beam_width: Optional[Annotated[int, Field(ge=1, le=4)]] = None
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[Annotated[PositiveInt, Field(serialization_alias='tokens_to_generate')]] = None
    skip_info_mask: Optional[bool] = None
    continue_last_message: Optional[bool] = None

    @field_validator('temperature', 'top_p', mode='after')
    @classmethod
    def zero_is_not_valid(cls, value):
        if value == 0:
            return 0.01
        return value


def convert_to_minimax_message(message: Message) -> MinimaxMessage:
    if not is_text_message(message):
        raise MessageError(f'Invalid message type: {type(message)}, only TextMessage is allowed')
    role = message['role']
    if role != 'assistant' and role != 'user':
        raise MessageError(f'Invalid message role: {role}, only "user" and "assistant" are allowed')

    if role == 'assistant':
        return {
            'sender_type': 'BOT',
            'text': message['content'],
        }
    else:
        return {
            'sender_type': 'USER',
            'text': message['content'],
        }


class MinimaxChat(HttpChatModel[MinimaxChatParameters]):
    model_type: ClassVar[str] = 'minimax'
    default_api_base: ClassVar[str] = 'https://api.minimax.chat/v1/text/chatcompletion'

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt: str | None = None,
        parameters: MinimaxChatParameters | None = None,
        **kwagrs: Unpack[HttpChatModelKwargs],
    ):
        parameters = MinimaxChatParameters()
        if system_prompt is not None:
            parameters.prompt = system_prompt
        super().__init__(parameters=parameters, **kwagrs)

        self.model = model
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.api_base = api_base or self.default_api_base

    @override
    def get_request_parameters(self, messages: Messages, parameters: MinimaxChatParameters) -> HttpxPostKwargs:
        minimax_messages = [convert_to_minimax_message(message) for message in messages]
        parameters_dict = parameters.model_dump(exclude_none=True, by_alias=True)
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data = {
            'model': self.model,
            'messages': minimax_messages,
            **parameters_dict,
        }
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        return {
            'url': self.api_base,
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.group_id},
        }

    @override
    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        try:
            return [TextMessage(role='assistant', content=response['choices'][0]['text'])]
        except (KeyError, IndexError, TypeError) as e:
            raise ResponseFailedError(f'Response Failed: {response}') from e

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
