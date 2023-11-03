from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import Field, PositiveInt, field_validator, model_validator
from typing_extensions import Annotated, NotRequired, TypedDict, Unpack, override

from lmclient.exceptions import MessageError, UnexpectedResponseError
from lmclient.models.http import HttpChatModel, HttpChatModelKwargs, HttpxPostKwargs
from lmclient.types import (
    Function,
    FunctionCallMessage,
    Message,
    Messages,
    ModelParameters,
    ModelResponse,
    Probability,
    Stream,
    Temperature,
    TextMessage,
)
from lmclient.utils import is_function_call_message, is_text_message


class BotSettingDict(TypedDict):
    bot_name: str
    content: str


class GlyphDict(TypedDict):
    type: str
    raw_glpyh: str
    json_properties: Dict[str, Any]


class ReplyConstrainsDict(TypedDict):
    sender_type: str
    sender_name: str
    glyph: NotRequired[GlyphDict]


class MinimaxFunctionCall(TypedDict):
    name: str
    arguments: str


class MinimaxProMessage(TypedDict):
    sender_type: Literal['USER', 'BOT', 'FUNCTION']
    sender_name: str
    text: str
    function_call: NotRequired[MinimaxFunctionCall]


class MinimaxProChatParameters(ModelParameters):
    reply_constraints: ReplyConstrainsDict = {'sender_type': 'BOT', 'sender_name': 'MM智能助理'}
    bot_setting: List[BotSettingDict] = [
        {'bot_name': 'MM智能助理', 'content': 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'}
    ]
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Annotated[Optional[PositiveInt], Field(serialization_alias='tokens_to_generate')] = None
    mask_sensitive_info: Optional[bool] = None
    sample_messages: Optional[List[MinimaxProMessage]] = None
    functions: Optional[List[Function]] = None
    plugins: Optional[List[str]] = None

    @model_validator(mode='after')
    def check_bot_name(self):
        names: set[str] = {bot_setting['bot_name'] for bot_setting in self.bot_setting}
        if len(self.bot_setting) == 1:
            if self.reply_constraints['sender_name'] != self.bot_setting[0]['bot_name']:
                raise ValueError(
                    f'reply_constraints sender_name {self.reply_constraints["sender_name"]} must be the same as bot_setting bot_name {self.bot_setting[0]["bot_name"]}'
                )

        if (sender_name := self.reply_constraints['sender_name']) not in names:
            raise ValueError(f'reply_constraints sender_name {sender_name} must be in bot_setting names: {names}')
        return self

    @field_validator('temperature', 'top_p', mode='after')
    @classmethod
    def zero_is_not_valid(cls, value):
        if value == 0:
            return 0.01
        return value

    @property
    def bot_name(self) -> str | None:
        if len(self.bot_setting) == 1:
            return self.bot_setting[0]['bot_name']
        else:
            return None

    def set_system_prompt(self, system_prompt: str) -> None:
        if len(self.bot_setting) == 1:
            self.bot_setting[0]['content'] = system_prompt
        else:
            raise ValueError('set system_prompt is not supported when bot_setting has more than one bot')

    def set_bot_name(self, bot_name: str) -> None:
        if len(self.bot_setting) == 1:
            self.bot_setting[0]['bot_name'] = bot_name
            self.reply_constraints['sender_name'] = bot_name
        else:
            raise ValueError('set bot_name is not supported when bot_setting has more than one bot')


def convert_to_minimax_pro_message(
    message: Message, default_bot_name: str | None = None, default_user_name: str = '用户'
) -> MinimaxProMessage:
    if is_function_call_message(message):
        sender_name = message.get('name') or default_bot_name
        if sender_name is None:
            raise MessageError(f'bot name is required for function call, message: {message}')
        return {
            'sender_type': 'BOT',
            'sender_name': sender_name,
            'text': '',
            'function_call': {
                'name': message['content']['name'],
                'arguments': message['content']['arguments'],
            },
        }
    elif is_text_message(message):
        if message['role'] == 'assistant':
            sender_name = message.get('name') or default_bot_name
            if sender_name is None:
                raise MessageError(f'bot name is required, message: {message}')
            return {
                'sender_type': 'BOT',
                'sender_name': sender_name,
                'text': message['content'],
            }
        elif message['role'] == 'function':
            name = message.get('name')
            if name is None:
                raise MessageError(f'function name is required, message: {message}')
            return {
                'sender_type': 'FUNCTION',
                'sender_name': name,
                'text': message['content'],
            }
        elif message['role'] == 'user':
            sender_name = message.get('name') or default_user_name
            return {'sender_type': 'USER', 'sender_name': sender_name, 'text': message['content']}
        else:
            raise MessageError(f'invalid message role: {message["role"]}')
    else:
        raise MessageError(f'invalid role {message["role"]}, must be one of "user", "assistant", "function"')


class MinimaxProChat(HttpChatModel[MinimaxProChatParameters]):
    model_type: ClassVar[str] = 'minimax_pro'
    default_api_base: ClassVar[str] = 'https://api.minimax.chat/v1/text/chatcompletion_pro'

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        bot_name: str | None = None,
        system_prompt: str | None = None,
        default_user_name: str = '用户',
        parameters: MinimaxProChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ):
        parameters = parameters or MinimaxProChatParameters()
        self.default_user_name = default_user_name
        if system_prompt is not None:
            parameters.set_system_prompt(system_prompt)
        if bot_name is not None:
            parameters.set_bot_name(bot_name)
        super().__init__(parameters=parameters, **kwargs)

        self.model = model
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.api_base = api_base or self.default_api_base

    @override
    def get_request_parameters(self, messages: Messages, parameters: MinimaxProChatParameters) -> HttpxPostKwargs:
        minimax_pro_messages = [
            convert_to_minimax_pro_message(
                message, default_bot_name=parameters.bot_name, default_user_name=self.default_user_name
            )
            for message in messages
        ]
        parameters_dict = parameters.model_dump(exclude_none=True, by_alias=True)
        json_data = {'model': self.model, 'messages': minimax_pro_messages, **parameters_dict}
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
    def parse_reponse(self, response: ModelResponse) -> Messages:
        try:
            return [self._convert_to_message(i) for i in response['choices'][0]['messages']]
        except (KeyError, IndexError, TypeError) as e:
            raise UnexpectedResponseError(response) from e

    @override
    def get_stream_request_parameters(self, messages: Messages, parameters: MinimaxProChatParameters) -> HttpxPostKwargs:
        http_parameters = self.get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def parse_stream_response(self, response: ModelResponse) -> Stream:
        delta = response['choices'][0]['messages'][0]['text']
        if response['reply']:
            return Stream(delta=delta, control='finish')
        return Stream(delta=delta, control='continue')

    @staticmethod
    def _convert_to_message(message: MinimaxProMessage) -> Message:
        role_map: dict[str, Literal['user', 'assistant', 'function']] = {
            'USER': 'user',
            'BOT': 'assistant',
            'FUNCTION': 'function',
        }

        if 'function_call' in message:
            return FunctionCallMessage(
                role='assistant',
                name=message['sender_name'],
                content={'name': message['function_call']['name'], 'arguments': message['function_call']['arguments']},
            )
        else:
            return TextMessage(
                role=role_map[message['sender_type']],
                name=message['sender_name'],
                content=message['text'],
            )

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
