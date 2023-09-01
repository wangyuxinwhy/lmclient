from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional, Type

from typing_extensions import NotRequired, TypedDict

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, RetryStrategy
from lmclient.parser import ModelResponseParser, ParserError
from lmclient.types import (
    Field,
    FunctionCallDict,
    FunctionDict,
    GeneralParameters,
    Message,
    Messages,
    ModelParameters,
    ModelResponse,
)
from lmclient.utils import to_dict

DEFAULT_BOT_PROMPT = "MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。"


class BotSettingDict(TypedDict):
    bot_name: str
    content: str


class GlyphDict(TypedDict):
    type: str
    raw_glpyh: str
    json_properties: dict


class ReplyConstrainsDict(TypedDict):
    sender_type: str
    sender_name: str
    glyph: NotRequired[GlyphDict]


class MinimaxMessageDict(TypedDict):
    sender_type: Literal['USER', 'BOT', 'FUNCTION']
    sender_name: str
    text: str
    function_call: NotRequired[FunctionCallDict]


class MinimaxProChatParameters(ModelParameters):
    temperature: float = 1
    top_p: float = 1
    tokens_to_generate: int = 1024
    mask_sensitive_info: bool = True
    bot_setting: List[BotSettingDict] = Field(default_factory=list)
    reply_constrains: ReplyConstrainsDict = Field(default_factory=list)
    sample_messages: Optional[List[MinimaxMessageDict]] = None
    functions: Optional[List[FunctionDict]] = None
    plugins : Optional[List[str]] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
            tokens_to_generate=general_parameters.max_tokens or 1024,
            functions=general_parameters.functions,
        )

class MinimaxProFunctionCallParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> FunctionCallDict:
        try:
            function_call_dict = response['choices'][0]['messages'][-1]['function_call']
            return FunctionCallDict(
                name=function_call_dict['name'],
                arguments=json.loads(function_call_dict['arguments'])
            )
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e


class MinimaxProTextParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> str:
        try:
            output = response['reply']
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output


class MinimaxProParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> Messages:
        return [self._minimax_to_lmclient(i) for i in response['choices'][0]['messages']]

    @staticmethod
    def _minimax_to_lmclient(message: MinimaxMessageDict) -> Message:
        if 'function_call' in message:
            return Message(
                role=message['sender_type'],
                name=message['sender_name'],
                content=message['function_call']
            )
        else:
            return Message(
                role=message['sender_type'],
                name=message['sender_name'],
                content=message['text'],
            )


class MinimaxProChat(HttpChatModel[MinimaxProChatParameters]):
    parameters_type = MinimaxProChatParameters

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        base_url: str = 'https://api.minimax.chat/v1/text/chatcompletion_pro',
        group_id: str | None = None,
        api_key: str | None = None,
        bot_name: str = 'MM智能助理',
        user_name: str = '用户',
        system_prompt: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        default_parameters: MinimaxProChatParameters | None = None,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(default_parameters=default_parameters, timeout=timeout, retry=retry, use_cache=use_cache)
        self.model = model
        self.base_url = base_url
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.bot_name = bot_name
        self.system_prompt = system_prompt or DEFAULT_BOT_PROMPT
        self.user_name = user_name

    def get_post_parameters(self, messages: Messages, parameters: MinimaxProChatParameters | None = None) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        json_data = {
            'model': self.model,
            'messages': [self._lmclient_to_minimax(message, self.bot_name, self.user_name) for message in messages]
        }

        parameters = parameters or MinimaxProChatParameters()
        if not parameters.bot_setting:
            parameters.bot_setting = [{'bot_name': self.bot_name, 'content': self.system_prompt}]
        if not parameters.reply_constrains:
            parameters.reply_constrains = {'sender_type': 'USER', 'sender_name': self.bot_name}
        parameters_dict = to_dict(parameters, exclude_defaults=True) if parameters else {}
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data.update(parameters_dict)

        return {
            'url': self.base_url,
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.group_id},
        }

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        return [self._minimax_to_lmclient(i) for i in response['choices'][0]['messages']]

    @staticmethod
    def _minimax_to_lmclient(message: MinimaxMessageDict) -> Message:
        if 'function_call' in message:
            return Message(
                role=message['sender_type'],
                name=message['sender_name'],
                content=message['function_call']
            )
        else:
            return Message(
                role=message['sender_type'],
                name=message['sender_name'],
                content=message['text'],
            )

    def _lmclient_to_minimax(self, message: Message, default_bot_name: str = 'MM智能助理', default_user_name: str = '用户') -> MinimaxMessageDict:
        if isinstance(message.content, dict):
            if message.role != 'BOT':
                raise MessageError(f'Invalid role {message.role} for function call, must be BOT')
            return {
                'sender_type': message.role,
                'sender_name': message.name or default_bot_name,
                'text': '',
                'function_call': message.content,
            }
        elif message.role == 'BOT':
                return {
                    'sender_type': message.role,
                    'sender_name': message.name or default_bot_name,
                    'text': message.content,
                }
        elif message.role == 'FUNCTION':
            if message.name is None:
                raise MessageError(f'Function name is required, message: {message}')
            return {
                'sender_type': message.role,
                'sender_name': message.name,
                'text': message.content,
            }
        elif message.role == 'USER':
            return {
                'sender_type': message.role,
                'sender_name': message.name or default_user_name,
                'text': message.content,
            }
        else:
            raise MessageError(f'Invalid role {message.role}, must be BOT, FUNCTION, or USER')

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
