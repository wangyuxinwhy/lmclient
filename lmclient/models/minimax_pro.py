from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field
from typing_extensions import NotRequired, TypedDict

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, ProxiesTypes, RetryStrategy
from lmclient.types import (
    FunctionCallDict,
    FunctionDict,
    GeneralParameters,
    Message,
    Messages,
    ModelParameters,
    ModelResponse,
    Role,
)

DEFAULT_MINIMAX_BOT_NAME = 'MM智能助理'
DEFAULT_MINIMAX_USER_NAME = '用户'
DEFAULT_MINIMAX_BOT_PROMPT = 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'


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


class MinimaxMessageDict(TypedDict):
    sender_type: Literal['USER', 'BOT', 'FUNCTION']
    sender_name: str
    text: str
    function_call: NotRequired[FunctionCallDict]


def default_bot_setting():
    return [{'bot_name': 'MM智能助理', 'content': DEFAULT_MINIMAX_BOT_PROMPT}]


def default_reply_constrains():
    return {'sender_type': 'BOT', 'sender_name': DEFAULT_MINIMAX_BOT_NAME}


class MinimaxProChatParameters(ModelParameters):
    bot_setting: List[BotSettingDict] = Field(default_factory=default_bot_setting)
    reply_constraints: ReplyConstrainsDict = Field(default_factory=default_reply_constrains)
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tokens_to_generate: Optional[int] = None
    mask_sensitive_info: Optional[bool] = None
    sample_messages: Optional[List[MinimaxMessageDict]] = None
    functions: Optional[List[FunctionDict]] = None
    plugins: Optional[List[str]] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
            tokens_to_generate=general_parameters.max_tokens or 1024,
            functions=general_parameters.functions,
        )


class MinimaxProChat(HttpChatModel[MinimaxProChatParameters]):
    model_type = 'minimax_pro'

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        base_url: str = 'https://api.minimax.chat/v1/text/chatcompletion_pro',
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        parameters: MinimaxProChatParameters = MinimaxProChatParameters(),
        use_cache: Path | str | bool = False,
        proxies: ProxiesTypes | None = None,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache, proxies=proxies)
        self.model = model
        self.base_url = base_url
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']

    def get_request_parameters(self, messages: Messages, parameters: MinimaxProChatParameters) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

        json_data = {'model': self.model, 'messages': [self._lmclient_to_minimax(message) for message in messages]}
        parameters_dict = parameters.model_dump(exclude_none=True)
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
        role_map: dict[str, Role] = {
            'USER': 'user',
            'BOT': 'assistant',
            'FUNCTION': 'function',
        }

        if 'function_call' in message:
            return Message(role=role_map[message['sender_type']], name=message['sender_name'], content=message['function_call'])
        else:
            return Message(
                role=role_map[message['sender_type']],
                name=message['sender_name'],
                content=message['text'],
            )

    def _lmclient_to_minimax(
        self,
        message: Message,
        default_bot_name: str = DEFAULT_MINIMAX_BOT_NAME,
        default_user_name: str = DEFAULT_MINIMAX_USER_NAME,
    ) -> MinimaxMessageDict:
        if isinstance(message.content, dict):
            if message.role != 'assistant':
                raise MessageError(f'Invalid role {message.role} for function call, must be assistant')
            return {
                'sender_type': 'BOT',
                'sender_name': message.name or default_bot_name,
                'text': '',
                'function_call': message.content,
            }
        elif message.role == 'assistant':
            return {
                'sender_type': 'BOT',
                'sender_name': message.name or default_bot_name,
                'text': message.content,
            }
        elif message.role == 'function':
            if message.name is None:
                raise MessageError(f'Function name is required, message: {message}')
            return {
                'sender_type': 'FUNCTION',
                'sender_name': message.name,
                'text': message.content,
            }
        elif message.role == 'user':
            return {
                'sender_type': 'USER',
                'sender_name': message.name or default_user_name,
                'text': message.content,
            }
        else:
            raise MessageError(f'Invalid role {message.role}, must be BOT, FUNCTION, or USER')

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
