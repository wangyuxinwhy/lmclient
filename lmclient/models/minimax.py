from __future__ import annotations

import os
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional

from pydantic import Field
from typing_extensions import TypedDict

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, ProxiesTypes, RetryStrategy
from lmclient.types import (
    GeneralParameters,
    Message,
    Messages,
    ModelParameters,
    ModelResponse,
    Role,
)

DEFAULT_MINIMAX_BOT_PROMPT = 'MM智能助理是一款由MiniMax自研的，没有调用其他产品的接口的大型语言模型。MiniMax是一家中国科技公司，一直致力于进行大模型相关的研究。'


class MinimaxMessageDict(TypedDict):
    sender_type: Literal['USER', 'BOT']
    text: str


class RoleMeta(TypedDict):
    user_name: str
    bot_name: str


def default_role_meta():
    return {'user_name': '用户', 'bot_name': 'MM智能助理'}


class MinimaxChatParameters(ModelParameters):
    prompt: str = DEFAULT_MINIMAX_BOT_PROMPT
    role_meta: RoleMeta = Field(default_factory=default_role_meta)
    beam_width: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tokens_to_generate: Optional[int] = None
    skip_info_mask: Optional[bool] = None
    continue_last_message: Optional[bool] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
            tokens_to_generate=general_parameters.max_tokens,
        )


class MinimaxChat(HttpChatModel[MinimaxChatParameters]):
    model_type = 'minimax'
    default_api_base: ClassVar[str] = 'https://api.minimax.chat/v1/text/chatcompletion'

    def __init__(
        self,
        model: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        parameters: MinimaxChatParameters = MinimaxChatParameters(),
        use_cache: Path | str | bool = False,
        proxies: ProxiesTypes | None = None,
    ):
        if system_prompt is not None:
            parameters = parameters.model_copy(update={'prompt': system_prompt}, deep=True)
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache, proxies=proxies)

        self.model = model
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.api_base = api_base or self.default_api_base

    def get_request_parameters(self, messages: Messages, parameters: MinimaxChatParameters) -> dict[str, Any]:
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
            'url': self.api_base,
            'json': json_data,
            'headers': headers,
            'params': {'GroupId': self.group_id},
        }

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        try:
            return [Message(role='assistant', content=response['choices'][0]['text'])]
        except (KeyError, IndexError, TypeError) as e:
            raise MessageError(f'Invalid response from Minimax: {response}') from e

    @staticmethod
    def _minimax_to_lmclient(message: MinimaxMessageDict) -> Message:
        role_map: dict[str, Role] = {
            'USER': 'user',
            'BOT': 'assistant',
        }

        return Message(
            role=role_map[message['sender_type']],
            content=message['text'],
        )

    def _lmclient_to_minimax(self, message: Message) -> MinimaxMessageDict:
        if isinstance(message.content, dict):
            raise MessageError(f'Function call is not supported in Minimax: {message}')
        elif message.role == 'function':
            raise MessageError(f'Function role is not supported in Minimax: {message}')
        elif message.role == 'assistant':
            return {
                'sender_type': 'BOT',
                'text': message.content,
            }
        elif message.role == 'user':
            return {
                'sender_type': 'USER',
                'text': message.content,
            }
        else:
            raise MessageError(f'Invalid role {message.role}, must be user or assistant')

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
