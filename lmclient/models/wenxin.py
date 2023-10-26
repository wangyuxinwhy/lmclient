from __future__ import annotations

import os
from datetime import datetime, timedelta
from email.errors import MessageError
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from typing_extensions import NotRequired, Self, TypedDict

from lmclient.exceptions import ResponseError
from lmclient.models.http import HttpChatModel, ProxiesTypes, RetryStrategy
from lmclient.types import (
    GeneralParameters,
    Message,
    Messages,
    ModelParameters,
    ModelResponse,
    is_function_call_message,
    is_text_message,
)


class WenxinMessageDict(TypedDict):
    role: Literal['user', 'assistant', 'function']
    content: str
    name: NotRequired[str]
    function_call: NotRequired[WenxinFunctionCallDict]


class WenxinFunctionCallDict(TypedDict):
    name: str
    arguments: str
    thoughts: NotRequired[str]


class WenxinFunctionDict(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]
    responses: NotRequired[Dict[str, str]]
    examples: NotRequired[List[WenxinMessageDict]]


def format_message_to_wenxin(message: Message) -> WenxinMessageDict:
    role = message.role
    if role == 'error' or role == 'system':
        raise MessageError(f'Invalid message role: {role}, only "user", "assistant" and "function" are allowed')

    if is_function_call_message(message):
        return {
            'role': 'assistant',
            'function_call': {
                'name': message.content['name'],
                'arguments': message.content['arguments'],
                'thoughts': message.content.get('thoughts', ''),
            },
            'content': '',
        }
    elif is_text_message(message):
        if role == 'function':
            name = message.name
            if name is None:
                raise MessageError(f'Function name is required, message: {message}')
            return {
                'role': role,
                'name': name,
                'content': message.content,
            }
        else:
            return {
                'role': role,
                'content': message.content,
            }
    else:
        raise MessageError(f'Invalid message type: {message}')


class WenxinChatParameters(ModelParameters):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    functions: Optional[List[WenxinFunctionDict]] = None
    penalty_score: Optional[float] = None
    system: Optional[str] = None
    user_id: Optional[str] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        if general_parameters.functions is not None:
            wenxin_functions: list[WenxinFunctionDict] | None = []
            for general_function in general_parameters.functions:
                wenxin_function = WenxinFunctionDict(
                    name=general_function['name'],
                    description=general_function.get('description', ''),
                    parameters=general_function['parameters'],
                )
                if 'responses' in general_function:
                    wenxin_function['responses'] = general_function['responses']
                if 'examples' in general_function:
                    messages = [Message(**message_dict) for message_dict in general_function['examples']]
                    wenxin_messages = [format_message_to_wenxin(message) for message in messages]
                    wenxin_function['examples'] = wenxin_messages
                wenxin_functions.append(wenxin_function)
        else:
            wenxin_functions = None
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
            functions=wenxin_functions,
        )


class WenxinChat(HttpChatModel[WenxinChatParameters]):
    model_type = 'wenxin'
    model_name_entrypoint_map: dict[str, str] = {
        'llama_2_7b': 'llama_2_7b',
        'llama_2_13b': 'llama_2_13b',
        'llama_2_70b': 'llama_2_70b',
        'ERNIE-Bot': 'completions',
        'ERNIE-Bot-turbo': 'eb-instant',
        'ERNIE-Bot-4': 'completions_pro',
    }
    access_token_refresh_days: int = 20
    access_token_url = 'https://aip.baidubce.com/oauth/2.0/token'
    default_api_base = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/'

    def __init__(
        self,
        model: str = 'ERNIE-Bot',
        api_key: str | None = None,
        api_base: str | None = None,
        secret_key: str | None = None,
        parameters: WenxinChatParameters = WenxinChatParameters(),
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
        proxies: ProxiesTypes | None = None,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache, proxies=proxies)
        self.model = self.normalize_model(model)
        self.api_base = api_base or self.default_api_base
        self._api_key = api_key or os.environ['WENXIN_API_KEY']
        self._secret_key = secret_key or os.environ['WENXIN_SECRET_KEY']
        self._access_token = self.get_access_token()
        self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    @property
    def name(self) -> str:
        return self.model

    @property
    def api_url(self) -> str:
        return self.api_base + self.model_name_entrypoint_map[self.model]

    @staticmethod
    def normalize_model(model: str):
        _map = {
            'llama-2-7b-chat': 'llama_2_7b',
            'llama-2-13b-chat': 'llama_2_13b',
            'llama-2-70b-chat': 'llama_2_70b',
        }
        return _map.get(model, model)

    def get_access_token(self) -> str:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        params = {'grant_type': 'client_credentials', 'client_id': self._api_key, 'client_secret': self._secret_key}
        response = httpx.post(self.access_token_url, headers=headers, params=params)
        response.raise_for_status()
        response_dict = response.json()
        if 'error' in response_dict:
            raise ResponseError(response_dict['error_description'])
        return response_dict['access_token']

    def get_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> dict[str, Any]:
        self.maybe_refresh_access_token()

        message_dicts: list[WenxinMessageDict] = [format_message_to_wenxin(message) for message in messages]
        parameters_dict = parameters.model_dump(exclude_none=True)
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data = {'messages': message_dicts, **parameters_dict}

        return {
            'url': self.api_url,
            'json': json_data,
            'params': {'access_token': self._access_token},
            'headers': {'Content-Type': 'application/json'},
        }

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        if response.get('error_msg'):
            raise ResponseError(response['error_msg'])
        if response.get('function_call'):
            arguments = response['function_call']['arguments']
            name = response['function_call']['name']
            return [Message(role='assistant', content={'name': name, 'arguments': arguments})]
        else:
            return [Message(role='assistant', content=response['result'])]

    def maybe_refresh_access_token(self):
        if self._access_token_expires_at < datetime.now():
            self._access_token = self.get_access_token()
            self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
