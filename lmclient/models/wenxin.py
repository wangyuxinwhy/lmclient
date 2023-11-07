from __future__ import annotations

import os
from datetime import datetime, timedelta
from email.errors import MessageError
from typing import Any, ClassVar, List, Literal, Optional

import httpx
from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from lmclient.exceptions import UnexpectedResponseError
from lmclient.models.http import HttpChatModel, HttpChatModelKwargs, HttpxPostKwargs
from lmclient.types import (
    FunctionCallMessage,
    GeneralParameters,
    JsonSchema,
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


class WenxinMessage(TypedDict):
    role: Literal['user', 'assistant', 'function']
    content: str
    name: NotRequired[str]
    function_call: NotRequired[WenxinFunctionCall]


class WenxinFunctionCall(TypedDict):
    name: str
    arguments: str
    thoughts: NotRequired[str]


class WenxinFunction(TypedDict):
    name: str
    description: str
    parameters: JsonSchema
    responses: NotRequired[JsonSchema]
    examples: NotRequired[List[WenxinMessage]]


def convert_to_wenxin_message(message: Message) -> WenxinMessage:
    role = message['role']

    if role == 'system':
        raise MessageError(f'Invalid message role: {role}, only "user", "assistant" and "function" are allowed')

    if is_function_call_message(message):
        return {
            'role': 'assistant',
            'function_call': {
                'name': message['content']['name'],
                'arguments': message['content']['arguments'],
                'thoughts': message['content'].get('thoughts', ''),
            },
            'content': '',
        }
    if is_text_message(message):
        if role == 'function':
            if (name := message.get('name')) is None:
                raise MessageError(f'Function name is required, message: {message}')
            return {
                'role': role,
                'name': name,
                'content': message['content'],
            }

        return {
            'role': role,
            'content': message['content'],
        }
    raise MessageError(f'Invalid message type: {type(message)}')


class WenxinChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    functions: Optional[List[WenxinFunction]] = None
    penalty_score: Optional[Annotated[float, Field(ge=1, le=2)]] = None
    system: Optional[str] = None
    user_id: Optional[str] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        parameters = general_parameters.model_copy(deep=True)
        if parameters.functions is not None:
            wenxin_functions: list[WenxinFunction] | None = []
            for general_function in parameters.functions:
                wenxin_function = WenxinFunction(
                    name=general_function['name'],
                    description=general_function.get('description', ''),
                    parameters=general_function['parameters'],
                )
                if 'responses' in general_function:
                    wenxin_function['responses'] = general_function['responses']
                if 'examples' in general_function:
                    wenxin_messages = [convert_to_wenxin_message(message) for message in general_function['examples']]
                    wenxin_function['examples'] = wenxin_messages
                wenxin_functions.append(wenxin_function)
        else:
            wenxin_functions = None
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
            functions=wenxin_functions,
        )

    @model_validator(mode='after')
    def system_function_coflict(self) -> Self:
        if self.system is not None and self.functions is not None:
            raise ValueError('system and functions cannot be used together')
        return self

    @field_validator('temperature', mode='after')
    @classmethod
    def temperature_gt_0(cls, value: float) -> float | Any:
        if value == 0:
            return 0.01
        return value


class WenxinChat(HttpChatModel[WenxinChatParameters]):
    model_type: ClassVar[str] = 'wenxin'
    model_name_entrypoint_map: ClassVar[dict[str, str]] = {
        'llama_2_7b': 'llama_2_7b',
        'llama_2_13b': 'llama_2_13b',
        'llama_2_70b': 'llama_2_70b',
        'ERNIE-Bot': 'completions',
        'ERNIE-Bot-turbo': 'eb-instant',
        'ERNIE-Bot-4': 'completions_pro',
    }
    access_token_refresh_days: ClassVar[int] = 20
    access_token_url: ClassVar[str] = 'https://aip.baidubce.com/oauth/2.0/token'
    default_api_base: ClassVar[str] = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/'

    def __init__(
        self,
        model: str = 'ERNIE-Bot',
        api_key: str | None = None,
        api_base: str | None = None,
        secret_key: str | None = None,
        parameters: WenxinChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ) -> None:
        parameters = parameters or WenxinChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = self.normalize_model(model)
        self.api_base = api_base or self.default_api_base
        self._api_key = api_key or os.environ['WENXIN_API_KEY']
        self._secret_key = secret_key or os.environ['WENXIN_SECRET_KEY']
        self._access_token = self.get_access_token()
        self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    @property
    @override
    def name(self) -> str:
        return self.model

    @property
    def api_url(self) -> str:
        return self.api_base + self.model_name_entrypoint_map[self.model]

    @staticmethod
    def normalize_model(model: str) -> str:
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
            raise UnexpectedResponseError(response_dict)
        return response_dict['access_token']

    @override
    def _get_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> HttpxPostKwargs:
        self.maybe_refresh_access_token()

        wenxin_messages: list[WenxinMessage] = [convert_to_wenxin_message(message) for message in messages]
        parameters_dict = parameters.model_dump(exclude_none=True)
        if 'temperature' in parameters_dict:
            parameters_dict['temperature'] = max(0.01, parameters_dict['temperature'])
        json_data = {'messages': wenxin_messages, **parameters_dict}

        return {
            'url': self.api_url,
            'json': json_data,
            'params': {'access_token': self._access_token},
            'headers': {'Content-Type': 'application/json'},
        }

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def _parse_stream_response(self, response: ModelResponse) -> Stream:
        if response['is_end']:
            return Stream(delta=response['result'], control='finish')
        return Stream(delta=response['result'], control='continue')

    @override
    def _parse_reponse(self, response: ModelResponse) -> Messages:
        if response.get('error_msg'):
            raise UnexpectedResponseError(response)
        if response.get('function_call'):
            return [
                FunctionCallMessage(
                    role='assistant',
                    content={
                        'name': response['function_call']['name'],
                        'arguments': response['function_call']['arguments'],
                        'thoughts': response['function_call']['thoughts'],
                    },
                )
            ]

        return [TextMessage(role='assistant', content=response['result'])]

    def maybe_refresh_access_token(self) -> None:
        if self._access_token_expires_at < datetime.now():
            self._access_token = self.get_access_token()
            self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
