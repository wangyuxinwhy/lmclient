from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, RetryStrategy
from lmclient.parser import ParserError
from lmclient.types import FunctionCallDict, FunctionDict, GeneralParameters, Message, Messages, ModelParameters, ModelResponse
from lmclient.utils import to_dict


class FunctionCallNameDict(TypedDict):
    name: str


class OpenAIMessageDict(TypedDict):
    role: str
    content: Optional[str]
    name: NotRequired[str]
    function_call: NotRequired[FunctionCallDict]


class OpenAIChatParameters(ModelParameters):
    temperature: float = 1
    top_p: float = 1
    max_tokens: Optional[int] = None
    functions: Optional[List[FunctionDict]] = None
    function_call: Union[Literal['auto'], FunctionCallNameDict, None] = None
    stop: Union[str, List[str], None] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[int, int]] = None
    user: Optional[str] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        if general_parameters.function_call != 'auto' and general_parameters.function_call is not None:
            function_call = FunctionCallNameDict(name=general_parameters.function_call)
        else:
            function_call = general_parameters.function_call

        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
            max_tokens=general_parameters.max_tokens,
            functions=general_parameters.functions,
            function_call=function_call,
        )


class OpenAIExtractParameters(ModelParameters):
    temperature: float = 1
    top_p: float = 1
    stop: Union[str, List[str], None] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    logit_bias: Optional[Dict[int, int]] = None
    user: Optional[str] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
        )


def convert_lmclient_to_openai(message: Message, valid_roles: set[str] | None = None) -> OpenAIMessageDict:
    valid_roles = valid_roles or {'user', 'assistant', 'function', 'system'}
    if message.role not in valid_roles:
        raise MessageError(f'Invalid role "{message.role}", supported roles are {valid_roles}')

    content = message.content

    if isinstance(content, dict):
        if message.role != 'assistant':
            raise MessageError(f'Invalid role "{message.role}" for function call, can only be made by "assistant"')
        return {
            'role': message.role,
            'function_call': content,
            'content': None,
        }
    elif message.role == 'function':
        name = message.name
        if name is None:
            raise MessageError(f'Function name is required, message: {message}')
        return {
            'role': message.role,
            'name': name,
            'content': content,
        }
    else:
        return {
            'role': message.role,
            'content': content,
        }


def parse_openai_model_reponse(response: ModelResponse) -> Messages:
    funcation_call = response['choices'][0]['message'].get('function_call')
    try:
        if bool(funcation_call):
            return [
                Message(
                    role='assistant',
                    content=funcation_call,
                )
            ]
        else:
            text: str = response['choices'][0]['message']['content']
            return [
                Message(
                    role='assistant',
                    content=text,
                )
            ]
    except (KeyError, IndexError) as e:
        raise ParserError('Parse response failed') from e


class OpenAIChat(HttpChatModel[OpenAIChatParameters]):
    model_type = 'openai'

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        parameters: OpenAIChatParameters = OpenAIChatParameters(),
        use_cache: Path | str | bool = False,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache)
        self.model = model
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
        self.api_key = api_key or os.environ['OPENAI_API_KEY']

    def get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        parameters_dict = to_dict(parameters, exclude_defaults=True)
        openai_messages = [convert_lmclient_to_openai(message) for message in messages]
        params = {
            'model': self.model,
            'messages': openai_messages,
            **parameters_dict,
        }
        return {
            'url': f'{self.api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        return parse_openai_model_reponse(response)

    @property
    def name(self) -> str:
        return self.model

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
