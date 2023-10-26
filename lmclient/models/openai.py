from __future__ import annotations

import os
from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import NotRequired, TypedDict

from lmclient.exceptions import MessageError
from lmclient.models.http import HttpChatModel, ProxiesTypes, RetryStrategy
from lmclient.parser import ParserError
from lmclient.types import (
    FunctionCallDict,
    FunctionDict,
    GeneralParameters,
    Message,
    Messages,
    ModelParameters,
    ModelResponse,
    is_function_call_message,
    is_text_message,
)


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


def format_message_to_openai(message: Message) -> OpenAIMessageDict:
    role = message.role
    if role == 'error':
        raise MessageError(f'Invalid message role: {role}, only "user", "assistant", "system" and "function" are allowed')

    if is_function_call_message(message):
        function_call = copy(message.content)
        if 'thoughts' in function_call:
            function_call.pop('thoughts')
        return {
            'role': 'assistant',
            'function_call': function_call,
            'content': None,
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
        system_prompt: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        parameters: OpenAIChatParameters = OpenAIChatParameters(),
        use_cache: Path | str | bool = False,
        proxies: ProxiesTypes | None = None,
    ):
        super().__init__(parameters=parameters, timeout=timeout, retry=retry, use_cache=use_cache, proxies=proxies)
        self.model = model
        self.system_prompt = system_prompt
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
        self.api_key = api_key or os.environ['OPENAI_API_KEY']

    def get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        parameters_dict = parameters.model_dump(exclude_defaults=True)
        openai_messages = [format_message_to_openai(message) for message in messages]
        if self.system_prompt:
            openai_messages.insert(0, {'role': 'system', 'content': self.system_prompt})
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
