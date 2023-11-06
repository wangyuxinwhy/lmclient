from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import Field, PositiveInt
from typing_extensions import Annotated, NotRequired, Self, TypedDict, Unpack, override

from lmclient.exceptions import MessageError, UnexpectedResponseError
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


class FunctionCallName(TypedDict):
    name: str


class OpenaiFunction(TypedDict):
    name: str
    parameters: JsonSchema
    description: NotRequired[str]


class OpenaiFunctionCall(TypedDict):
    name: str
    arguments: str


class OpenAIMessage(TypedDict):
    role: str
    content: Optional[str]
    name: NotRequired[str]
    function_call: NotRequired[OpenaiFunctionCall]


class OpenAIChatParameters(ModelParameters):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    functions: Optional[List[OpenaiFunction]] = None
    function_call: Union[Literal['auto'], FunctionCallName, None] = None
    stop: Union[str, List[str], None] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    logit_bias: Optional[Dict[int, Annotated[int, Field(ge=-100, le=100)]]] = None
    user: Optional[str] = None

    @override
    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        parameters = general_parameters.model_copy(deep=True)
        if parameters.functions:
            for function in parameters.functions:
                function.pop('examples', None)
                function.pop('responses', None)
        return super().from_general_parameters(parameters)


def convert_to_openai_message(message: Message) -> OpenAIMessage:
    if is_function_call_message(message):
        return {
            'role': 'assistant',
            'function_call': {
                'name': message['content']['name'],
                'arguments': message['content']['arguments'],
            },
            'content': None,
        }
    elif is_text_message(message):
        role = message['role']
        if role == 'function':
            name = message.get('name')
            if (name := message.get('name')) is None:
                raise MessageError(f'function name is required, message: {message}')
            return {
                'role': role,
                'name': name,
                'content': message['content'],
            }
        else:
            return {
                'role': role,
                'content': message['content'],
            }
    else:
        raise MessageError(f'invalid message type: {type(message)}')


def parse_openai_model_reponse(response: ModelResponse) -> Messages:
    function_call: OpenaiFunctionCall = response['choices'][0]['message'].get('function_call')
    try:
        if function_call:
            return [
                FunctionCallMessage(
                    role='assistant',
                    content={
                        'name': function_call['name'],
                        'arguments': function_call['arguments'],
                    },
                )
            ]
        else:
            text: str = response['choices'][0]['message']['content']
            return [
                TextMessage(
                    role='assistant',
                    content=text,
                )
            ]
    except (KeyError, IndexError) as e:
        raise UnexpectedResponseError(response) from e


class OpenAIChat(HttpChatModel[OpenAIChatParameters]):
    model_type: ClassVar[str] = 'openai'
    support_stream: ClassVar[bool] = True
    default_api_base: ClassVar[str] = 'https://api.openai.com/v1'

    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        api_key: str | None = None,
        api_base: str | None = None,
        system_prompt: str | None = None,
        parameters: OpenAIChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelKwargs],
    ):
        parameters = parameters or OpenAIChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = model
        self.system_prompt = system_prompt
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or self.default_api_base
        self.api_key = api_key or os.environ['OPENAI_API_KEY']

    @override
    def _get_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        openai_messages = [convert_to_openai_message(message) for message in messages]
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        parameters_dict = parameters.model_dump(exclude_defaults=True)
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

    @override
    def _parse_reponse(self, response: ModelResponse) -> Messages:
        return parse_openai_model_reponse(response)

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: OpenAIChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def _parse_stream_response(self, response: ModelResponse) -> Stream:
        if response.get('data') == '[DONE]':
            return Stream(delta='', control='done')
        else:
            delta = response['choices'][0]['delta']
            if 'role' in delta:
                return Stream(delta='', control='start')
            elif 'content' in delta:
                return Stream(delta=delta['content'], control='continue')
            else:
                return Stream(delta='', control='finish')

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any):
        return cls(model=name, **kwargs)
