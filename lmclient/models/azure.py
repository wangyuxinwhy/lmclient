from __future__ import annotations

import os
from typing import Any, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, PositiveInt
from typing_extensions import Annotated, Self, Unpack, override

from lmclient.message import Messages
from lmclient.model_output import ChatModelOutput, Stream
from lmclient.models.http import HttpChatModel, HttpChatModelInitKwargs, HttpResponse, HttpxPostKwargs
from lmclient.models.openai import (
    FunctionCallName,
    convert_to_openai_message,
    parse_openai_model_reponse,
)
from lmclient.types import FunctionJsonSchema, Probability, Temperature


class AzureChatParameters(BaseModel):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    functions: Optional[List[FunctionJsonSchema]] = None
    function_call: Union[Literal['auto'], FunctionCallName, None] = None
    stop: Union[str, List[str], None] = None
    presence_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    frequency_penalty: Optional[Annotated[float, Field(ge=-2, le=2)]] = None
    logit_bias: Optional[Dict[int, Annotated[int, Field(ge=-100, le=100)]]] = None
    user: Optional[str] = None


class AzureChat(HttpChatModel[AzureChatParameters]):
    model_type: ClassVar[str] = 'azure'

    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        parameters: AzureChatParameters | None = None,
        **kwargs: Unpack[HttpChatModelInitKwargs],
    ) -> None:
        parameters = parameters or AzureChatParameters()
        super().__init__(parameters=parameters, **kwargs)
        self.model = model or os.environ['AZURE_CHAT_API_ENGINE'] or os.environ['AZURE_CHAT_MODEL_NAME']
        self.system_prompt = system_prompt
        self.api_key = api_key or os.environ['AZURE_API_KEY']
        self.api_base = api_base or os.environ['AZURE_API_BASE']
        self.api_version = api_version or os.getenv('AZURE_API_VERSION')

    @override
    def _get_request_parameters(self, messages: Messages, parameters: AzureChatParameters) -> HttpxPostKwargs:
        openai_messages = [convert_to_openai_message(message) for message in messages]
        if self.system_prompt:
            openai_messages.insert(0, {'role': 'system', 'content': self.system_prompt})
        parameters_dict = {**parameters.model_dump(exclude_none=True), **parameters.model_dump(exclude_unset=True)}
        json_data = {
            'model': self.model,
            'messages': openai_messages,
            **parameters_dict,
        }
        headers = {
            'api-key': self.api_key,
        }
        return {
            'url': f'{self.api_base}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}',
            'headers': headers,
            'json': json_data,
        }

    @override
    def _get_stream_request_parameters(self, messages: Messages, parameters: AzureChatParameters) -> HttpxPostKwargs:
        http_parameters = self._get_request_parameters(messages, parameters)
        http_parameters['json']['stream'] = True
        return http_parameters

    @override
    def _parse_stream_response(self, response: HttpResponse) -> Stream:
        raise NotImplementedError('Azure does not support streaming')

    @override
    def _parse_reponse(self, response: HttpResponse) -> ChatModelOutput:
        return parse_openai_model_reponse(response)

    @property
    @override
    def name(self) -> str:
        return self.model

    @classmethod
    @override
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
