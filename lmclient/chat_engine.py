from __future__ import annotations

import json
from typing import Any, Generic, List, Optional, TypedDict, TypeVar

from typing_extensions import Self, Unpack

from lmclient.function import function
from lmclient.models import BaseChatModel, OverrideParameters, load_from_model_id
from lmclient.types import (
    ChatModelOutput,
    FunctionCall,
    GeneralParameters,
    Message,
    ModelParameters,
    TextMessage,
)
from lmclient.utils import is_function_call_message, is_text_message

T_P = TypeVar('T_P', bound=ModelParameters)
T_O = TypeVar('T_O', bound=ChatModelOutput)


class ChatEngineKwargs(TypedDict, total=False):
    temperature: float
    top_p: float
    functions: Optional[List[function]]
    function_call_raise_error: bool
    max_function_calls_per_turn: int


class ChatEngine(Generic[T_P, T_O]):
    def __init__(
        self,
        chat_model: BaseChatModel[T_P, T_O],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        functions: Optional[List[function]] = None,
        function_call_raise_error: bool = False,
        max_function_calls_per_turn: int = 5,
    ):
        self._chat_model = chat_model
        self._functions = functions or []
        self._function_map = {function.name: function for function in self._functions}
        if functions:
            functions_schema = [function.json_schema for function in functions]
        else:
            functions_schema = None
        self.engine_parameters = GeneralParameters(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            functions=functions_schema,
        )
        self._chat_model.parameters.update_with_general_parameters(self.engine_parameters)
        self.function_call_raise_error = function_call_raise_error
        self.max_function_calls_per_turn = max_function_calls_per_turn
        self.history: list[Message] = []
        self._function_call_count = 0

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Unpack[ChatEngineKwargs]) -> Self:
        chat_model = load_from_model_id(model_id)
        return cls(chat_model, **kwargs)

    @property
    def chat_model(self):
        return self._chat_model

    def chat(self, user_input: str, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any) -> str:
        self._function_call_count = 0

        self.history.append(TextMessage(role='user', content=user_input))
        model_response = self._chat_model.chat_completion(self.history, override_parameters, **kwargs)
        self.history.extend(model_response.messages)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            reply = last_message['content']
            return reply
        elif is_function_call_message(last_message):
            function_call = last_message['content']
            return self._recursive_function_call(function_call, override_parameters, **kwargs)
        else:
            raise RuntimeError(f'Invalid message type: {type(last_message)}')

    async def async_chat(self, user_input: str, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any) -> str:
        self._function_call_count = 0

        self.history.append(TextMessage(role='user', content=user_input))
        model_response = await self._chat_model.async_chat_completion(self.history, override_parameters, **kwargs)
        self.history.extend(model_response.messages)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            reply = last_message['content']
            return reply
        elif is_function_call_message(last_message):
            function_call = last_message['content']
            return await self._async_recursive_function_call(function_call, override_parameters, **kwargs)
        else:
            raise RuntimeError(f'Invalid message type: {type(last_message)}')

    def run_function_call(self, function_call: FunctionCall):
        function = self._function_map.get(function_call['name'])
        if function is None:
            if self.function_call_raise_error:
                raise ValueError(f'Function {function_call["name"]} not found')
            else:
                return 'Function not found, please try another function.'

        try:
            arguments = json.loads(function_call['arguments'], strict=False)
            function_call_return = function(**arguments)
            return function_call_return
        except Exception as e:
            if self.function_call_raise_error:
                raise e
            else:
                return f'Error: {e}'

    def _recursive_function_call(
        self, function_call: FunctionCall, override_parameters: OverrideParameters[T_P] = None, **kwargs
    ) -> str:
        function_output = self.run_function_call(function_call)
        self.history.append(
            TextMessage(role='function', name=function_call['name'], content=json.dumps(function_output, ensure_ascii=False))
        )
        model_response = self._chat_model.chat_completion(self.history, override_parameters, **kwargs)
        self.history.extend(model_response.messages)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            reply = last_message['content']
            return reply
        elif is_function_call_message(last_message):
            self._function_call_count += 1
            if self._function_call_count > self.max_function_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = last_message['content']
            return self._recursive_function_call(function_call, override_parameters, **kwargs)
        else:
            raise RuntimeError(f'Invalid message type: {type(last_message)}')

    async def _async_recursive_function_call(
        self, function_call: FunctionCall, override_parameters: OverrideParameters[T_P] = None, **kwargs
    ) -> str:
        function_output = self.run_function_call(function_call)
        self.history.append(
            TextMessage(role='function', name=function_call['name'], content=json.dumps(function_output, ensure_ascii=False))
        )
        model_response = await self._chat_model.async_chat_completion(self.history, override_parameters, **kwargs)
        self.history.extend(model_response.messages)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            reply = last_message['content']
            return reply
        elif is_function_call_message(last_message):
            self._function_call_count += 1
            if self._function_call_count > self.max_function_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = last_message['content']
            return await self._async_recursive_function_call(function_call, override_parameters, **kwargs)
        else:
            raise RuntimeError(f'Invalid message type: {type(last_message)}')

    def reset(self) -> None:
        self.history.clear()
