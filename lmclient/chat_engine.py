from __future__ import annotations

import json
from typing import Any, Generic, List, Literal, Optional, TypedDict, TypeVar

from typing_extensions import Self, Unpack, overload

from lmclient.function import function
from lmclient.models import BaseChatModel, OverrideParameters, load_from_model_id
from lmclient.printer import Printer, SimplePrinter
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


class ChatEngineKwargs(TypedDict, total=False):
    temperature: float
    top_p: float
    functions: Optional[List[function]]
    function_call_raise_error: bool
    max_function_calls_per_turn: int


class ChatEngine(Generic[T_P]):
    printer: Printer | None

    def __init__(
        self,
        chat_model: BaseChatModel[T_P],
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
        functions: Optional[List[function]] = None,
        function_call_raise_error: bool = False,
        max_function_calls_per_turn: int = 5,
        stream: bool = True,
        printer: Printer | Literal['auto'] | None = 'auto',
    ) -> None:
        self._chat_model = chat_model
        self._functions = functions or []
        self._function_map = {function.name: function for function in self._functions}
        functions_schema = [function.json_schema for function in functions] if functions else None
        self.engine_parameters = GeneralParameters(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            functions=functions_schema,
        )
        self._chat_model.parameters.update_with_general_parameters(self.engine_parameters)
        self.function_call_raise_error = function_call_raise_error
        self.max_function_calls_per_turn = max_function_calls_per_turn
        self.stream = stream

        if printer == 'auto':
            self.printer = SimplePrinter() if stream else None
        else:
            self.printer = printer

        self.history: list[Message] = []
        self._function_call_count = 0

    @classmethod
    def from_model_id(cls, model_id: str, **kwargs: Unpack[ChatEngineKwargs]) -> Self:
        chat_model = load_from_model_id(model_id)
        return cls(chat_model, **kwargs)

    @property
    def chat_model(self) -> BaseChatModel[T_P]:
        return self._chat_model

    @property
    def print_message(self) -> bool:
        return self.printer is not None

    def chat(self, user_input: str, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any) -> str:
        self._function_call_count = 0

        user_input_message = TextMessage(role='user', content=user_input)
        self.history.append(user_input_message)
        if self.printer:
            self.printer.print_message(user_input_message)

        if self.stream:
            model_response = self._stream_chat_helper(override_parameters, **kwargs)
            if model_response is None:
                raise RuntimeError('Stream finished unexpectedly.')
        else:
            model_response = self._chat_model.chat_completion(self.history, override_parameters=override_parameters, **kwargs)
            if self.printer:
                for message in model_response.messages:
                    self.printer.print_message(message)
        self.history.extend(model_response.messages)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            return last_message['content']

        if is_function_call_message(last_message):
            function_call = last_message['content']
            return self._recursive_function_call(function_call, override_parameters, **kwargs)

        raise RuntimeError(f'Invalid message type: {type(last_message)}')

    def _stream_chat_helper(
        self, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> ChatModelOutput[T_P] | None:
        for stream_output in self._chat_model.stream_chat_completion(
            self.history, override_parameters=override_parameters, **kwargs
        ):
            if self.printer:
                self.printer.print_stream(stream_output.stream)
            if stream_output.is_finish:
                return stream_output
        return None

    @overload
    async def async_chat(
        self,
        user_input: str,
        override_parameters: OverrideParameters[T_P] = None,
        return_reply: Literal[False] = False,
        **kwargs: Any,
    ) -> None:
        ...

    @overload
    async def async_chat(
        self,
        user_input: str,
        override_parameters: OverrideParameters[T_P] = None,
        return_reply: Literal[True] = True,
        **kwargs: Any,
    ) -> str:
        ...

    async def async_chat(
        self, user_input: str, override_parameters: OverrideParameters[T_P] = None, return_reply: bool = False, **kwargs: Any
    ) -> str | None:
        self._function_call_count = 0

        user_input_message = TextMessage(role='user', content=user_input)
        self.history.append(user_input_message)
        if self.printer:
            self.printer.print_message(user_input_message)

        if self.stream:
            model_response = await self._async_stream_chat_helper(override_parameters, **kwargs)
            if model_response is None:
                raise RuntimeError('Stream finished unexpectedly.')
        else:
            model_response = await self._chat_model.async_chat_completion(
                self.history, override_parameters=override_parameters, **kwargs
            )
        self.history.extend(model_response.messages)
        if self.printer:
            for message in model_response.messages:
                self.printer.print_message(message)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            reply = last_message['content']
            if return_reply:
                return reply
            return None
        if is_function_call_message(last_message):
            function_call = last_message['content']
            reply = await self._async_recursive_function_call(function_call, override_parameters, **kwargs)
            if return_reply:
                return reply
            return None
        raise RuntimeError(f'Invalid message type: {type(last_message)}')

    async def _async_stream_chat_helper(
        self, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> ChatModelOutput[T_P] | None:
        async for stream_output in self._chat_model.async_stream_chat_completion(
            self.history, override_parameters=override_parameters, **kwargs
        ):
            if self.printer:
                self.printer.print_stream(stream_output.stream)
            if stream_output.is_finish:
                return stream_output
        return None

    def run_function_call(self, function_call: FunctionCall) -> Any | str:
        function = self._function_map.get(function_call['name'])
        if function is None:
            if self.function_call_raise_error:
                raise ValueError(f'Function {function_call["name"]} not found')

            return 'Function not found, please try another function.'

        try:
            arguments = json.loads(function_call['arguments'], strict=False)
            return function(**arguments)
        except Exception as e:
            if self.function_call_raise_error:
                raise

            return f'Error: {e}'

    def _recursive_function_call(
        self, function_call: FunctionCall, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> str:
        function_output = self.run_function_call(function_call)
        function_message = TextMessage(
            role='function', name=function_call['name'], content=json.dumps(function_output, ensure_ascii=False)
        )
        self.history.append(function_message)
        if self.printer:
            self.printer.print_message(function_message)

        model_response = self._chat_model.chat_completion(self.history, override_parameters=override_parameters, **kwargs)
        self.history.extend(model_response.messages)
        if self.printer:
            for message in model_response.messages:
                self.printer.print_message(message)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            return last_message['content']
        if is_function_call_message(last_message):
            self._function_call_count += 1
            if self._function_call_count > self.max_function_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = last_message['content']
            return self._recursive_function_call(function_call, override_parameters, **kwargs)

        raise RuntimeError(f'Invalid message type: {type(last_message)}')

    async def _async_recursive_function_call(
        self, function_call: FunctionCall, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> str:
        function_output = self.run_function_call(function_call)
        function_message = TextMessage(
            role='function', name=function_call['name'], content=json.dumps(function_output, ensure_ascii=False)
        )
        self.history.append(function_message)
        if self.printer:
            self.printer.print_message(function_message)

        model_response = await self._chat_model.async_chat_completion(self.history, override_parameters, **kwargs)
        self.history.extend(model_response.messages)
        if self.printer:
            for message in model_response.messages:
                self.printer.print_message(message)

        last_message = model_response.messages[-1]
        if is_text_message(last_message):
            return last_message['content']
        if is_function_call_message(last_message):
            self._function_call_count += 1
            if self._function_call_count > self.max_function_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = last_message['content']
            return await self._async_recursive_function_call(function_call, override_parameters, **kwargs)

        raise RuntimeError(f'Invalid message type: {type(last_message)}')

    def reset(self) -> None:
        self.history.clear()
