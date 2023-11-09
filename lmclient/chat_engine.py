from __future__ import annotations

import json
from typing import Any, Generic, List, Literal, Optional, TypedDict, TypeVar

from typing_extensions import Self, Unpack

from lmclient.function import function
from lmclient.models import BaseChatModel, OverrideParameters, load_from_model_id
from lmclient.printer import Printer, SimplePrinter
from lmclient.types import (
    ChatModelOutput,
    FunctionCall,
    FunctionCallMessage,
    GeneralParameters,
    Message,
    ModelParameters,
    TextMessage,
)

T_P = TypeVar('T_P', bound=ModelParameters)


class ChatEngineKwargs(TypedDict, total=False):
    temperature: float
    top_p: float
    functions: Optional[List[function]]
    function_call_raise_error: bool
    max_function_calls_per_turn: int


class ChatEngine(Generic[T_P]):
    """
    A chat engine that uses a chat model to generate responses.It will manage the chat history and function calls.

    Args:
        chat_model (BaseChatModel[T_P]): The chat model to use for generating responses.
        temperature (float | None, optional): Controls the randomness of the generated responses. Defaults to None.
        top_p (float | None, optional): Controls the diversity of the generated responses. Defaults to None.
        max_tokens (int | None, optional): Controls the length of the generated responses. Defaults to None.
        functions (Optional[List[function]], optional): A list of functions to use for function call. Defaults to None.
        function_call_raise_error (bool, optional): Whether to raise an error if a function call fails. Defaults to False.
        max_function_calls_per_turn (int, optional): The maximum number of function calls allowed per turn. Defaults to 5.
        stream (bool, optional): Whether to stream the generated text rely. Defaults to True.
        printer (Printer | Literal['auto'] | None, optional): The printer to use for displaying the generated responses. Defaults to 'auto'.
    """

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
        stream: bool | Literal['auto'] = 'auto',
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

        if stream == 'auto':
            self.stream = not bool(self._functions)
        else:
            if self._functions and stream:
                raise ValueError('Cannot stream when functions are provided.')
            self.stream = stream

        if printer == 'auto':
            self.printer = SimplePrinter() if stream else None
        else:
            self.printer = printer

        self.history: list[Message] = []
        self.model_ouptuts: list[ChatModelOutput[T_P]] = []
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
            model_output = self._stream_chat_helper(override_parameters, **kwargs)
            if model_output is None:
                raise RuntimeError('Stream finished unexpectedly.')
        else:
            model_output = self._chat_model.chat_completion(self.history, override_parameters=override_parameters, **kwargs)
        self.handle_model_output(model_output)

        if isinstance(model_output.last_message, TextMessage):
            return model_output.last_message.content

        if isinstance(model_output.last_message, FunctionCallMessage):
            function_call = model_output.last_message.content
            return self._recursive_function_call(function_call, override_parameters, **kwargs)

        raise RuntimeError(f'Invalid message type: {type(model_output.last_message)}')

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

    async def async_chat(
        self, user_input: str, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> str | None:
        self._function_call_count = 0

        user_input_message = TextMessage(role='user', content=user_input)
        self.history.append(user_input_message)
        if self.printer:
            self.printer.print_message(user_input_message)

        if self.stream:
            model_output = await self._async_stream_chat_helper(override_parameters, **kwargs)
            if model_output is None:
                raise RuntimeError('Stream finished unexpectedly.')
        else:
            model_output = await self._chat_model.async_chat_completion(
                self.history, override_parameters=override_parameters, **kwargs
            )
        self.handle_model_output(model_output)

        if isinstance(model_output.last_message, TextMessage):
            return model_output.last_message.content

        if isinstance(model_output.last_message, FunctionCallMessage):
            function_call = model_output.last_message.content
            return await self._async_recursive_function_call(function_call, override_parameters, **kwargs)

        raise RuntimeError(f'Invalid message type: {type(model_output.last_message)}')

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
        function = self._function_map.get(function_call.name)
        if function is None:
            if self.function_call_raise_error:
                raise ValueError(f'Function {function_call.name} not found')

            return 'Function not found, please try another function.'

        try:
            arguments = json.loads(function_call.arguments, strict=False)
            return function(**arguments)
        except Exception as e:
            if self.function_call_raise_error:
                raise

            return str(e)

    def _recursive_function_call(
        self, function_call: FunctionCall, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> str:
        function_output = self.run_function_call(function_call)
        function_message = TextMessage(
            role='function', name=function_call.name, content=json.dumps(function_output, ensure_ascii=False)
        )
        self.history.append(function_message)
        if self.printer:
            self.printer.print_message(function_message)

        model_output = self._chat_model.chat_completion(self.history, override_parameters=override_parameters, **kwargs)
        self.handle_model_output(model_output)

        if isinstance(model_output.last_message, TextMessage):
            return model_output.last_message.content

        if isinstance(model_output.last_message, FunctionCallMessage):
            self._function_call_count += 1
            if self._function_call_count > self.max_function_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = model_output.last_message.content
            return self._recursive_function_call(function_call, override_parameters, **kwargs)

        raise RuntimeError(f'Invalid message type: {type(model_output.last_message)}')

    async def _async_recursive_function_call(
        self, function_call: FunctionCall, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> str:
        function_output = self.run_function_call(function_call)
        function_message = TextMessage(
            role='function', name=function_call.name, content=json.dumps(function_output, ensure_ascii=False)
        )
        self.history.append(function_message)
        if self.printer:
            self.printer.print_message(function_message)

        model_output = await self._chat_model.async_chat_completion(self.history, override_parameters, **kwargs)
        self.handle_model_output(model_output)

        if isinstance(model_output.last_message, TextMessage):
            return model_output.last_message.content

        if isinstance(model_output.last_message, FunctionCallMessage):
            self._function_call_count += 1
            if self._function_call_count > self.max_function_calls_per_turn:
                raise RuntimeError('Maximum number of function calls reached.')
            function_call = model_output.last_message.content
            return await self._async_recursive_function_call(function_call, override_parameters, **kwargs)

        raise RuntimeError(f'Invalid message type: {type(model_output.last_message)}')

    def handle_model_output(self, model_output: ChatModelOutput) -> None:
        self.model_ouptuts.append(model_output)
        self.history.extend(model_output.messages)
        if self.printer:
            for message in model_output.messages:
                if self.stream and message.role == 'assistant':
                    continue
                self.printer.print_message(message)

    def reset(self) -> None:
        self.history.clear()
