from __future__ import annotations

import json
from typing import Any, Generic, List, Optional, TypeVar, cast

from lmclient.models import BaseChatModel, load_from_model_id
from lmclient.types import ChatModelOutput, FunctionCallDict, GeneralParameters, Message, Messages, ModelParameters
from lmclient.utils import function

T_P = TypeVar('T_P', bound=ModelParameters)
T_O = TypeVar('T_O', bound=ChatModelOutput)


class ChatEngine(Generic[T_P, T_O]):
    def __init__(
        self,
        chat_model: BaseChatModel[T_P, T_O] | str,
        temperature: float = 1,
        top_p: float = 1,
        functions: Optional[List[function]] = None,
        function_call_raise_error: bool = False,
        **extra_parameters: Any,
    ):
        if isinstance(chat_model, str):
            self._chat_model: BaseChatModel[T_P, T_O] = load_from_model_id(chat_model)  # type: ignore
        else:
            self._chat_model = chat_model

        self.functions = functions or []
        if functions:
            functions_schema = [function.schema for function in functions]
            function_call = 'auto'
        else:
            functions_schema = None
            function_call = None

        self.engine_parameters = GeneralParameters(
            temperature=temperature,
            top_p=top_p,
            functions=functions_schema,
            function_call=function_call,
        )
        self._extra_parameters = extra_parameters
        _parameters = self._chat_model.parameters_type.from_general_parameters(self.engine_parameters).model_copy(
            update=self._extra_parameters
        )
        self._parameters = cast(T_P, _parameters)
        self.function_call_raise_error = function_call_raise_error
        self.history: Messages = []

    @property
    def extra_parameters(self) -> dict[str, Any]:
        return self._extra_parameters

    @extra_parameters.setter
    def extra_parameters(self, extra_parameters: dict[str, Any]):
        self._extra_parameters = extra_parameters
        self._parameters = self._parameters.model_copy(update=self._extra_parameters)

    @property
    def chat_model(self):
        return self._chat_model

    def chat(self, user_input: str, **extra_parameters: Any) -> str:
        parameters = self._parameters.model_copy(update=extra_parameters)
        self.history.append(Message(role='user', content=user_input))
        model_response = self._chat_model.chat_completion(self.history, parameters)
        self.history.extend(model_response.messages)
        if isinstance(reply := model_response.messages[-1].content, str):
            return reply
        else:
            return self._recursive_function_call(reply, parameters)

    async def async_chat(self, user_input: str, **extra_parameters: Any) -> str:
        parameters = self._parameters.model_copy(update=extra_parameters)
        self.history.append(Message(role='user', content=user_input))
        model_response = await self._chat_model.async_chat_completion(self.history, parameters)
        self.history.extend(model_response.messages)
        if isinstance(reply := model_response.messages[-1].content, str):
            return reply
        else:
            return await self._async_recursive_function_call(reply, parameters)

    def run_function_call(self, function_call: FunctionCallDict):
        function = None
        for i in self.functions:
            if i.name == function_call['name']:
                function = i
        if function is None:
            if self.function_call_raise_error:
                raise ValueError(f'Function {function_call["name"]} not found')
            else:
                return 'Function not found, please try another function.'

        try:
            arguments = json.loads(function_call['arguments'], strict=False)
            return function(**arguments)
        except Exception as e:
            if self.function_call_raise_error:
                raise e
            else:
                return f'Error: {e}'

    def _recursive_function_call(self, function_call: FunctionCallDict, parameters: T_P) -> str:
        function_output = self.run_function_call(function_call)
        self.history.append(
            Message(role='function', name=function_call['name'], content=json.dumps(function_output, ensure_ascii=False))
        )
        model_response = self._chat_model.chat_completion(self.history, parameters)
        self.history.extend(model_response.messages)
        if isinstance(reply := model_response.messages[-1].content, str):
            return reply
        else:
            return self._recursive_function_call(reply, parameters)

    async def _async_recursive_function_call(self, function_call: FunctionCallDict, parameters: T_P) -> str:
        function_output = self.run_function_call(function_call)
        self.history.append(
            Message(role='function', name=function_call['name'], content=json.dumps(function_output, ensure_ascii=False))
        )
        model_response = await self._chat_model.async_chat_completion(self.history, parameters)
        self.history.extend(model_response.messages)
        if isinstance(reply := model_response.messages[-1].content, str):
            return reply
        else:
            return self._recursive_function_call(reply, parameters)

    def reset(self) -> None:
        self.history.clear()
