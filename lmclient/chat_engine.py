from __future__ import annotations

import json
from typing import List, Optional

from lmclient.models.base import BaseChatModel
from lmclient.types import FunctionCallDict, GeneralParameters, Message, Messages, ModelParameters
from lmclient.utils import lm_function


class ChatEngine:
    def __init__(
        self,
        chat_model: BaseChatModel,
        temperature: float = 1,
        top_p: float = 1,
        functions: Optional[List[lm_function]] = None,
        function_call_raise_error: bool = False,
        **extra_model_parameters
    ):
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
        self._extra_model_parameters = extra_model_parameters
        self._model_parameters: ModelParameters = self._chat_model.parameters_type.from_general_parameters(self.engine_parameters).model_copy(update=self._extra_model_parameters)
        self.function_call_raise_error = function_call_raise_error
        self.history: Messages = []

    @property
    def chat_model(self):
        return self._chat_model

    @chat_model.setter
    def chat_model(self, model: BaseChatModel):
        self._chat_model = model
        self._model_parameters = self._chat_model.parameters_type.from_general_parameters(self.engine_parameters).model_copy(update=self._extra_model_parameters)

    @property
    def model_parameters(self):
        return self._model_parameters

    @property
    def extra_model_parameters(self):
        return self._extra_model_parameters

    @extra_model_parameters.setter
    def extra_model_parameters(self, extra_model_parameters: dict):
        self._extra_model_parameters = extra_model_parameters
        self._model_parameters = self.model_parameters.model_copy(update=self._extra_model_parameters)

    def chat(self, user_input: str, **extra_model_parameters) -> str:
        model_parameters = self.model_parameters.model_copy(update=extra_model_parameters)
        self.history.append(Message(role='user', content=user_input))
        model_response = self.chat_model.chat_completion(self.history, model_parameters)
        self.history.extend(model_response.messages)
        if isinstance(reply := model_response.messages[-1].content, str):
            return reply
        else:
            return self._recursive_function_call(reply, model_parameters)

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
            arguments = json.loads(function_call["arguments"], strict=False)
            return function(**arguments)
        except Exception as e:
            if self.function_call_raise_error:
                raise e
            else:
                return f'Error: {e}'

    def _recursive_function_call(self, function_call: FunctionCallDict, model_parameters: ModelParameters) -> str:
        function_output = self.run_function_call(function_call)
        self.history.append(Message(role='function', name=function_call['name'], content=json.dumps(function_output, ensure_ascii=False)))
        model_response = self.chat_model.chat_completion(self.history, model_parameters)
        self.history.extend(model_response.messages)
        if isinstance(reply := model_response.messages[-1].content, str):
            return reply
        else:
            return self._recursive_function_call(reply, model_parameters)

    def reset(self) -> None:
        self.history.clear()
