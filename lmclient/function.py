# type: ignore
from __future__ import annotations

import json
from functools import wraps
from typing import Any, Callable, TypeVar

from docstring_parser import parse
from pydantic import TypeAdapter, validate_call
from typing_extensions import ParamSpec

from lmclient.types import Function, Message
from lmclient.utils import is_function_call_message

P = ParamSpec('P')
T = TypeVar('T')


class function:  # noqa: N801
    def __init__(self, function: Callable[P, T]) -> None:
        self.function = validate_call(function)
        self.name = self.function.__name__
        self.docstring = parse(self.function.__doc__ or '')
        parameters = TypeAdapter(function).json_schema()
        for param in self.docstring.params:
            if (name := param.arg_name) in parameters['properties'] and (description := param.description):
                parameters['properties'][name]['description'] = description
        parameters['required'] = sorted(k for k, v in parameters['properties'].items() if 'default' not in v)
        recusive_remove(parameters, 'additionalProperties')
        recusive_remove(parameters, 'title')
        self.json_schema: Function = {
            'name': self.name,
            'description': self.docstring.short_description or '',
            'parameters': parameters,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        @wraps(self.function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return self.function(*args, **kwargs)

        return wrapper(*args, **kwargs)

    def call_with_message(self, message: Message) -> T:
        if is_function_call_message(message):
            function_call = message['content']
            arguments = json.loads(function_call['arguments'], strict=False)
            return self.function(**arguments)
        raise ValueError(f'message is not a function call: {message}')


def recusive_remove(obj: Any, remove_key: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == remove_key:
                del obj[key]
            else:
                recusive_remove(obj[key], remove_key)
