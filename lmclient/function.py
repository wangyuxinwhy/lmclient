# type: ignore
from __future__ import annotations

import json
from functools import wraps
from typing import Any, Callable

from docstring_parser import parse
from pydantic import TypeAdapter, validate_call

from lmclient.types import Function, Message
from lmclient.utils import is_function_call_message


class function:
    def __init__(self, callable: Callable) -> None:
        self.callable = validate_call(callable)
        self.name = self.callable.__name__
        self.docstring = parse(self.callable.__doc__ or '')
        parameters = TypeAdapter(callable).json_schema()
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
        @wraps(self.callable)
        def wrapper(*args, **kwargs):
            return self.callable(*args, **kwargs)
        return wrapper(*args, **kwargs)

    def call_with_message(self, message: Message):
        if is_function_call_message(message):
            function_call = message['content']
            arguments = json.loads(function_call['arguments'], strict=False)
            return self.callable(**arguments)
        raise ValueError(f'message is not a function call: {message}')

def recusive_remove(object: Any, remove_key: str) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(object, dict):
        for key in list(object.keys()):
            if key == remove_key:
                del object[key]
            else:
                recusive_remove(object[key], remove_key)
