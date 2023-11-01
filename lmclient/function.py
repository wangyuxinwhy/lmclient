# type: ignore
from __future__ import annotations

import json
from functools import wraps
from typing import Any, Callable

from docstring_parser import parse
from pydantic import BaseModel, validate_arguments

from lmclient.exceptions import MessageError
from lmclient.types import Function, Message


class function:
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.name = self.func.__name__
        self.validate_func = validate_arguments(func)
        self.docstring = parse(self.func.__doc__ or '')

        self._validate_func_model: BaseModel = self.validate_func.model
        try:
            parameters = self._validate_func_model.model_json_schema()
        except AttributeError:
            parameters = self._validate_func_model.schema()

        parameters['properties'] = {
            k: v for k, v in parameters['properties'].items() if k not in ('v__duplicate_kwargs', 'args', 'kwargs')
        }
        for param in self.docstring.params:
            if (name := param.arg_name) in parameters['properties'] and (description := param.description):
                parameters['properties'][name]['description'] = description
        parameters['required'] = sorted(k for k, v in parameters['properties'].items() if 'default' not in v)
        _remove_a_key(parameters, 'additionalProperties')
        _remove_a_key(parameters, 'title')
        self.json_schema: Function = {
            'name': self.name,
            'description': self.docstring.short_description or '',
            'parameters': parameters,
        }
        self.model = self.validate_func.model

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        @wraps(self.func)
        def wrapper(*args, **kwargs):
            return self.validate_func(*args, **kwargs)

        return wrapper(*args, **kwargs)

    def from_message(self, message: Message):
        function_call = message.content
        if isinstance(function_call, str):
            raise MessageError(f'{message} is not a valid function call message')
        arguments = json.loads(function_call['arguments'], strict=False)
        return self.validate_func(**arguments)


def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)
