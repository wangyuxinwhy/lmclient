# type: ignore
from __future__ import annotations

import hashlib
import json
from functools import wraps
from typing import Any, Callable

from docstring_parser import parse
from pydantic import BaseModel, validate_arguments

from lmclient.exceptions import MessageError
from lmclient.types import FunctionDict, Message, Messages, ModelParameters
from lmclient.version import __cache_version__


def get_pydantic_version():
    import pydantic
    from packaging import version

    return version.parse(pydantic.__version__).major


PydanticVersion = get_pydantic_version()


def generate_chat_completion_hash_key(model_id: str, messages: Messages, parameters: ModelParameters) -> str:
    messages_text = '---'.join([f'{k}={v}' for message in messages for k, v in to_dict(message).items()])
    messages_hash = md5_hash(messages_text)
    parameters_hash = md5_hash(parameters.model_dump_json(exclude_none=True))
    return f'{model_id}|{messages_hash}|{parameters_hash}|v{__cache_version__}'


def md5_hash(string: str) -> str:
    return hashlib.md5(string.encode()).hexdigest()


def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


class function:
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.name = self.func.__name__
        self.validate_func = validate_arguments(func)
        self.docstring = parse(self.func.__doc__ or '')

        parameters = self.validate_func.model.model_json_schema()
        parameters['properties'] = {
            k: v for k, v in parameters['properties'].items() if k not in ('v__duplicate_kwargs', 'args', 'kwargs')
        }
        for param in self.docstring.params:
            if (name := param.arg_name) in parameters['properties'] and (description := param.description):
                parameters['properties'][name]['description'] = description
        parameters['required'] = sorted(k for k, v in parameters['properties'].items() if 'default' not in v)
        _remove_a_key(parameters, 'additionalProperties')
        _remove_a_key(parameters, 'title')
        self.schema: FunctionDict = {
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


class BaseSchema(BaseModel):
    @classmethod
    @property
    def openai_schema(cls):
        schema = cls.model_json_schema()
        docstring = parse(cls.__doc__ or '')
        parameters = {k: v for k, v in schema.items() if k not in ('title', 'description')}
        for param in docstring.params:
            if (name := param.arg_name) in parameters['properties'] and (description := param.description):
                if 'description' not in parameters['properties'][name]:
                    parameters['properties'][name]['description'] = description

        parameters['required'] = sorted(k for k, v in parameters['properties'].items() if 'default' not in v)

        if 'description' not in schema:
            if docstring.short_description:
                schema['description'] = docstring.short_description
            else:
                schema['description'] = (
                    f'Correctly extracted `{cls.__name__}` with all ' f'the required parameters with correct types'
                )

        _remove_a_key(parameters, 'additionalProperties')
        _remove_a_key(parameters, 'title')
        return {
            'name': schema['title'],
            'description': schema['description'],
            'parameters': parameters,
        }

    @classmethod
    def from_message(cls, message: Message):
        function_call = message.content
        if isinstance(function_call, str):
            raise MessageError(f'{message} is not a valid function call message')
        arguments = json.loads(function_call['arguments'], strict=False)
        return cls(**arguments)


def to_dict(value: BaseModel, exclude_defaults: bool = False, exclude_none: bool = False):
    if PydanticVersion == 2:
        return value.model_dump(exclude_defaults=exclude_defaults, exclude_none=exclude_none)
    else:
        return value.dict(exclude_defaults=exclude_defaults, exclude_none=exclude_none)
