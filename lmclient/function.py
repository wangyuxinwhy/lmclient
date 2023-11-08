from __future__ import annotations

import json
from typing import Callable, Generic, TypeVar

from docstring_parser import parse
from pydantic import TypeAdapter, validate_call
from typing_extensions import ParamSpec

from lmclient.types import Function, Message
from lmclient.utils import is_function_call_message

P = ParamSpec('P')
T = TypeVar('T')


class function(Generic[P, T]):  # noqa: N801
    """
    A decorator class that wraps a callable function and provides additional functionality.

    Args:
        function (Callable[P, T]): The function to be wrapped.

    Attributes:
        function (Callable[P, T]): The wrapped function.
        name (str): The name of the wrapped function.
        docstring (ParsedDocstring): The parsed docstring of the wrapped function.
        json_schema (Function): The JSON schema of the wrapped function.

    Methods:
        __call__(self, *args: Any, **kwargs: Any) -> Any: Calls the wrapped function with the provided arguments.
        call_with_message(self, message: Message) -> T: Calls the wrapped function with the arguments provided in the message.
    """

    def __init__(self, function: Callable[P, T]) -> None:
        self.function: Callable[P, T] = validate_call(function)
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

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        return self.function(*args, **kwargs)

    def call_with_message(self, message: Message) -> T:
        if is_function_call_message(message):
            function_call = message['content']
            arguments = json.loads(function_call['arguments'], strict=False)
            return self.function(**arguments)  # type: ignore
        raise ValueError(f'message is not a function call: {message}')


def recusive_remove(dictionary: dict, remove_key: str) -> None:
    """
    Recursively removes a key from a dictionary and all its nested dictionaries.

    Args:
        dictionary (dict): The dictionary to remove the key from.
        remove_key (str): The key to remove from the dictionary.

    Returns:
        None
    """
    if isinstance(dictionary, dict):
        for key in list(dictionary.keys()):
            if key == remove_key:
                del dictionary[key]
            else:
                recusive_remove(dictionary[key], remove_key)
