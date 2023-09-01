# MIT License
#
# Copyright (c) 2023 Jason Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from functools import wraps
from typing import Any, Callable

from docstring_parser import parse
from pydantic import validate_arguments

from lmclient.exceptions import MessageError
from lmclient.types import Message


def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


class lm_function:
    def __init__(self, func: Callable) -> None:
        self.func = func
        self.name = self.func.__name__
        self.validate_func = validate_arguments(func)
        self.docstring = parse(self.func.__doc__ or '')

        parameters = self.validate_func.model.model_json_schema()
        parameters["properties"] = {
            k: v
            for k, v in parameters["properties"].items()
            if k not in ("v__duplicate_kwargs", "args", "kwargs")
        }
        for param in self.docstring.params:
            if (name := param.arg_name) in parameters["properties"] and (
                description := param.description
            ):
                parameters["properties"][name]["description"] = description
        parameters["required"] = sorted(
            k for k, v in parameters["properties"].items() if "default" not in v
        )
        _remove_a_key(parameters, "additionalProperties")
        _remove_a_key(parameters, "title")
        self.openai_schema = {
            "name": self.name,
            "description": self.docstring.short_description,
            "parameters": parameters,
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
        arguments = json.loads(function_call["arguments"], strict=False)
        return self.validate_func(**arguments)
