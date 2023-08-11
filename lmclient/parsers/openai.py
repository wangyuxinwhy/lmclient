from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from lmclient.exceptions import ParserError
from lmclient.parsers.base import ModelResponseParser
from lmclient.types import ModelResponse

T = TypeVar('T', bound='OpenAISchema')


class OpenAIParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> str | dict[str, str]:
        try:
            if self.is_function_call(response):
                fucntion_call_output: dict[str, str] = response['choices'][0]['message']['function_call']
                return fucntion_call_output
            else:
                content_output: str = response['choices'][0]['message']['content']
                return content_output
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e

    @staticmethod
    def is_function_call(reponse: ModelResponse) -> bool:
        message = reponse['choices'][0]['message']
        return bool(message.get('function_call'))


class OpenAIFunctionCallParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> dict[str, str]:
        try:
            output: dict[str, str] = response['choices'][0]['message']['function_call']
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output


class OpenAIContentParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> str:
        try:
            output: str = response['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output


# COPY FROM openai_function_call
def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


class OpenAISchema(BaseModel):
    @classmethod
    def openai_schema(cls):
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.model_json_schema()
        parameters = {k: v for k, v in schema.items() if k not in ('title', 'description')}
        parameters['required'] = sorted(parameters['properties'])
        _remove_a_key(parameters, 'title')

        if 'description' not in schema:
            schema['description'] = f'Correctly extracted `{cls.__name__}` with all the required parameters with correct types'

        return {
            'name': schema['title'],
            'description': schema['description'],
            'parameters': parameters,
        }

    @classmethod
    def from_response(cls, response: ModelResponse):
        """Execute the function from the response of an openai chat completion

        Parameters:
            completion (openai.ChatCompletion): The response from an openai chat completion
            throw_error (bool): Whether to throw an error if the function call is not detected

        Returns:
            cls (OpenAISchema): An instance of the class
        """
        message = response['choices'][0]['message']

        if 'function_call' not in message:
            raise ParserError('No function call detected')

        function_call = message['function_call']
        arguments = json.loads(function_call['arguments'])
        return cls(**arguments)
