import json

try:
    from pydantic.v1 import BaseModel
    from pydantic.v1 import Field as Field
except ImportError:
    from pydantic import BaseModel
    from pydantic import Field as Field

from lmclient.parser import ParserError
from lmclient.types import ModelResponse


def _remove_a_key(d, remove_key) -> None:
    """Remove a key from a dictionary recursively"""
    if isinstance(d, dict):
        for key in list(d.keys()):
            if key == remove_key:
                del d[key]
            else:
                _remove_a_key(d[key], remove_key)


class OpenAISchema(BaseModel):  # type: ignore
    @classmethod
    def openai_schema(cls):
        """
        Return the schema in the format of OpenAI's schema as jsonschema

        Note:
            Its important to add a docstring to describe how to best use this class, it will be included in the description attribute and be part of the prompt.

        Returns:
            model_json_schema (dict): A dictionary in the format of OpenAI's schema as jsonschema
        """
        schema = cls.schema()
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
