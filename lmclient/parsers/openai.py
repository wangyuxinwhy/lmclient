from __future__ import annotations

from lmclient.exceptions import ParserError
from lmclient.types import ModelResponse


class OpenAIParser:
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


class OpenAIFunctionCallParser:
    def __call__(self, response: ModelResponse) -> dict[str, str]:
        try:
            output: dict[str, str] = response['choices'][0]['message']['function_call']
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output


class OpenAIContentParser:
    def __call__(self, response: ModelResponse) -> str:
        try:
            output: str = response['choices'][0]['message']['content']
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output
