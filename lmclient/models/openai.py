from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Type, TypeVar

from lmclient.models.base import HttpChatModel, RetryStrategy
from lmclient.openai_schema import OpenAISchema
from lmclient.parser import ModelResponseParser, ParserError
from lmclient.types import Messages, ModelResponse

T = TypeVar('T')
T_O = TypeVar('T_O', bound=OpenAISchema)


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


class OpenAIChat(HttpChatModel[T]):
    def __init__(
        self,
        model: str = 'gpt-3.5-turbo',
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int | None = 60,
        response_parser: ModelResponseParser[T] | None = None,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
    ):
        response_parser = response_parser or OpenAIContentParser()
        super().__init__(timeout=timeout, response_parser=response_parser, retry=retry, use_cache=use_cache)
        self.model = model
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
        self.api_key = api_key or os.environ['OPENAI_API_KEY']
        self.timeout = timeout

    def get_post_parameters(self, messages: Messages, **kwargs) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        params = {
            'model': self.model,
            'messages': messages,
            **kwargs,
        }
        return {
            'url': f'{self.api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'


class OpenAIExtract(HttpChatModel[T_O]):
    def __init__(
        self,
        schema: Type[T_O],
        model: str = 'gpt-3.5-turbo',
        system_prompt: str = 'Extract structured data from a given text',
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: int | None = 60,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(timeout=timeout, response_parser=schema.from_response, retry=retry, use_cache=use_cache)
        self.schema = schema
        self.model = model
        self.system_prompt = system_prompt
        self.api_base = api_base or os.getenv('OPENAI_API_BASE') or 'https://api.openai.com/v1'
        self.api_key = api_key or os.environ['OPENAI_API_KEY']
        self.timeout = timeout

    def get_post_parameters(self, messages: Messages, **kwargs) -> dict[str, Any]:
        messages = [{'role': 'system', 'content': self.system_prompt}] + list(messages)   # type: ignore
        headers = {
            'Authorization': f'Bearer {self.api_key}',
        }
        params = {
            'model': self.model,
            'messages': messages,
            'functions': [self.schema.openai_schema()],
            'function_call': {'name': self.schema.openai_schema()['name']},
            **kwargs,
        }
        return {
            'url': f'{self.api_base}/chat/completions',
            'headers': headers,
            'json': params,
        }

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}(model={self.model}, system_prompt={self.system_prompt})'
