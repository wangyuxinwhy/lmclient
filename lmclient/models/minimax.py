from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

from lmclient.models.base import HttpChatModel, RetryStrategy
from lmclient.parser import ModelResponseParser, ParserError
from lmclient.types import Messages, ModelResponse

T = TypeVar('T')


class MinimaxTextParser(ModelResponseParser):
    def __call__(self, response: ModelResponse) -> str:
        try:
            output = response['choices'][0]['text']
        except (KeyError, IndexError) as e:
            raise ParserError('Parse response failed') from e
        return output


class MinimaxChat(HttpChatModel[T]):
    def __init__(
        self,
        model: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        timeout: int | None = 60,
        response_parser: ModelResponseParser[T] | None = None,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
    ):
        response_parser = response_parser or MinimaxTextParser()
        super().__init__(timeout=timeout, response_parser=response_parser, retry=retry, use_cache=use_cache)
        self.model = model
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']

    def get_post_parameters(self, messages: Messages, **kwargs) -> dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        json_data = self._messages_to_request_json_data(messages)
        if 'temperature' in kwargs:
            kwargs['temperature'] = max(0.01, kwargs['temperature'])
        json_data.update(kwargs)
        return {
            'url': f'https://api.minimax.chat/v1/text/chatcompletion?GroupId={self.group_id}',
            'json': json_data,
            'headers': headers,
        }

    def _messages_to_request_json_data(self, messages: Messages):
        data: dict[str, Any] = {
            'model': self.model,
            'role_meta': {'user_name': '用户', 'bot_name': 'MM智能助理'},
        }

        if messages[0]['role'] == 'system':
            data['prompt'] = messages[0]['content']
            messages = messages[1:]
        else:
            data['prompt'] = '你是MM智能助理'
        minimax_messages = []
        for message in messages:
            if message['role'] == 'user':
                role = 'USER'
            elif message['role'] == 'assistant':
                role = 'BOT'
            else:
                raise ValueError(f'Invalid role: {message["role"]}')

            minimax_messages.append(
                {
                    'sender_type': role,
                    'text': message['content'],
                }
            )
        data['messages'] = minimax_messages
        return data

    @property
    def identifier(self) -> str:
        return f'{self.__class__.__name__}({self.model})'
