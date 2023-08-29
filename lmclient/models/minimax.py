from __future__ import annotations

import os
from typing import Any

import httpx

from lmclient.models.base import BaseChatModel
from lmclient.types import Messages, ModelResponse, Prompt
from lmclient.utils import ensure_messages


class MinimaxChat(BaseChatModel):
    def __init__(
        self,
        model_name: str = 'abab5.5-chat',
        group_id: str | None = None,
        api_key: str | None = None,
        timeout: int | None = 60,
    ):
        self.model_name = model_name
        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.timeout = timeout

    def chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        json_data = self._messages_to_request_json_data(messages)
        if 'temperature' in kwargs:
            kwargs['temperature'] = max(0.01, kwargs['temperature'])
        json_data.update(kwargs)
        response = httpx.post(
            f'https://api.minimax.chat/v1/text/chatcompletion?GroupId={self.group_id}',
            json=json_data,
            headers=headers,
            timeout=self.timeout,
        ).json()
        return response

    async def async_chat(self, prompt: Prompt, **kwargs) -> ModelResponse:
        messages = ensure_messages(prompt)

        async with httpx.AsyncClient() as client:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            json_data = self._messages_to_request_json_data(messages)
            if 'temperature' in kwargs:
                kwargs['temperature'] = max(0.01, kwargs['temperature'])
            json_data.update(kwargs)
            response = await client.post(
                f'https://api.minimax.chat/v1/text/chatcompletion?GroupId={self.group_id}',
                json=json_data,
                headers=headers,
                timeout=self.timeout,
            )
            response = response.json()
        return response

    def _messages_to_request_json_data(self, messages: Messages):
        data: dict[str, Any] = {
            'model': self.model_name,
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
        return f'{self.__class__.__name__}({self.model_name})'
