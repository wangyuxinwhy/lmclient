from __future__ import annotations

import os
from typing import Any

import httpx
import requests

from lmclient.types import ChatModel, Message, Messages, ModelResponse


class MinimaxChat(ChatModel):
    def __init__(
        self,
        model_name: str,
        group_id: str | None = None,
        api_key: str | None = None,
    ):
        self.model_name = model_name

        self.group_id = group_id or os.environ['MINIMAX_GROUP_ID']
        self.api_key = api_key or os.environ['MINIMAX_API_KEY']
        self.timeout = None

    def chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }
        json_data = self._messages_to_request_json_data(prompt)
        if 'temperature' in kwargs:
            kwargs['temperature'] = max(0.01, kwargs['temperature'])
        json_data.update(kwargs)
        response = requests.post(
            f'https://api.minimax.chat/v1/text/chatcompletion?GroupId={self.group_id}',
            json=json_data,
            headers=headers,
            timeout=self.timeout,
        ).json()
        return response

    async def async_chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        if isinstance(prompt, str):
            prompt = [Message(role='user', content=prompt)]

        async with httpx.AsyncClient() as client:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            }
            json_data = self._messages_to_request_json_data(prompt)
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

    @staticmethod
    def default_postprocess_function(response: ModelResponse) -> ModelResponse:
        try:
            response['content'] = response['choices'][0]['message']['text']
        except (KeyError, IndexError):
            response['content'] = 'Error Response'
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
