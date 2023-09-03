from __future__ import annotations

import os
from datetime import datetime, timedelta
from email.errors import MessageError
from pathlib import Path
from typing import Any, Literal, Optional

import httpx
from typing_extensions import Self, TypedDict

from lmclient.exceptions import ResponseError
from lmclient.models.http import HttpChatModel
from lmclient.types import GeneralParameters, Message, Messages, ModelParameters, ModelResponse, RetryStrategy

WENXIN_ACCESS_TOKEN_URL = 'https://aip.baidubce.com/oauth/2.0/token'
WENXIN_BASE_URL = 'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/'


class WenxinMessageDict(TypedDict):
    role: Literal['user', 'assistant']
    content: str


class WenxinChatParameters(ModelParameters):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    penalty_score: Optional[float] = None

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        return cls(
            temperature=general_parameters.temperature,
            top_p=general_parameters.top_p,
        )


class WenxinChat(HttpChatModel[WenxinChatParameters]):
    model_type = 'wenxin'
    model_name_entrypoint_map: dict[str, str] = {
        'llama_2_7b': 'llama_2_7b',
        'llama_2_13b': 'llama_2_13b',
        'llama_2_70b': 'llama_2_70b',
        'ERNIE-Bot': 'completions',
        'ERNIE-Bot-turbo': 'eb-instant',
    }
    access_token_refresh_days: int = 20

    def __init__(
        self,
        model: str = 'ERNIE-Bot',
        api_key: str | None = None,
        secret_key: str | None = None,
        parameters: WenxinChatParameters = WenxinChatParameters(),
        timeout: int | None = None,
        retry: bool | RetryStrategy = False,
        use_cache: Path | str | bool = False,
    ):
        super().__init__(parameters, timeout, retry, use_cache)
        self.model = self.normalize_model(model)
        self._api_key = api_key or os.getenv('WENXIN_API_KEY')
        self._secret_key = secret_key or os.getenv('WENXIN_SECRET_KEY')
        self._access_token = self.get_access_token()
        self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    @property
    def name(self) -> str:
        return self.model

    @property
    def api_url(self) -> str:
        return WENXIN_BASE_URL + self.model_name_entrypoint_map[self.model]

    @staticmethod
    def normalize_model(model: str):
        _map = {
            'llama-2-7b-chat': 'llama_2_7b',
            'llama-2-13b-chat': 'llama_2_13b',
            'llama-2-70b-chat': 'llama_2_70b',
        }
        return _map.get(model, model)

    def get_access_token(self, base_url: str = WENXIN_ACCESS_TOKEN_URL) -> str:
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        params = {'grant_type': 'client_credentials', 'client_id': self._api_key, 'client_secret': self._secret_key}
        response = httpx.post(base_url, headers=headers, params=params)
        response.raise_for_status()
        response_dict = response.json()
        if 'error' in response_dict:
            raise ResponseError(response_dict['error_description'])
        return response_dict['access_token']

    def get_request_parameters(self, messages: Messages, parameters: WenxinChatParameters) -> dict[str, Any]:
        self.maybe_refresh_access_token()

        message_dicts: list[WenxinMessageDict] = []
        for message in messages:
            role = message.role
            if role != 'assistant' and role != 'user':
                raise MessageError(f'Invalid message role: {role}, only "user" and "assistant" are allowed')
            if not isinstance(content := message.content, str):
                raise MessageError(f'Invalid message content: {content}, only string is allowed')
            message_dicts.append(WenxinMessageDict(content=content, role=role))
        parameters_dict = parameters.model_dump(exclude_none=True)
        json_data = {'messages': message_dicts, **parameters_dict}

        return {
            'url': self.api_url,
            'json': json_data,
            'params': {'access_token': self._access_token},
            'headers': {'Content-Type': 'application/json'},
        }

    def parse_model_reponse(self, response: ModelResponse) -> Messages:
        if response.get('error_msg'):
            raise ResponseError(response['error_msg'])
        return [Message(role='assistant', content=response['result'])]

    def maybe_refresh_access_token(self):
        if self._access_token_expires_at < datetime.now():
            self._access_token = self.get_access_token()
            self._access_token_expires_at = datetime.now() + timedelta(days=self.access_token_refresh_days)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(model=name, **kwargs)
