from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from lmclient.chat_completion.base import ChatCompletionModel
from lmclient.chat_completion.http import HttpChatModel, HttpChatModelInitKwargs
from lmclient.chat_completion.model_output import ChatCompletionModelOutput, ChatCompletionModelStreamOutput
from lmclient.chat_completion.model_parameters import ModelParameters
from lmclient.chat_completion.models import (
    AzureChat,
    BaichuanChat,
    BaichuanChatParameters,
    HunyuanChat,
    HunyuanChatParameters,
    MinimaxChat,
    MinimaxChatParameters,
    MinimaxProChat,
    MinimaxProChatParameters,
    OpenAIChat,
    OpenAIChatParameters,
    WenxinChat,
    WenxinChatParameters,
    ZhiPuCharacterChat,
    ZhiPuCharacterChatParameters,
    ZhiPuChat,
    ZhiPuChatParameters,
)

ChatModels: list[tuple[Type[ChatCompletionModel], Type[BaseModel]]] = [
    (AzureChat, OpenAIChatParameters),
    (OpenAIChat, OpenAIChatParameters),
    (MinimaxProChat, MinimaxProChatParameters),
    (MinimaxChat, MinimaxProChatParameters),
    (ZhiPuChat, ZhiPuChatParameters),
    (ZhiPuCharacterChat, ZhiPuCharacterChatParameters),
    (WenxinChat, WenxinChatParameters),
    (HunyuanChat, HunyuanChatParameters),
    (BaichuanChat, BaichuanChatParameters),
]

ChatModelRegistry: dict[str, tuple[Type[ChatCompletionModel], Type[BaseModel]]] = {
    model_cls.model_type: (model_cls, parameter_cls) for model_cls, parameter_cls in ChatModels
}


def load_from_model_id(model_id: str, **kwargs: Any) -> ChatCompletionModel:
    if '/' not in model_id:
        model_type = model_id
        return ChatModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = model_id.split('/')
    model_cls = ChatModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def list_chat_model_types() -> list[str]:
    return list(ChatModelRegistry.keys())


__all__ = [
    'ChatCompletionModel',
    'ChatCompletionModelOutput',
    'ChatCompletionModelStreamOutput',
    'ModelParameters',
    'HttpChatModel',
    'HttpChatModelInitKwargs',
    'AzureChat',
    'MinimaxProChat',
    'MinimaxProChatParameters',
    'MinimaxChat',
    'MinimaxChatParameters',
    'OpenAIChat',
    'OpenAIChatParameters',
    'ZhiPuChat',
    'ZhiPuChatParameters',
    'ZhiPuCharacterChat',
    'ZhiPuCharacterChatParameters',
    'WenxinChat',
    'WenxinChatParameters',
    'HunyuanChat',
    'HunyuanChatParameters',
    'BaichuanChat',
    'BaichuanChatParameters',
]
