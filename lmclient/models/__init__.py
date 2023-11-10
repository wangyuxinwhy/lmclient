from __future__ import annotations

from typing import Any, Type

from pydantic import BaseModel

from lmclient.models.azure import AzureChat
from lmclient.models.baichuan import BaichuanChat, BaichuanChatParameters
from lmclient.models.base import BaseChatModel
from lmclient.models.http import HttpChatModel, HttpChatModelInitKwargs
from lmclient.models.hunyuan import HunyuanChat, HunyuanChatParameters
from lmclient.models.minimax import MinimaxChat, MinimaxChatParameters
from lmclient.models.minimax_pro import MinimaxProChat, MinimaxProChatParameters
from lmclient.models.openai import OpenAIChat, OpenAIChatParameters
from lmclient.models.wenxin import WenxinChat, WenxinChatParameters
from lmclient.models.zhipu import ZhiPuCharacterChat, ZhiPuCharacterChatParameters, ZhiPuChat, ZhiPuChatParameters

ChatModels: list[tuple[Type[BaseChatModel], Type[BaseModel]]] = [
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

ChatModelRegistry: dict[str, tuple[Type[BaseChatModel], Type[BaseModel]]] = {
    model_cls.model_type: (model_cls, parameter_cls) for model_cls, parameter_cls in ChatModels
}


def load_from_model_id(model_id: str, **kwargs: Any) -> BaseChatModel:
    if '/' not in model_id:
        model_type = model_id
        return ChatModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = model_id.split('/')
    model_cls = ChatModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def list_chat_model_types() -> list[str]:
    return list(ChatModelRegistry.keys())


__all__ = [
    'BaseChatModel',
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
