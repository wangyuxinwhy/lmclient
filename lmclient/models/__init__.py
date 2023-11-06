from __future__ import annotations

from typing import Any, Type

from lmclient.models.azure import AzureChat
from lmclient.models.baichuan import BaichuanChat, BaichuanChatParameters
from lmclient.models.base import BaseChatModel, ModelParameters, OverrideParameters
from lmclient.models.http import HttpChatModel, HttpChatModelKwargs
from lmclient.models.hunyuan import HunyuanChat, HunyuanChatParameters
from lmclient.models.minimax import MinimaxChat, MinimaxChatParameters
from lmclient.models.minimax_pro import MinimaxProChat, MinimaxProChatParameters
from lmclient.models.openai import OpenAIChat, OpenAIChatParameters
from lmclient.models.wenxin import WenxinChat, WenxinChatParameters
from lmclient.models.zhipu import ZhiPuCharacterChat, ZhiPuCharacterChatParameters, ZhiPuChat, ZhiPuChatParameters

ModelRegistry: dict[str, tuple[Type[BaseChatModel], Type[ModelParameters]]] = {
    AzureChat.model_type: (AzureChat, OpenAIChatParameters),
    OpenAIChat.model_type: (OpenAIChat, OpenAIChatParameters),
    MinimaxProChat.model_type: (MinimaxProChat, MinimaxProChatParameters),
    MinimaxChat.model_type: (MinimaxChat, MinimaxProChatParameters),
    ZhiPuChat.model_type: (ZhiPuChat, ZhiPuChatParameters),
    ZhiPuCharacterChat.model_type: (ZhiPuCharacterChat, ZhiPuCharacterChatParameters),
    WenxinChat.model_type: (WenxinChat, WenxinChatParameters),
    HunyuanChat.model_type: (HunyuanChat, HunyuanChatParameters),
    BaichuanChat.model_type: (BaichuanChat, BaichuanChatParameters),
}


def load_from_model_id(model_id: str, **kwargs: Any):
    if '/' not in model_id:
        model_type = model_id
        return ModelRegistry[model_type][0](**kwargs)  # type: ignore
    model_type, name = model_id.split('/')
    model_cls = ModelRegistry[model_type][0]
    return model_cls.from_name(name, **kwargs)


def list_chat_model_types():
    return list(ModelRegistry.keys())


__all__ = [
    'BaseChatModel',
    'HttpChatModel',
    'HttpChatModelKwargs',
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
    'OverrideParameters',
]
