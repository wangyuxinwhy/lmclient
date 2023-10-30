from typing import Any

from lmclient.models.azure import AzureChat
from lmclient.models.baichuan import BaichuanChat, BaichuanChatParameters
from lmclient.models.base import BaseChatModel
from lmclient.models.hunyuan import HunyuanChat, HunyuanChatParameters
from lmclient.models.minimax import MinimaxChat, MinimaxChatParameters
from lmclient.models.minimax_pro import MinimaxProChat, MinimaxProChatParameters
from lmclient.models.openai import OpenAIChat, OpenAIChatParameters
from lmclient.models.wenxin import WenxinChat, WenxinChatParameters
from lmclient.models.zhipu import ZhiPuChat, ZhiPuChatParameters

ModelRegistry = {
    AzureChat.model_type: AzureChat,
    OpenAIChat.model_type: OpenAIChat,
    MinimaxProChat.model_type: MinimaxProChat,
    MinimaxChat.model_type: MinimaxChat,
    ZhiPuChat.model_type: ZhiPuChat,
    WenxinChat.model_type: WenxinChat,
    HunyuanChat.model_type: HunyuanChat,
    BaichuanChat.model_type: BaichuanChat,
}


def load_from_model_id(model_id: str, **kwargs: Any):
    if '/' not in model_id:
        model_type = model_id
        return ModelRegistry[model_type](**kwargs)  # type: ignore
    model_type, name = model_id.split('/')
    model_cls = ModelRegistry[model_type]
    return model_cls.from_name(name, **kwargs)


def list_chat_model_types():
    return list(ModelRegistry.keys())


__all__ = [
    'AzureChat',
    'BaseChatModel',
    'MinimaxProChat',
    'MinimaxProChatParameters',
    'MinimaxChat',
    'MinimaxChatParameters',
    'OpenAIChat',
    'OpenAIChatParameters',
    'ZhiPuChat',
    'ZhiPuChatParameters',
    'WenxinChat',
    'WenxinChatParameters',
    'HunyuanChat',
    'HunyuanChatParameters',
    'BaichuanChat',
    'BaichuanChatParameters',
]
