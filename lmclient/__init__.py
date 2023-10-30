from lmclient.chat_engine import ChatEngine
from lmclient.completion_engine import CompletionEngine
from lmclient.models import (
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
    ZhiPuChat,
    ZhiPuChatParameters,
)
from lmclient.types import GeneralParameters, Message, RetryStrategy
from lmclient.utils import BaseSchema, PydanticVersion, function
from lmclient.version import __version__

LMClient = CompletionEngine

if PydanticVersion == 1:
    from pydantic import BaseModel

    BaseModel.model_copy = BaseModel.copy  # type: ignore
    BaseModel.model_dump = BaseModel.dict  # type: ignore
    BaseModel.model_dump_json = BaseModel.json  # type: ignore


__all__ = [
    'CompletionEngine',
    'ChatEngine',
    'AzureChat',
    'OpenAIChat',
    'OpenAIChatParameters',
    'MinimaxProChat',
    'MinimaxProChatParameters',
    'MinimaxChat',
    'MinimaxChatParameters',
    'ZhiPuChat',
    'ZhiPuChatParameters',
    'WenxinChat',
    'WenxinChatParameters',
    'HunyuanChat',
    'HunyuanChatParameters',
    'BaichuanChat',
    'BaichuanChatParameters',
    'BaseSchema',
    'function',
    'GeneralParameters',
    'Message',
    'RetryStrategy',
    '__version__',
]
