from lmclient.chat_engine import ChatEngine
from lmclient.client import LMClient
from lmclient.models import (
    AzureChat,
    MinimaxProChat,
    MinimaxProChatParameters,
    OpenAIChat,
    OpenAIChatParameters,
    ZhiPuChat,
    ZhiPuChatParameters,
)
from lmclient.utils import BaseSchema, PydanticVersion, function
from lmclient.version import __version__

if PydanticVersion == 1:
    from pydantic import BaseModel

    BaseModel.model_copy = BaseModel.copy  # type: ignore
    BaseModel.model_dump = BaseModel.dict  # type: ignore
    BaseModel.model_dump_json = BaseModel.json  # type: ignore


__all__ = [
    'LMClient',
    'ChatEngine',
    'AzureChat',
    'OpenAIChat',
    'OpenAIChatParameters',
    'MinimaxProChat',
    'MinimaxProChatParameters',
    'ZhiPuChat',
    'ZhiPuChatParameters',
    'BaseSchema',
    'function',
    '__version__',
]
