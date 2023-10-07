import anyio
import pytest

from lmclient.models import AzureChat, BaseChatModel, HunyuanChat, MinimaxProChat, OpenAIChat, WenxinChat, ZhiPuChat
from lmclient.types import HttpChatModelOutput, Message, ModelParameters


@pytest.mark.parametrize('chat_model', (AzureChat(), MinimaxProChat(), OpenAIChat(), ZhiPuChat(), WenxinChat(), HunyuanChat()))
def test_http_chat_model(chat_model: BaseChatModel[ModelParameters, HttpChatModelOutput]):
    test_messages = [Message(role='user', content='hello')]
    sync_output = chat_model.chat_completion(test_messages)
    async_output = anyio.run(chat_model.async_chat_completion, test_messages)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.reply, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.reply, str)
