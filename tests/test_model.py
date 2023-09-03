import anyio
import pytest

from lmclient.models import AzureChat, BaseChatModel, MinimaxProChat, OpenAIChat, WenxinChat, ZhiPuChat
from lmclient.types import HttpChatModelOutput, Message, ModelParameters


@pytest.mark.parametrize('chat_model', (AzureChat(), MinimaxProChat(), OpenAIChat(), ZhiPuChat(), WenxinChat()))
def test_http_chat_model(chat_model: BaseChatModel[ModelParameters, HttpChatModelOutput]):
    test_messages = [Message(role='user', content='hello')]
    sync_output = chat_model.chat_completion(test_messages)
    async_output = anyio.run(chat_model.async_chat_completion, test_messages)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.reply, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.reply, str)


# def test_azure_model():
#     chat_model = AzureChat()

#     test_messages = [Message(role='user', content='hello')]
#     sync_output = chat_model.chat_completion(test_messages)
#     async_output = anyio.run(chat_model.async_chat_completion, test_messages)

#     assert isinstance(sync_output.response, dict)
#     assert isinstance(sync_output.reply, str)
#     assert isinstance(async_output.response, dict)
#     assert isinstance(async_output.reply, str)


# def test_openai_model():
#     chat_model = OpenAIChat('gpt-3.5-turbo')

#     test_messages = [Message(role='user', content='hello')]
#     sync_output = chat_model.chat_completion(test_messages)
#     async_output = anyio.run(chat_model.async_chat_completion, test_messages)

#     assert isinstance(sync_output.response, dict)
#     assert isinstance(sync_output.reply, str)
#     assert isinstance(async_output.response, dict)
#     assert isinstance(async_output.reply, str)


# def test_minimax_model():
#     chat_model = MinimaxProChat()

#     test_messages = [Message(role='user', content='hello')]
#     sync_output = chat_model.chat_completion(test_messages)
#     async_output = anyio.run(chat_model.async_chat_completion, test_messages)

#     assert isinstance(sync_output.response, dict)
#     assert isinstance(sync_output.reply, str)
#     assert isinstance(async_output.response, dict)
#     assert isinstance(async_output.reply, str)


# def test_zhipu_model():
#     chat_model = ZhiPuChat()

#     test_messages = [Message(role='user', content='hello')]
#     sync_output = chat_model.chat_completion(test_messages)
#     async_output = anyio.run(chat_model.async_chat_completion, test_messages)

#     assert isinstance(sync_output.response, dict)
#     assert isinstance(sync_output.reply, str)
#     assert isinstance(async_output.response, dict)
#     assert isinstance(async_output.reply, str)


# def test_wenxin_model():
#     WeninChat()
