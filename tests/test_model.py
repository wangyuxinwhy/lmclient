import anyio

from lmclient.models import AzureChat, MinimaxProChat, OpenAIChat, ZhiPuChat
from lmclient.types import Message

test_messages = [Message(role='user', content='hello')]


def test_azure_model():
    chat_model = AzureChat()

    test_messages = [Message(role='user', content='hello')]
    sync_output = chat_model.chat_completion(test_messages)
    async_output = anyio.run(chat_model.async_chat_completion, test_messages)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.reply, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.reply, str)


def test_openai_model():
    chat_model = OpenAIChat('gpt-3.5-turbo')

    test_messages = [Message(role='user', content='hello')]
    sync_output = chat_model.chat_completion(test_messages)
    async_output = anyio.run(chat_model.async_chat_completion, test_messages)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.reply, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.reply, str)


def test_minimax_model():
    chat_model = MinimaxProChat()

    test_messages = [Message(role='user', content='hello')]
    sync_output = chat_model.chat_completion(test_messages)
    async_output = anyio.run(chat_model.async_chat_completion, test_messages)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.reply, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.reply, str)


def test_zhipu_model():
    chat_model = ZhiPuChat()

    test_messages = [Message(role='user', content='hello')]
    sync_output = chat_model.chat_completion(test_messages)
    async_output = anyio.run(chat_model.async_chat_completion, test_messages)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.reply, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.reply, str)
