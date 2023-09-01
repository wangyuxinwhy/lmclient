import anyio
import pytest

from lmclient.models import AzureChat, MinimaxProChat, OpenAIChat, ZhiPuChat
from lmclient.models.openai import OpenAITextParser
from lmclient.types import Message

test_messages = [Message(role='user', content='hello')]

@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        test_messages
    ],
)
def test_azure_model(prompt):
    model = AzureChat(response_parser=OpenAITextParser())

    sync_output = model.chat(prompt)
    async_output = anyio.run(model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.message, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.message, str)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        test_messages
    ],
)
def test_openai_model(prompt):
    chat_model = OpenAIChat('gpt-3.5-turbo', response_parser=OpenAITextParser())

    sync_output = chat_model.chat(prompt)
    async_output = anyio.run(chat_model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.message, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.message, str)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        test_messages
    ],
)
def test_minimax_model(prompt):
    completion_model = MinimaxProChat('abab5.5-chat')

    sync_output = completion_model.chat(prompt)
    async_output = anyio.run(completion_model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.message, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.message, str)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        test_messages
    ],
)
def test_zhipu_model(prompt):
    completion_model = ZhiPuChat()

    sync_output = completion_model.chat(prompt)
    async_output = anyio.run(completion_model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.message, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.message, str)
