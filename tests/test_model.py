import anyio
import pytest

from lmclient.models import AzureChat, MinimaxChat, OpenAIChat, ZhiPuChat
from lmclient.models.openai import OpenAIContentParser


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_azure_model(prompt):
    model = AzureChat(response_parser=OpenAIContentParser())

    sync_output = model.chat(prompt)
    async_output = anyio.run(model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.parsed_result, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.parsed_result, str)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_openai_model(prompt):
    chat_model = OpenAIChat('gpt-3.5-turbo', response_parser=OpenAIContentParser())

    sync_output = chat_model.chat(prompt)
    async_output = anyio.run(chat_model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.parsed_result, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.parsed_result, str)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_minimax_model(prompt):
    completion_model = MinimaxChat('abab5.5-chat')

    sync_output = completion_model.chat(prompt)
    async_output = anyio.run(completion_model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.parsed_result, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.parsed_result, str)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_zhipu_model(prompt):
    completion_model = ZhiPuChat()

    sync_output = completion_model.chat(prompt)
    async_output = anyio.run(completion_model.async_chat, prompt)

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.parsed_result, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.parsed_result, str)
