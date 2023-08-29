import anyio
import pytest

from lmclient.models import AzureChat, MinimaxChat, OpenAIChat, ZhiPuChat


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_azure_model(prompt):
    model = AzureChat()

    sync_output = model.chat(prompt)
    async_output = anyio.run(model.async_chat, prompt)

    assert isinstance(sync_output, dict)
    assert isinstance(async_output, dict)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_openai_model(prompt):
    completion_model = OpenAIChat('gpt-3.5-turbo')

    sync_output = completion_model.chat(prompt)
    async_output = anyio.run(completion_model.async_chat, prompt)

    assert isinstance(sync_output, dict)
    assert isinstance(async_output, dict)


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

    assert isinstance(sync_output, dict)
    assert isinstance(async_output, dict)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_zhipu_model(prompt):
    completion_model = ZhiPuChat()

    sync_output = completion_model.chat(prompt)
    async_output = anyio.run(completion_model.async_chat, prompt)

    assert isinstance(sync_output, dict)
    assert isinstance(async_output, dict)
