import anyio
import pytest

from lmclient.models import AzureChat, OpenAIChat


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_azure_completion(prompt):
    completion_model = AzureChat()

    sync_output = completion_model.complete(prompt)
    async_output = anyio.run(completion_model.async_complete, prompt)

    assert isinstance(sync_output, str)
    assert isinstance(async_output, str)


@pytest.mark.parametrize(
    'prompt',
    [
        'Hello, my name is',
        [{'role': 'system', 'content': 'your are lmclient demo assistant'}, {'role': 'user', 'content': 'hello, who are you?'}],
    ],
)
def test_openai_completion(prompt):
    completion_model = OpenAIChat('gpt-3.5-turbo')

    sync_output = completion_model.complete(prompt)
    async_output = anyio.run(completion_model.async_complete, prompt)

    assert isinstance(sync_output, str)
    assert isinstance(async_output, str)
