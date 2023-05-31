import anyio

from lmclient.completions import AzureCompletion, OpenAICompletion


def test_azure_completion():
    completion_model = AzureCompletion()

    sync_output = completion_model.complete('Hello, my name is')
    async_output = anyio.run(completion_model.async_complete, 'Hello, my name is')

    assert isinstance(sync_output, str)
    assert isinstance(async_output, str)


def test_openai_completion():
    completion_model = OpenAICompletion('gpt-3.5-turbo')

    sync_output = completion_model.complete('Hello, my name is')
    async_output = anyio.run(completion_model.async_complete, 'Hello, my name is')

    assert isinstance(sync_output, str)
    assert isinstance(async_output, str)
