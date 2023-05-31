import time

from lmclient.client import LMClient


class TestCompletion:
    def complete(self, prompt: str, **kwargs) -> str:
        return f'Completed: {prompt}'

    async def async_complete(self, prompt: str, **kwargs) -> str:
        return f'Completed: {prompt}'


def test_sync_completion():
    completion_model = TestCompletion()
    client = LMClient(completion_model)

    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball']
    completions = client.run(prompts)

    assert isinstance(completions[0], str)
    assert completions[0] == 'Completed: Hello, my name is'
    assert len(completions) == len(prompts)


def test_async_completion():
    completion_model = TestCompletion()
    client = LMClient(completion_model, async_capacity=2, max_requests_per_minute=5)
    LMClient.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball'] * 4
    completions = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert isinstance(completions[0], str)
    assert completions[0] == 'Completed: Hello, my name is'
    assert len(completions) == len(prompts)
    assert elapsed_time > 4
