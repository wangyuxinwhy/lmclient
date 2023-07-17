from __future__ import annotations

import time

from lmclient.client import LMClient
from lmclient.types import ChatModel, Messages


class TestModel(ChatModel):
    def __init__(self) -> None:
        self.identifier = 'TestCompletion'
        self.timeout = None

    def chat(self, prompt: str | Messages, **kwargs) -> str:
        return f'Completed: {prompt}'

    async def async_chat(self, prompt: str | Messages, **kwargs) -> str:
        return f'Completed: {prompt}'


def test_sync_completion():
    completion_model = TestModel()
    client = LMClient(completion_model)

    messages = [
        {'role': 'system', 'content': 'your are lmclient demo assistant'},
        {'role': 'user', 'content': 'hello, who are you?'},
    ]
    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball', messages]
    completions = client.run(prompts)

    assert isinstance(completions[0], str)
    assert completions[0] == 'Completed: Hello, my name is'
    assert len(completions) == len(prompts)


def test_async_completion():
    completion_model = TestModel()
    client = LMClient(completion_model, async_capacity=2, max_requests_per_minute=5)
    LMClient.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    messages = [
        {'role': 'system', 'content': 'your are lmclient demo assistant'},
        {'role': 'user', 'content': 'hello, who are you?'},
    ]
    prompts = ['Hello, my name is', 'I am a student', messages] * 4
    completions = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert isinstance(completions[0], str)
    assert completions[0] == 'Completed: Hello, my name is'
    assert len(completions) == len(prompts)
    assert elapsed_time > 4


def test_async_completion_with_cache(tmp_path):
    completion_model = TestModel()
    client = LMClient(completion_model, async_capacity=2, max_requests_per_minute=5, cache_dir=str(tmp_path))
    LMClient.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball'] * 4
    completions = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert isinstance(completions[0], str)
    assert completions[3] == 'Completed: Hello, my name is'
    assert len(completions) == len(prompts)
    assert elapsed_time < 2
    assert len(list(client.cache)) == 3  # type: ignore
