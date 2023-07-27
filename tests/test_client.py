from __future__ import annotations

import time

from lmclient.client import LMClient
from lmclient.types import ChatModel, Messages, ModelResponse


class TestModel(ChatModel):
    def __init__(self) -> None:
        self.identifier = 'TestCompletion'
        self.timeout = None

    def chat(self, prompt: str | Messages, **kwargs) -> ModelResponse:
        return {
            'content': f'Completed: {prompt}',
        }

    async def async_chat(self, prompt: str | Messages, **kwargs) -> ModelResponse:
        return {
            'content': f'Completed: {prompt}',
        }

    def default_postprocess_function(self, response: ModelResponse) -> ModelResponse:
        return response


def test_sync_completion():
    completion_model = TestModel()
    client = LMClient(completion_model)

    messages = [
        {'role': 'system', 'content': 'your are lmclient demo assistant'},
        {'role': 'user', 'content': 'hello, who are you?'},
    ]
    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball', messages]
    results = client.run(prompts)

    assert isinstance(results[0].response['content'], str)
    assert results[0].response['content'] == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)


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
    results = client.async_run(prompts)
    content = results[0].response['content']
    elapsed_time = time.perf_counter() - start_time

    assert isinstance(content, str)
    assert content == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time > 4


def test_async_completion_with_cache(tmp_path):
    completion_model = TestModel()
    client = LMClient(completion_model, async_capacity=2, max_requests_per_minute=5, cache_dir=str(tmp_path))
    LMClient.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball'] * 4
    results = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert isinstance(results[0].response['content'], str)
    assert results[3].response['content'] == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time < 2
    assert len(list(client.cache)) == 3  # type: ignore
