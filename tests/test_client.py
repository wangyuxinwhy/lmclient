from __future__ import annotations

import time

from lmclient.client import LMClient
from lmclient.models.base import BaseChatModel
from lmclient.types import Messages, ModelResponse


class TestModel(BaseChatModel):
    def chat_completion(self, messages: Messages, **kwargs) -> ModelResponse:
        return {
            'content': f'Completed: {messages[-1].content}',
        }

    async def async_chat_completion(self, messages: Messages, **kwargs) -> ModelResponse:
        return {
            'content': f'Completed: {messages[-1].content}',
        }

    def default_postprocess_function(self, response: ModelResponse) -> str:
        return response['content']

    @property
    def identifier(self) -> str:
        return 'TestModel'


def model_parser(response):
    return response['content']


def test_sync_completion():
    completion_model = TestModel(response_parser=model_parser, use_cache=False)
    client = LMClient(completion_model)
    prompts = [
        'Hello, my name is',
        [
            {'role': 'system', 'content': 'your are lmclient demo assistant'},
            {'role': 'user', 'content': 'hello, who are you?'},
        ],
    ]
    results = client.run(prompts)

    assert isinstance(results[0].message, str)
    assert results[0].message == 'Completed: Hello, my name is'
    assert results[1].message == 'Completed: hello, who are you?'
    assert len(results) == len(prompts)


def test_async_completion():
    completion_model = TestModel(response_parser=model_parser, use_cache=False)
    client = LMClient(completion_model, async_capacity=2, max_requests_per_minute=5)
    LMClient.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    messages = [
        {'role': 'system', 'content': 'your are lmclient demo assistant'},
        {'role': 'user', 'content': 'hello, who are you?'},
    ]
    prompts = ['Hello, my name is', 'I am a student', messages] * 4
    results = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert results[0].response['content'] == 'Completed: Hello, my name is'
    assert results[0].message == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time > 4


def test_async_completion_with_cache(tmp_path):
    completion_model = TestModel(use_cache=tmp_path, response_parser=model_parser)
    client = LMClient(completion_model, async_capacity=2, max_requests_per_minute=5)
    LMClient.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball'] * 4
    results = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert isinstance(results[0].response['content'], str)
    assert results[3].response['content'] == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time < 2
    assert len(list(completion_model._cache)) == 3  # type: ignore
