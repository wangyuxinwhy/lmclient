from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from lmclient.completion_engine import CompletionEngine
from lmclient.models.base import BaseChatModel
from lmclient.types import ChatModelOutput, Messages, ModelParameters, TextMessage


class TestModelParameters(ModelParameters):
    prefix: str = 'Completed:'


class TestModel(BaseChatModel[TestModelParameters, ChatModelOutput]):
    model_type = 'test'

    def __init__(self, default_parameters: TestModelParameters | None = None, use_cache: Path | str | bool = False) -> None:
        default_parameters = default_parameters or TestModelParameters()
        super().__init__(default_parameters, use_cache)

    def _chat_completion(self, messages: Messages, parameters: TestModelParameters) -> ChatModelOutput:
        content = f'Completed: {messages[-1]["content"]}'
        return ChatModelOutput(
            model_id='test', parameters=parameters, messages=[TextMessage(role='assistant', content=content)], reply=content
        )

    async def _async_chat_completion(self, messages: Messages, parameters: TestModelParameters) -> ChatModelOutput:
        content = f'Completed: {messages[-1]["content"]}'
        return ChatModelOutput(
            model_id='test', parameters=parameters, messages=[TextMessage(role='assistant', content=content)], reply=content
        )

    @property
    def name(self) -> str:
        return 'TestModel'

    @classmethod
    def from_name(cls, name: str, **kwargs: Any):
        return cls()


def test_sync_completion():
    completion_model = TestModel()
    client = CompletionEngine(completion_model)
    prompts = [
        'Hello, my name is',
        TextMessage(role='user', content='hello, who are you?'),
    ]
    results = list(client.run(prompts))

    assert isinstance(results[0].reply, str)
    assert results[0].reply == 'Completed: Hello, my name is'
    assert results[1].reply == 'Completed: hello, who are you?'
    assert len(results) == len(prompts)


def test_async_completion():
    completion_model = TestModel()
    client = CompletionEngine(completion_model, async_capacity=2, max_requests_per_minute=5)
    CompletionEngine.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    messages: list[TextMessage] = [{'role': 'user', 'content': 'hello, who are you?'}]
    prompts = ['Hello, my name is', 'I am a student', messages] * 4
    results = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert results[0].reply == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time > 4


def test_async_completion_with_cache(tmp_path: Path):
    completion_model = TestModel(use_cache=tmp_path)
    client = CompletionEngine(completion_model, async_capacity=2, max_requests_per_minute=5)
    CompletionEngine.NUM_SECONDS_PER_MINUTE = 2

    start_time = time.perf_counter()
    prompts = ['Hello, my name is', 'I am a student', 'I like to play basketball'] * 4
    results = client.async_run(prompts)
    elapsed_time = time.perf_counter() - start_time

    assert isinstance(results[0].reply, str)
    assert results[3].reply == 'Completed: Hello, my name is'
    assert len(results) == len(prompts)
    assert elapsed_time < 2
