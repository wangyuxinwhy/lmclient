from __future__ import annotations

import asyncio
from typing import Literal, Sequence, Type

import pytest
from pydantic import BaseModel

from lmclient.chat_completion import (
    ChatCompletionModel,
    ChatCompletionModelStreamOutput,
    ChatModelRegistry,
    ChatModels,
    load_from_model_id,
)
from lmclient.chat_completion.http import HttpChatModel
from lmclient.chat_completion.message import Prompt

param_type = Literal['model', 'model_cls', 'parameter', 'parameter_cls']


def get_pytest_params(id_prefix: str, types: Sequence[param_type] | param_type, exclude: Sequence[str] | None = None) -> list:
    exclude = exclude or []
    if isinstance(types, str):
        types = [types]

    pytest_params: list = []
    for model_name, (model_cls, paramter_cls) in ChatModelRegistry.items():
        if model_name in exclude:
            continue
        values = []
        for t in types:
            if t == 'model':
                values.append(model_cls(parameters=paramter_cls()))
            elif t == 'model_cls':
                values.append(model_cls)
            elif t == 'parameter':
                values.append(paramter_cls())
            elif t == 'parameter_cls':
                values.append(paramter_cls)
            else:
                raise ValueError(f'Unknown type {t}')
        pytest_params.append(pytest.param(*values, id=f'{id_prefix}_{model_name}'))
    return pytest_params


def test_model_type_is_unique() -> None:
    assert len(ChatModels) == len(ChatModelRegistry)


def test_load_from_model_id() -> None:
    model = load_from_model_id('openai/gpt-3.5-turbo')
    assert model.model_type == 'openai'
    assert model.name == 'gpt-3.5-turbo'


@pytest.mark.parametrize('chat_model', get_pytest_params('test_chat_completion', types='model'))
@pytest.mark.parametrize(
    'parameters',
    [
        {},
        {'temperature': 0.5, 'top_p': 0.85, 'max_tokens': 20},
        {'temperature': 0},
        {'top_p': 0},
    ],
)
def test_http_chat_model(chat_model: HttpChatModel, parameters: dict) -> None:
    chat_model.timeout = 20
    prompt = '这是测试，只回复你好'
    sync_output = chat_model.completion(prompt, **parameters)
    async_output = asyncio.run(chat_model.async_completion(prompt))

    assert sync_output.reply != ''
    assert async_output.reply != ''


@pytest.mark.parametrize('chat_model', get_pytest_params('test_stream_chat_completion', types='model', exclude=['azure']))
def test_http_stream_chat_model(chat_model: HttpChatModel) -> None:
    chat_model.timeout = 10
    prompt = '这是测试，只回复你好'
    sync_output = list(chat_model.stream_completion(prompt))[-1]
    async_output = asyncio.run(async_stream_helper(chat_model, prompt))

    assert sync_output.stream.control in ('finish', 'done')
    assert sync_output.reply != ''
    assert async_output.reply != ''


async def async_stream_helper(model: ChatCompletionModel, prompt: Prompt) -> ChatCompletionModelStreamOutput:
    async for output in model.async_stream_completion(prompt):
        if output.stream.control == 'finish':
            return output
    raise RuntimeError('Stream did not finish')


@pytest.mark.parametrize(
    ('model_cls', 'parameters'),
    get_pytest_params('test_chat_parameters', types=('model_cls', 'parameter'), exclude=('zhipu-character', 'bailian')),
)
def test_init_chat_parameters(model_cls: Type[ChatCompletionModel], parameters: BaseModel, temperature: float = 0.8) -> None:
    parameters.temperature = temperature

    model = model_cls(parameters=parameters)

    assert model.parameters.temperature == temperature
