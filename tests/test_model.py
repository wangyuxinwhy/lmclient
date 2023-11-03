from __future__ import annotations

import asyncio
from typing import Literal, Sequence, Type

import pytest

from lmclient.models import BaseChatModel, ModelRegistry, load_from_model_id
from lmclient.models.http import HttpChatModel
from lmclient.types import GeneralParameters, Messages, ModelParameters, TextMessage

param_type = Literal['model', 'model_cls', 'parameter', 'parameter_cls']


def get_pytest_params(id_prefix: str, types: Sequence[param_type] | param_type, exclude: Sequence[str] | None = None):
    exclude = exclude or []
    if isinstance(types, str):
        types = [types]

    pytest_params = []
    for model_name, (model_cls, paramter_cls) in ModelRegistry.items():
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


def test_load_from_model_id():
    model = load_from_model_id('openai/gpt-3.5-turbo')
    assert model.model_type == 'openai'
    assert model.name == 'gpt-3.5-turbo'


@pytest.mark.parametrize('chat_model', get_pytest_params('test_chat_completion', types='model'))
@pytest.mark.parametrize(
    'parameters',
    (
        None,
        GeneralParameters(temperature=0.5, top_p=0.85, max_tokens=20),
        GeneralParameters(temperature=0),
        GeneralParameters(top_p=0),
    ),
)
def test_http_chat_model(chat_model: HttpChatModel[ModelParameters], parameters: GeneralParameters | None):
    chat_model.timeout = 20
    test_messages = [TextMessage(role='user', content='这是测试，只回复你好')]
    sync_output = chat_model.chat_completion(test_messages, override_parameters=parameters)
    async_output = asyncio.run(chat_model.async_chat_completion(test_messages))

    assert sync_output.reply != ''
    assert async_output.reply != ''


@pytest.mark.parametrize('chat_model', get_pytest_params('test_stream_chat_completion', types='model', exclude=['azure']))
def test_http_stream_chat_model(chat_model: HttpChatModel[ModelParameters]):
    chat_model.timeout = 10
    test_messages = [TextMessage(role='user', content='这是测试，只回复你好')]
    sync_output = list(chat_model.stream_chat_completion(test_messages))[-1]
    async_output = asyncio.run(async_stream_helper(chat_model, test_messages))

    assert sync_output.stream.control == 'finish' or sync_output.stream.control == 'done'
    assert sync_output.reply != ''
    assert async_output.reply != ''


async def async_stream_helper(model: BaseChatModel, messages: Messages):
    async for output in model.async_stream_chat_completion(messages):
        if output.stream.control == 'finish':
            return output
    raise Exception('Stream did not finish')


@pytest.mark.parametrize(
    'model_cls, parameters',
    get_pytest_params('test_chat_parameters', types=('model_cls', 'parameter'), exclude='zhipu-character'),
)
def test_init_chat_parameters(model_cls: Type[BaseChatModel], parameters: ModelParameters):
    # Step 3: Create a ZhiPuChatParameters instance and set some parameters
    parameters.temperature = 0.8

    model = model_cls(parameters=parameters)

    assert model.parameters.temperature == 0.8
