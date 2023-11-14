from __future__ import annotations

import asyncio
from typing import Type

import pytest

from lmclient.parameters import ModelParameters
from lmclient.test import get_pytest_params
from lmclient.text_to_speech import (
    SpeechModelRegistry,
    SpeechModels,
    TextToSpeechModel,
    load_speech_model,
)


def test_model_type_is_unique() -> None:
    assert len(SpeechModels) == len(SpeechModelRegistry)


def test_load_from_model_id() -> None:
    model = load_speech_model('openai/tts-1-hd')
    assert model.model_type == 'openai'
    assert model.name == 'tts-1-hd'


@pytest.mark.parametrize('speech_model', get_pytest_params('test_text_to_speech', SpeechModelRegistry, types='model'))
def test_speech_model(speech_model: TextToSpeechModel) -> None:
    prompt = '你好，这是一个测试用例'
    sync_output = speech_model.generate(prompt)
    async_output = asyncio.run(speech_model.async_generate(prompt))

    assert len(sync_output.audio) != 0
    assert len(async_output.audio) != 0


@pytest.mark.parametrize(
    ('model_cls', 'parameters'),
    get_pytest_params('test_speech_parameters', SpeechModelRegistry, types=('model_cls', 'parameter')),
)
def test_init_chat_parameters(model_cls: Type[TextToSpeechModel], parameters: ModelParameters) -> None:
    parameters.voice_id = 'test'
    model = model_cls(parameters=parameters)
    assert model.parameters.voice_id == 'test'
