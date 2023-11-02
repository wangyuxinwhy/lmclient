from __future__ import annotations

import anyio
import pytest

from lmclient.models import (
    AzureChat,
    BaichuanChat,
    HunyuanChat,
    MinimaxChat,
    MinimaxProChat,
    OpenAIChat,
    WenxinChat,
    ZhiPuChat,
)
from lmclient.models.http import HttpChatModel
from lmclient.types import GeneralParameters, ModelParameters, TextMessage


@pytest.mark.parametrize(
    'chat_model',
    (AzureChat(), MinimaxProChat(), MinimaxChat(), OpenAIChat(), ZhiPuChat(), WenxinChat(), HunyuanChat(), BaichuanChat()),
)
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
    chat_model.timeout = 10
    test_messages = [TextMessage(role='user', content='hello')]
    sync_output = chat_model.chat_completion(test_messages, override_parameters=parameters)
    async_output = anyio.run(chat_model.async_chat_completion, test_messages)

    assert sync_output.reply != ''
    assert async_output.reply != ''


@pytest.mark.parametrize(
    'chat_model',
    (MinimaxProChat(), MinimaxChat(), OpenAIChat(), ZhiPuChat(), WenxinChat(), HunyuanChat(), BaichuanChat()),
)
def test_http_stream_chat_model(chat_model: HttpChatModel[ModelParameters]):
    chat_model.timeout = 10
    test_messages = [TextMessage(role='user', content='这是测试，只回复你好')]
    sync_output = list(chat_model.stream_chat_completion(test_messages))[-1]

    assert sync_output.stream.control == 'finish' or sync_output.stream.control == 'done'
    assert sync_output.reply != ''
