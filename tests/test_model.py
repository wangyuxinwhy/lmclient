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

    assert isinstance(sync_output.response, dict)
    assert isinstance(sync_output.reply, str)
    assert isinstance(async_output.response, dict)
    assert isinstance(async_output.reply, str)
