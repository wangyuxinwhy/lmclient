# type: ignore
from __future__ import annotations

import hashlib

from typing_extensions import TypeGuard

from lmclient.types import FunctionCallMessage, Message, Messages, ModelParameters, Prompt, TextMessage, prompt_validator
from lmclient.version import __cache_version__


def generate_chat_completion_hash_key(model_id: str, messages: Messages, parameters: ModelParameters) -> str:
    messages_text = '---'.join([f'{k}={v}' for message in messages for k, v in message.items()])
    messages_hash = md5_hash(messages_text)
    parameters_hash = md5_hash(parameters.model_dump_json(exclude_none=True))
    return f'{model_id}|{messages_hash}|{parameters_hash}|v{__cache_version__}'


def md5_hash(string: str) -> str:
    return hashlib.md5(string.encode()).hexdigest()


def is_function_call_message(message: Message) -> TypeGuard[FunctionCallMessage]:
    return isinstance(message['content'], dict)


def is_text_message(message: Message) -> TypeGuard[TextMessage]:
    return isinstance(message['content'], str)


def ensure_messages(prompt: Prompt) -> Messages:
    prompt = prompt_validator.validate_python(prompt)

    if isinstance(prompt, str):
        return [TextMessage(role='user', content=prompt)]
    if isinstance(prompt, dict):
        return [prompt]
    else:
        return prompt
