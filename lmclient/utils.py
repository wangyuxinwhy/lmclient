from __future__ import annotations

from typing import Any

from typing_extensions import TypeGuard

from lmclient.message import Message, Messages, MessageTypes, Prompt, TextMessage, message_validator, prompt_validator
from lmclient.model_output import ChatModelOutput, ChatModelStreamOutput


def is_stream_model_output(model_output: ChatModelOutput) -> TypeGuard[ChatModelStreamOutput]:
    return getattr(model_output, 'stream', None) is not None


def is_message(message: Any) -> TypeGuard[Message]:
    return isinstance(message, MessageTypes)


def ensure_messages(prompt: Prompt) -> Messages:
    """
    Ensure that the given prompt is in the form of a list of messages.

    Args:
        prompt (Prompt): The prompt to be validated.

    Returns:
        Messages: A list of message.

    Raises:
        ValidationError: If the prompt is not valid.
    """
    prompt = prompt_validator.validate_python(prompt)

    if isinstance(prompt, str):
        return [TextMessage(role='user', content=prompt)]
    if isinstance(prompt, dict):
        return [message_validator.validate_python(prompt)]
    if isinstance(prompt, MessageTypes):
        return [prompt]

    messages: list[Message] = []
    for i in prompt:
        if is_message(i):
            messages.append(i)
        else:
            messages.append(message_validator.validate_python(i))
    return messages
