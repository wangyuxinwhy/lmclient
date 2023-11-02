from __future__ import annotations

from typing_extensions import TypeGuard

from lmclient.types import (
    FunctionCallMessage,
    Message,
    Messages,
    Prompt,
    TextMessage,
    prompt_validator,
)


def is_function_call_message(message: Message) -> TypeGuard[FunctionCallMessage]:
    return isinstance(message['content'], dict)


def is_text_message(message: Message) -> TypeGuard[TextMessage]:
    return isinstance(message['content'], str)


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
        return [prompt]
    else:
        return prompt
