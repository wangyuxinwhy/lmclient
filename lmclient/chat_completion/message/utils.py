from typing import Any, Literal

from lmclient.chat_completion.message.core import (
    FunctionCall,
    Message,
    Messages,
    Prompt,
    UserMessage,
    content_validator,
    message_validator,
)


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
    if isinstance(prompt, str):
        return [UserMessage(role='user', content=prompt)]
    if isinstance(prompt, dict):
        if prompt['role'] == 'assistant':
            prompt['content_type'] = infer_content_type(prompt['content'])
        return [message_validator.validate_python(prompt)]
    if isinstance(prompt, Message):
        return [prompt]

    messages: list[Message] = []
    for i in prompt:
        if isinstance(i, Message):
            messages.append(i)
        else:
            if i['role'] == 'assistant':
                i['content_type'] = infer_content_type(i['content'])
            messages.append(message_validator.validate_python(i))
    return messages


def infer_content_type(message_content: Any) -> Literal['text', 'function_call', 'tool_calls']:
    obj = content_validator.validate_python(message_content)
    if isinstance(obj, str):
        return 'text'
    if isinstance(obj, FunctionCall):
        return 'function_call'
    if isinstance(obj, list):
        return 'tool_calls'
    raise ValueError(f'Unknown content type: {obj}')
