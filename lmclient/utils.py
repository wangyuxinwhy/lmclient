from __future__ import annotations

from lmclient.types import Message, MessageNotRequiredKeys, MessageRequiredKeys, Messages, Prompt


def ensure_messages(value: Prompt) -> Messages:
    if isinstance(value, str):
        return [Message(role='user', content=value)]
    else:
        messages: list[Message] = []
        for message_dict in value:
            temp_dict = {}
            for key in MessageRequiredKeys:
                temp_dict[key] = message_dict[key]
            for key in MessageNotRequiredKeys:
                if key in message_dict:
                    temp_dict[key] = message_dict[key]
            messages.append(Message(**temp_dict))
        return messages
