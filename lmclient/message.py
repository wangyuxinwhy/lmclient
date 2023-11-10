from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Union

from pydantic import BaseModel, TypeAdapter


class TextMessage(BaseModel):
    role: Literal['user', 'assistant', 'function', 'system']
    name: Optional[str] = None
    content: str


class FunctionCall(BaseModel):
    name: str
    arguments: str
    thoughts: Optional[str] = None


class FunctionCallMessage(BaseModel):
    role: Literal['assistant']
    name: Optional[str] = None
    content: FunctionCall


MessageTypes = (TextMessage, FunctionCallMessage)
Message = Union[TextMessage, FunctionCallMessage]
Messages = Sequence[Message]
MessageDict = Dict[str, Any]
MessageDicts = Sequence[MessageDict]
Prompt = Union[str, Message, Messages, MessageDict, MessageDicts]
Prompts = Sequence[Prompt]
message_validator = TypeAdapter(Message)
prompt_validator = TypeAdapter(Prompt)
