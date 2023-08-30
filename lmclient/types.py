from __future__ import annotations

from typing import Any, Dict, Generic, Sequence, TypedDict, TypeVar, Union

try:
    from pydantic.v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field

from typing_extensions import NotRequired

T = TypeVar('T')


class Message(TypedDict):
    role: str
    content: str
    name: NotRequired[str]
    function_call: NotRequired[str]


MessageRequiredKeys = ('role', 'content')
MessageNotRequiredKeys = ('name', 'function')
Messages = Sequence[Message]
ModelResponse = Dict[str, Any]
Prompt = Union[str, Sequence[dict]]


class ChatModelOutput(BaseModel, Generic[T]):  # type: ignore
    parsed_result: T
    response: ModelResponse = Field(default_factory=dict)
