from __future__ import annotations

from typing import Any, Dict, Generic, Optional, Sequence, TypedDict, TypeVar

from pydantic import BaseModel, Field
from typing_extensions import NotRequired

T = TypeVar('T')


class Message(TypedDict):
    role: str
    content: str
    name: NotRequired[str]
    function_call: NotRequired[str]


Messages = Sequence[Message]
ModelResponse = Dict[str, Any]


class TaskResult(BaseModel, Generic[T]):
    output: Optional[T] = None
    response: ModelResponse = Field(default_factory=dict)
    error_message: Optional[str] = None
