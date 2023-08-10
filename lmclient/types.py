from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, TypedDict

from typing_extensions import NotRequired


class Message(TypedDict):
    role: str
    content: str
    name: NotRequired[str]
    function_call: NotRequired[str]


Messages = Sequence[Message]
ModelResponse = Dict[str, Any]


@dataclass
class TaskResult:
    response: ModelResponse = field(default_factory=dict)
    output: Any = None
    error_message: str | None = None
