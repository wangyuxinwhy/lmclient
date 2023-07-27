from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Sequence, TypedDict, runtime_checkable

from typing_extensions import NotRequired


class Message(TypedDict):
    role: str
    content: str
    name: NotRequired[str]
    function_call: NotRequired[str]


Messages = Sequence[Message]
ModelResponse = Dict[str, Any]


@runtime_checkable
class ChatModel(Protocol):
    timeout: float | None
    identifier: str

    def chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        ...

    async def async_chat(self, prompt: Messages | str, **kwargs) -> ModelResponse:
        ...

    @staticmethod
    def default_postprocess_function(response: ModelResponse) -> ModelResponse:
        ...


@dataclass
class TaskResult:
    response: ModelResponse = field(default_factory=dict)
    error_message: str | None = None
