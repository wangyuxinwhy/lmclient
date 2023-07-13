from __future__ import annotations

from typing import Protocol, Sequence, TypedDict, runtime_checkable

from typing_extensions import NotRequired


class Message(TypedDict):
    role: str
    content: str
    name: NotRequired[str]
    function_call: NotRequired[str]


Messages = Sequence[Message]


@runtime_checkable
class ChatModel(Protocol):
    identifier: str

    def complete(self, prompt: Messages | str, **kwargs) -> str:
        ...

    async def async_complete(self, prompt: Messages | str, **kwargs) -> str:
        ...
