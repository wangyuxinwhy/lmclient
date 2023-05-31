from typing import Protocol, runtime_checkable


@runtime_checkable
class CompletionModel(Protocol):
    identifier: str

    def complete(self, prompt: str, **kwargs) -> str:
        ...

    async def async_complete(self, prompt: str, **kwargs) -> str:
        ...
