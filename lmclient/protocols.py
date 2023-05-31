from typing import Protocol, runtime_checkable


@runtime_checkable
class CompletionModel(Protocol):
    def complete(self, prompt: str, **kwargs) -> str:
        ...

    async def async_complete(self, prompt: str, **kwargs) -> str:
        ...
