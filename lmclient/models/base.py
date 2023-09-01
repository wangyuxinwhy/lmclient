from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar, Generic, Type, TypeVar

from lmclient.cache import ChatCacheMixin
from lmclient.types import ChatModelOutput, Messages, ModelParameters

T = TypeVar("T", bound=ModelParameters)


class BaseChatModel(ABC, Generic[T], ChatCacheMixin):
    parameters_type: ClassVar[Type[ModelParameters]]

    def __init__(self, default_parameters: T | None = None, use_cache: Path | str | bool = False) -> None:
        super().__init__(use_cache=use_cache)
        self.default_parameters = default_parameters

    @property
    @abstractmethod
    def identifier(self) -> str:
        ...

    @abstractmethod
    def chat_completion(self, messages: Messages, override_parameters: T | None = None) -> ChatModelOutput:
        ...

    @abstractmethod
    async def async_chat_completion(self, messages: Messages, override_parameters: T | None = None) -> ChatModelOutput:
        ...
