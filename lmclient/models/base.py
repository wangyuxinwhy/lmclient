from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Generic, Type, TypeVar, cast

from typing_extensions import Self

from lmclient.cache import DiskCache
from lmclient.types import ChatModelOutput, Messages, ModelParameters
from lmclient.utils import generate_chat_completion_hash_key

T_P = TypeVar('T_P', bound=ModelParameters)
T_O = TypeVar('T_O', bound=ChatModelOutput)
DEFAULT_CACHE_DIR = Path(os.getenv('LMCLIENT_CACHE_DIR', '~/.cache/lmclient')).expanduser().resolve()


class BaseChatModel(Generic[T_P, T_O], ABC):
    model_type: ClassVar[str]
    chat_cache: DiskCache[T_O] | None
    parameters_type: Type[T_P]

    def __init__(self, parameters: T_P, cache: Path | str | bool = False) -> None:
        if cache is True:
            self.chat_cache = DiskCache[T_O](DEFAULT_CACHE_DIR)
        elif cache is False:
            self.chat_cache = None
        else:
            self.chat_cache = DiskCache[T_O](cache)

        self.parameters = parameters
        self.parameters_type: Type[T_P] = parameters.__class__

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def _chat_completion(self, messages: Messages, parameters: T_P) -> T_O:
        ...

    @abstractmethod
    async def _async_chat_completion(self, messages: Messages, parameters: T_P) -> T_O:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        ...

    @property
    def model_id(self) -> str:
        return f'{self.model_type}/{self.name}'

    def chat_completion(self, messages: Messages, override_parameters: T_P | None = None) -> T_O:
        if override_parameters is not None:
            parameters = self.parameters.model_copy(update=override_parameters.model_dump())
        else:
            parameters = self.parameters

        if self.chat_cache is not None:
            hash_key = generate_chat_completion_hash_key(self.model_id, messages, parameters)
            cached_output = self.chat_cache.get(hash_key)
            if cached_output is not None:
                cached_output.is_cache = True
                cached_output.hash_key = hash_key
                cached_output = cast(T_O, cached_output)
                return cached_output
            else:
                model_output = self._chat_completion(messages, parameters)
                model_output.hash_key = hash_key
                self.chat_cache.save(hash_key, model_output)
                return model_output
        else:
            model_output = self._chat_completion(messages, parameters)
            return model_output

    async def async_chat_completion(self, messages: Messages, override_parameters: T_P | None = None) -> T_O:
        if override_parameters is not None:
            parameters = self.parameters.model_copy(update=override_parameters.model_dump())
        else:
            parameters = self.parameters

        if self.chat_cache is not None:
            hash_key = generate_chat_completion_hash_key(self.model_id, messages, parameters)
            cached_output = self.chat_cache.get(hash_key)
            if cached_output is not None:
                cached_output.is_cache = True
                cached_output = cast(T_O, cached_output)
                return cached_output
            else:
                model_output = await self._async_chat_completion(messages, parameters)
                model_output.hash_key = hash_key
                self.chat_cache.save(hash_key, model_output)
                return model_output
        else:
            model_output = await self._async_chat_completion(messages, parameters)
            return model_output

    def update_parameters(self, **kwargs: Any):
        self.parameters = self.parameters.model_copy(update=kwargs)
