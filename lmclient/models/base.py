from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Generic, Iterator, Type, TypeVar, Union

from typing_extensions import Self

from lmclient.types import (
    ChatModelOutput,
    ChatModelStreamOutput,
    GeneralParameters,
    Messages,
    ModelParameters,
    Prompt,
)
from lmclient.utils import ensure_messages

T_P = TypeVar('T_P', bound=ModelParameters)
OverrideParameters = Union[T_P, GeneralParameters, None]


class BaseChatModel(Generic[T_P], ABC):
    model_type: ClassVar[str]
    parameters_type: Type[T_P]

    def __init__(self, parameters: T_P) -> None:
        self.parameters = parameters
        self.parameters_type: Type[T_P] = parameters.__class__

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        ...

    @abstractmethod
    def _chat_completion(self, messages: Messages, parameters: T_P) -> ChatModelOutput[T_P]:
        ...

    @abstractmethod
    async def _async_chat_completion(self, messages: Messages, parameters: T_P) -> ChatModelOutput[T_P]:
        ...

    @abstractmethod
    def _stream_chat_completion(self, messages: Messages, parameters: T_P) -> Iterator[ChatModelStreamOutput[T_P]]:
        ...

    @abstractmethod
    def _async_stream_chat_completion(self, messages: Messages, parameters: T_P) -> AsyncIterator[ChatModelStreamOutput[T_P]]:
        ...

    @property
    def model_id(self) -> str:
        return f'{self.model_type}/{self.name}'

    def chat_completion(
        self, prompt: Prompt, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> ChatModelOutput[T_P]:
        parameters = self.merge_parameters(override_parameters, **kwargs)
        messages = ensure_messages(prompt)
        return self._chat_completion(messages, parameters)

    async def async_chat_completion(
        self, prompt: Prompt, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> ChatModelOutput[T_P]:
        parameters = self.merge_parameters(override_parameters, **kwargs)
        messages = ensure_messages(prompt)
        return await self._async_chat_completion(messages, parameters)

    def stream_chat_completion(
        self, prompt: Prompt, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ) -> Iterator[ChatModelStreamOutput[T_P]]:
        parameters = self.merge_parameters(override_parameters, **kwargs)
        messages = ensure_messages(prompt)
        yield from self._stream_chat_completion(messages, parameters)

    async def async_stream_chat_completion(
        self, prompt: Prompt, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any
    ):
        parameters = self.merge_parameters(override_parameters, **kwargs)
        messages = ensure_messages(prompt)
        async for i in self._async_stream_chat_completion(messages, parameters):
            yield i

    def update_parameters(self, **kwargs: Any) -> None:
        self.parameters = self.parameters.model_copy(update=kwargs)

    def merge_parameters(self, override_parameters: OverrideParameters[T_P] = None, **kwargs: Any) -> T_P:
        if override_parameters is None:
            update_parameters = {}
        elif isinstance(override_parameters, GeneralParameters):
            update_parameters = self.parameters_type.from_general_parameters(override_parameters).model_dump(exclude_none=True)
        else:
            update_parameters = override_parameters.model_dump(exclude_none=True)

        if kwargs:
            update_parameters.update(kwargs)

        parameters_dict = {**self.parameters.model_dump(exclude_none=True), **update_parameters}
        return self.parameters_type.model_validate(parameters_dict)
