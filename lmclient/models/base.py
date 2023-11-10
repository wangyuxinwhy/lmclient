from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Generic, Iterator, TypeVar

from pydantic import BaseModel
from typing_extensions import Self

from lmclient.message import Messages, Prompt
from lmclient.model_output import ChatModelOutput, ChatModelStreamOutput
from lmclient.utils import ensure_messages

P = TypeVar('P', bound=BaseModel)


class BaseChatModel(Generic[P], ABC):
    model_type: ClassVar[str]

    def __init__(self, parameters: P) -> None:
        self.parameters = parameters

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @classmethod
    @abstractmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        ...

    @abstractmethod
    def _chat_completion(self, messages: Messages, parameters: P) -> ChatModelOutput:
        ...

    @abstractmethod
    async def _async_chat_completion(self, messages: Messages, parameters: P) -> ChatModelOutput:
        ...

    @abstractmethod
    def _stream_chat_completion(self, messages: Messages, parameters: P) -> Iterator[ChatModelStreamOutput]:
        ...

    @abstractmethod
    def _async_stream_chat_completion(self, messages: Messages, parameters: P) -> AsyncIterator[ChatModelStreamOutput]:
        ...

    @property
    def model_id(self) -> str:
        return f'{self.model_type}/{self.name}'

    def chat_completion(self, prompt: Prompt, **override_parameters: Any) -> ChatModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        return self._chat_completion(messages, parameters)

    async def async_chat_completion(self, prompt: Prompt, **override_parameters: Any) -> ChatModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        return await self._async_chat_completion(messages, parameters)

    def stream_chat_completion(self, prompt: Prompt, **override_parameters: Any) -> Iterator[ChatModelStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        yield from self._stream_chat_completion(messages, parameters)

    async def async_stream_chat_completion(
        self, prompt: Prompt, **override_parameters: Any
    ) -> AsyncIterator[ChatModelStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        async for i in self._async_stream_chat_completion(messages, parameters):
            yield i

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate({**self.parameters.model_dump(), **override_parameters})
