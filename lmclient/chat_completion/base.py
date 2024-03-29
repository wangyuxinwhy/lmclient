from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, ClassVar, Generic, Iterator, TypeVar

from typing_extensions import Self, TypeGuard

from lmclient.chat_completion.message import Messages, Prompt, ensure_messages
from lmclient.chat_completion.model_output import ChatCompletionModelOutput, ChatCompletionModelStreamOutput
from lmclient.chat_completion.model_parameters import ModelParameters

P = TypeVar('P', bound=ModelParameters)


class ChatCompletionModel(Generic[P], ABC):
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
    def _completion(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        ...

    @abstractmethod
    async def _async_completion(self, messages: Messages, parameters: P) -> ChatCompletionModelOutput:
        ...

    @abstractmethod
    def _stream_completion(self, messages: Messages, parameters: P) -> Iterator[ChatCompletionModelStreamOutput]:
        ...

    @abstractmethod
    def _async_stream_completion(self, messages: Messages, parameters: P) -> AsyncIterator[ChatCompletionModelStreamOutput]:
        ...

    @property
    def model_id(self) -> str:
        return f'{self.model_type}/{self.name}'

    def completion(self, prompt: Prompt, **override_parameters: Any) -> ChatCompletionModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        return self._completion(messages, parameters)

    async def async_completion(self, prompt: Prompt, **override_parameters: Any) -> ChatCompletionModelOutput:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        return await self._async_completion(messages, parameters)

    def stream_completion(self, prompt: Prompt, **override_parameters: Any) -> Iterator[ChatCompletionModelStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        yield from self._stream_completion(messages, parameters)

    async def async_stream_completion(
        self, prompt: Prompt, **override_parameters: Any
    ) -> AsyncIterator[ChatCompletionModelStreamOutput]:
        parameters = self._merge_parameters(**override_parameters)
        messages = ensure_messages(prompt)
        async for i in self._async_stream_completion(messages, parameters):
            yield i

    def _merge_parameters(self, **override_parameters: Any) -> P:
        return self.parameters.__class__.model_validate(
            {**self.parameters.model_dump(exclude_unset=True), **override_parameters}
        )


def is_stream_model_output(model_output: ChatCompletionModelOutput) -> TypeGuard[ChatCompletionModelStreamOutput]:
    return getattr(model_output, 'stream', None) is not None
