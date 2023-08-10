from typing import Any, Generic, Protocol, TypeVar

from lmclient.types import ModelResponse

T = TypeVar('T', covariant=True)


class ModelResponseParser(Protocol, Generic[T]):
    def __call__(self, response: ModelResponse) -> T:
        ...
