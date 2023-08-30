from typing import Protocol, TypeVar

from lmclient.types import ModelResponse

T = TypeVar('T', covariant=True)


class ModelResponseParser(Protocol[T]):
    def __call__(self, response: ModelResponse) -> T:
        ...


class ParserError(Exception):
    """Error raised when postprocess function fails."""
