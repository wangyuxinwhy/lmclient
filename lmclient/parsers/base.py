from typing import Any, Protocol

from lmclient.types import ModelResponse


class ModelResponseParser(Protocol):
    def __call__(self, response: ModelResponse) -> Any:
        ...
