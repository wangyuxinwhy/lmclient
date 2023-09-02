from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, Self, TypedDict

Messages = List['Message']
ModelResponse = Dict[str, Any]
Prompt = Union[str, 'Message', 'MessageDict', Sequence[Union['MessageDict', 'Message']]]


class FunctionDict(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, Any]


class FunctionCallDict(TypedDict):
    name: str
    arguments: str


class Message(BaseModel):
    role: str
    content: Union[str, FunctionCallDict]
    name: Optional[str] = None

    @property
    def is_function_call(self) -> bool:
        return isinstance(self.content, dict)


class MessageDict(TypedDict):
    role: str
    content: Union[str, FunctionCallDict]
    name: NotRequired[str]


class GeneralParameters(BaseModel):
    temperature: float = 1
    top_p: float = 1
    max_tokens: Optional[int] = None
    functions: Optional[List[FunctionDict]] = None
    function_call: Optional[str] = None


class ChatModelOutput(BaseModel):
    messages: Messages
    hash_key: str = ''
    is_cache: bool = False
    reply: str = ''


class ModelParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        raise NotImplementedError


class RetryStrategy(BaseModel):
    min_wait_seconds: int = 2
    max_wait_seconds: int = 20
    max_attempt: int = 3


class HttpChatModelOutput(ChatModelOutput):
    response: ModelResponse = Field(default_factory=dict)
