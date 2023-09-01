from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


class Message(BaseModel):
    role: str
    content: Union[str, FunctionCallDict]
    name: Optional[str] = None

    @property
    def is_function_call(self) -> bool:
        return isinstance(self.content, dict)


class ChatModelOutput(BaseModel):
    messages: Messages
    response: ModelResponse = Field(default_factory=dict)


class FunctionDict(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: dict


class GeneralParameters(BaseModel):
    temperature: float = 1
    top_p: float = 1
    max_tokens: Optional[int] = None
    functions: Optional[List[FunctionDict]] = None
    function_call: Optional[str] = None


class ModelParameters(BaseModel):

    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters):
        raise NotImplementedError


class FunctionCallDict(TypedDict):
    name: str
    arguments: str


Messages = List[Message]
ModelResponse = Dict[str, Any]
Prompt = Union[str, Messages]
