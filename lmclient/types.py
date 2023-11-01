from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, SerializeAsAny, TypeAdapter
from typing_extensions import Annotated, NotRequired, Self, TypedDict

JsonSchema = Dict[str, Any]
Temperature = Annotated[float, Field(ge=0, le=1)]
Probability = Annotated[float, Field(ge=0, le=1)]


class TextMessage(TypedDict):
    role: Literal['user', 'assistant', 'function', 'system']
    name: NotRequired[str]
    content: str


class FunctionCallMessage(TypedDict):
    role: Literal['assistant']
    name: NotRequired[str]
    content: FunctionCall


class Function(TypedDict):
    name: str
    parameters: JsonSchema
    description: NotRequired[str]
    responses: NotRequired[JsonSchema]
    examples: NotRequired['Messages']


class FunctionCall(TypedDict):
    name: str
    arguments: str
    thoughts: NotRequired[str]


class GeneralParameters(BaseModel):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    functions: Optional[List[Function]] = None


class ModelParameters(BaseModel):
    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        #!todo add check logic
        return cls(**general_parameters.model_dump(exclude_none=True))

    def update_with_general_parameters(self, general_parameters: GeneralParameters) -> None:
        parameters = self.__class__.from_general_parameters(general_parameters)
        for key, value in parameters:
            setattr(self, key, value)


T_P = TypeVar('T_P', bound='ModelParameters')


class ChatModelOutput(BaseModel, Generic[T_P]):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    parameters: SerializeAsAny[T_P]
    messages: 'Messages'
    reply: str = ''
    hash_key: str = ''
    is_cache: bool = False
    error_message: Optional[str] = None

    @property
    def is_error(self) -> bool:
        return self.error_message is not None


ModelResponse = Dict[str, Any]
PathOrStr = Union[str, Path]
Message = Union[TextMessage, FunctionCallMessage]
Messages = Sequence[Message]
Prompt = Union[str, TextMessage, Messages]
Prompts = Sequence[Prompt]
text_message_validator = TypeAdapter(TextMessage)
prompt_validator = TypeAdapter(Prompt)
PrimitiveData = Optional[Union[str, int, float, bool]]
