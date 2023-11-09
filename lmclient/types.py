from __future__ import annotations

from typing import Any, Dict, Generic, List, Literal, Optional, Sequence, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, SerializeAsAny, TypeAdapter
from typing_extensions import Annotated, NotRequired, Self, TypedDict

JsonSchema = Dict[str, Any]
Temperature = Annotated[float, Field(ge=0, le=1)]
Probability = Annotated[float, Field(ge=0, le=1)]
T_P = TypeVar('T_P', bound='ModelParameters')
T_O = TypeVar('T_O', bound='ChatModelOutput')


class TextMessage(BaseModel):
    role: Literal['user', 'assistant', 'function', 'system']
    name: Optional[str] = None
    content: str


class FunctionCall(BaseModel):
    name: str
    arguments: str
    thoughts: Optional[str] = None


class FunctionCallMessage(BaseModel):
    role: Literal['assistant']
    name: Optional[str] = None
    content: FunctionCall


class FunctionJsonSchema(TypedDict):
    name: str
    parameters: JsonSchema
    description: NotRequired[str]
    responses: NotRequired[JsonSchema]
    examples: NotRequired['Messages']


class GeneralParameters(BaseModel):
    temperature: Optional[Temperature] = None
    top_p: Optional[Probability] = None
    max_tokens: Optional[PositiveInt] = None
    functions: Optional[List[FunctionJsonSchema]] = None


class ModelParameters(BaseModel):
    @classmethod
    def from_general_parameters(cls, general_parameters: GeneralParameters) -> Self:
        #!todo add check logic
        return cls(**general_parameters.model_dump(exclude_none=True))

    def update_with_general_parameters(self, general_parameters: GeneralParameters) -> None:
        parameters = self.__class__.from_general_parameters(general_parameters)
        parameters = parameters.model_dump(exclude_unset=True, exclude_none=True)
        for key, value in parameters.items():
            setattr(self, key, value)


class ChatModelOutput(BaseModel, Generic[T_P]):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    parameters: SerializeAsAny[T_P]
    messages: 'Messages'
    error: Optional[str] = None
    extra_info: Dict[str, Any] = {}

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def last_message(self) -> Message | None:
        if self.messages:
            return self.messages[-1]
        return None

    @property
    def reply(self) -> str:
        if self.last_message and isinstance(self.last_message, TextMessage):
            return self.last_message.content
        return ''


class Stream(BaseModel):
    delta: str = ''
    control: Literal['start', 'continue', 'finish', 'done']


class ChatModelStreamOutput(ChatModelOutput[T_P]):
    stream: Stream

    @property
    def is_finish(self) -> bool:
        return self.stream.control == 'finish'


ModelResponse = Dict[str, Any]
Message = Union[TextMessage, FunctionCallMessage]
Messages = Sequence[Message]
MessageTypes = (TextMessage, FunctionCallMessage)
MessageDict = Dict[str, Any]
MessageDicts = Sequence[MessageDict]
Prompt = Union[str, Message, Messages, MessageDict, MessageDicts]
Prompts = Sequence[Prompt]
message_validator = TypeAdapter(Message)
prompt_validator = TypeAdapter(Prompt)
PrimitiveData = Optional[Union[str, int, float, bool]]

ChatModelOutput.model_rebuild()
ChatModelStreamOutput.model_rebuild()
GeneralParameters.model_rebuild()
