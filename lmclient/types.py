from __future__ import annotations

from typing import Any, Dict, Optional, Union

from pydantic import Field
from typing_extensions import Annotated, NotRequired, TypedDict

JsonSchema = Dict[str, Any]
Temperature = Annotated[float, Field(ge=0, le=1)]
Probability = Annotated[float, Field(ge=0, le=1)]
PrimitiveData = Optional[Union[str, int, float, bool]]


class FunctionJsonSchema(TypedDict):
    name: str
    parameters: JsonSchema
    description: NotRequired[str]
