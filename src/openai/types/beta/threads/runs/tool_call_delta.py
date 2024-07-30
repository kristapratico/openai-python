# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated
from pydantic import field_serializer

from ....._models import BaseModel
from ....._utils import PropertyInfo
from .function_tool_call_delta import FunctionToolCallDelta
from .file_search_tool_call_delta import FileSearchToolCallDelta
from .code_interpreter_tool_call_delta import CodeInterpreterToolCallDelta

__all__ = ["ToolCallDelta"]


class BaseToolCallDelta(BaseModel):
    index: int
    """The index of the tool call in the tool calls array."""

    type: None
    """The type of tool call.
    """

    @field_serializer('type', when_used='always')
    def serialize_unknown_type(self, type_: str) -> str:
        return type_


ToolCallDelta = Annotated[
    Union[BaseToolCallDelta, CodeInterpreterToolCallDelta, FileSearchToolCallDelta, FunctionToolCallDelta],
    PropertyInfo(discriminator="type"),
]
