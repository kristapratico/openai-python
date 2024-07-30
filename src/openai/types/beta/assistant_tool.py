# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated
from pydantic import field_serializer

from ..._utils import PropertyInfo
from ..._models import BaseModel
from .function_tool import FunctionTool
from .file_search_tool import FileSearchTool
from .code_interpreter_tool import CodeInterpreterTool

__all__ = ["AssistantTool"]


class BaseTool(BaseModel):
    type: None
    """A tool type"""

    @field_serializer('type', when_used='always')
    def serialize_unknown_type(self, type_: str) -> str:
        return type_


AssistantTool = Annotated[Union[BaseTool, CodeInterpreterTool, FileSearchTool, FunctionTool], PropertyInfo(discriminator="type")]
