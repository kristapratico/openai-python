# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated, Literal

from ...._compat import PYDANTIC_V2
from ...._models import BaseModel
from ...._utils import PropertyInfo
from .file_path_annotation import FilePathAnnotation
from .file_citation_annotation import FileCitationAnnotation

if PYDANTIC_V2:
    from pydantic import field_serializer


__all__ = ["Annotation", "BaseAnnotation"]


class BaseAnnotation(BaseModel):
    text: str
    """The index of the annotation in the text content part."""

    type: Literal["unknown"]
    """The type of annotation"""

    if PYDANTIC_V2:
        @field_serializer('type', when_used='always')
        def serialize_unknown_type(self, type_: str) -> str:
            return type_


Annotation = Annotated[Union[BaseAnnotation, FileCitationAnnotation, FilePathAnnotation], PropertyInfo(discriminator="type")]
