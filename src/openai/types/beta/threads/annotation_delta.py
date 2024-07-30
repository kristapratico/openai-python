# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Annotated
from pydantic import field_serializer

from ...._models import BaseModel
from ...._utils import PropertyInfo
from .file_path_delta_annotation import FilePathDeltaAnnotation
from .file_citation_delta_annotation import FileCitationDeltaAnnotation

__all__ = ["AnnotationDelta"]


class BaseDeltaAnnotation(BaseModel):
    index: int
    """The index of the annotation in the text content part."""

    type: None
    """The type of annotation"""

    @field_serializer('type', when_used='always')
    def serialize_unknown_type(self, type_: str) -> str:
        return type_


AnnotationDelta = Annotated[
    Union[BaseDeltaAnnotation, FileCitationDeltaAnnotation, FilePathDeltaAnnotation], PropertyInfo(discriminator="type")
]
