from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from .._utils import LazyProxy
from ._common import MissingDependencyError, format_instructions

if TYPE_CHECKING:
    import opentelemetry.trace as opentelemetry

TRACING_INSTRUCTIONS = format_instructions(library="opentelemetry", extra="tracing")


class OpenTelemetryProxy(LazyProxy[Any]):
    @override
    def __load__(self) -> Any:
        try:
            import opentelemetry.trace
        except ModuleNotFoundError as err:
            raise MissingDependencyError(TRACING_INSTRUCTIONS) from err

        return opentelemetry.trace


if not TYPE_CHECKING:
    opentelemetry = OpenTelemetryProxy()



def has_tracing_enabled() -> bool:
    tracing = os.getenv("OPENAI_TRACE_ENABLED")
    if tracing is None or tracing.lower() == "false":
        return False
    return True
