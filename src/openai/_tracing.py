from __future__ import annotations

import functools
from typing import TypeVar, Callable, Any
from typing_extensions import ParamSpec
import httpx
from opentelemetry import trace

P = ParamSpec("P")
T = TypeVar("T")

tracer = trace.get_tracer("openai")


def on_request(request: httpx.Request) -> None:
    with tracer.start_as_current_span("request") as span:
        span.set_attribute("method", request.method)
        span.set_attribute("url", str(request.url))
        span.set_attribute("user-agent", request.headers.get("user-agent", ""))
        span.set_attribute("content-length", request.headers.get("content-length", ""))


def on_response(response: httpx.Response) -> None:
    with tracer.start_as_current_span("response") as span:
        span.set_attribute("status_code", response.status_code)
        span.set_attribute("content-type", response.headers.get("content-type", ""))
        span.set_attribute("content-length", response.headers.get("content-length", ""))
        span.set_attribute("apim-request-id", response.headers.get("apim-request-id", ""))
        span.set_attribute("x-ratelimit-remaining-requests", response.headers.get("x-ratelimit-remaining-requests", ""))
        span.set_attribute("x-ratelimit-remaining-tokens", response.headers.get("x-ratelimit-remaining-tokens", ""))
        span.set_attribute("x-request-id", response.headers.get("x-request-id", ""))
        span.set_attribute("x-ms-client-request-id", response.headers.get("x-ms-client-request-id", ""))


def add_request_body_attributes(span: trace.Span, args: Any, kwargs: Any) -> None:
    span.set_attribute("request.model", kwargs.get("model", ""))
    span.set_attribute("request.max_tokens", kwargs.get("max_tokens", ""))
    # add more model-specific attributes here


def add_response_body_attributes(span: trace.Span, result: object) -> None:
    span.set_attribute("response.model", result.model)
    # add more model-specific attributes here


def traceable(
    __func: Callable[P, T] | None = None, *, span_name: str
) -> Callable[P, T]:

    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            with tracer.start_as_current_span(span_name) as span:
                add_request_body_attributes(span, args, kwargs)
                result = func(*args, **kwargs)
                add_response_body_attributes(span, result)
                return result

        return inner
    return wrapper if __func is None else wrapper(__func)
