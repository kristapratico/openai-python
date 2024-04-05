from __future__ import annotations

import json
import functools
from typing import TypeVar, Callable, Any
from typing_extensions import ParamSpec
import httpx

from ._extras import opentelemetry as trace, has_tracing_enabled

P = ParamSpec("P")
T = TypeVar("T")


def on_request(request: httpx.Request) -> None:
    span = trace.get_current_span()
    span.set_attribute("method", request.method)
    span.set_attribute("url", str(request.url))
    span.set_attribute("user-agent", request.headers.get("user-agent", ""))
    span.set_attribute("content-length", request.headers.get("content-length", ""))


def on_response(response: httpx.Response) -> None:
    span = trace.get_current_span()
    span.set_attribute("status_code", response.status_code)
    span.set_attribute("content-type", response.headers.get("content-type", ""))
    span.set_attribute("content-length", response.headers.get("content-length", ""))
    span.set_attribute("apim-request-id", response.headers.get("apim-request-id", ""))
    span.set_attribute("x-ratelimit-remaining-requests", response.headers.get("x-ratelimit-remaining-requests", ""))
    span.set_attribute("x-ratelimit-remaining-tokens", response.headers.get("x-ratelimit-remaining-tokens", ""))
    span.set_attribute("x-request-id", response.headers.get("x-request-id", ""))
    span.set_attribute("x-ms-client-request-id", response.headers.get("x-ms-client-request-id", ""))


def add_request_body_attributes(span: trace.Span, kwargs: Any) -> None:
    span.add_event(name="gen_ai.prompt", attributes={"event.body": json.dumps(kwargs, indent=2)})
    span.set_attribute("gen_ai.system", "openai")
    span.set_attribute("gen_ai.request.max_tokens", kwargs.get("max_tokens", ""))
    span.set_attribute("gen_ai.request.model", kwargs.get("model", ""))
    span.set_attribute("gen_ai.request.temperature", kwargs.get("temperature", ""))
    span.set_attribute("gen_ai.request.top_p", kwargs.get("top_p", ""))
    span.set_attribute("gen_ai.request.stream", kwargs.get("stream", ""))  # not LLM conv yet


def add_response_body_attributes(span: trace.Span, result: object) -> None:
    if hasattr(result, "__stream__"):
        return

    # TODO cast result
    span.add_event(name="gen_ai.completion", attributes={"event.body": result.model_dump_json(indent=2)})
    span.set_attribute("gen_ai.response.model", result.model)
    span.set_attribute("gen_ai.usage.completion_tokens", result.usage.completion_tokens)
    span.set_attribute("gen_ai.usage.prompt_tokens", result.usage.prompt_tokens)
    span.set_attribute("gen_ai.response.id", result.id)
    span.set_attribute("gen_ai.response.finish_reasons", result.choices[0].finish_reason)


def traceable(
    __func: Callable[P, T] | None = None, *, span_name: str
) -> Callable[P, T]:

    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> T:
            if not has_tracing_enabled():
                return func(*args, **kwargs)

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(span_name) as llm_span:
                add_request_body_attributes(llm_span, kwargs)
                result = func(*args, **kwargs)
                add_response_body_attributes(llm_span, result)
                return result

        return inner
    return wrapper if __func is None else wrapper(__func)


TRACING_EVENT_HOOKS = {"request": [on_request], "response": [on_response]}
