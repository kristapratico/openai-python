from __future__ import annotations

import json
import functools
from typing import TypeVar, Callable, Any, Generator, Iterator, Union
from typing_extensions import ParamSpec

from ._extras import opentelemetry as trace, has_tracing_enabled
from .types.chat import ChatCompletion, ChatCompletionChunk
from .types.completion import Completion
from ._streaming import Stream

TracedModels = Union[ChatCompletion, Completion]

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _set_attribute(span: trace.Span, key: str, value: Any) -> None:
    if value:
        span.set_attribute(key, value)


def _add_request_chat_message_event(span: trace.Span, **kwargs: Any) -> None:
    for message in kwargs.get("messages", []):
        try:
            message = message.to_dict()
        except AttributeError:
            pass

        name = ""
        data = {"role": message.get("role"), "content": message.get("content")}
        if message.get("role") == "user":
            name = "gen_ai.user.message"
            if message.get("name"):
                data.update({"name": message.get("name")})
        elif message.get("role") == "system":
            name = "gen_ai.system.message"
            if message.get("name"):
                data.update({"name": message.get("name")})
        elif message.get("role") == "assistant":
            name = "gen_ai.assistant.message"
            if message.get("tool_calls"):
                data.update({"tool_calls": message.get("tool_calls")})
        elif message.get("role") == "tool":
            name = "gen_ai.tool.message"
            data.update({"tool_call_id": message.get("tool_call_id")})
        if name and data:
            span.add_event(
                name=name,
                attributes={"event.data": json.dumps(data, indent=2)}
            )


def _add_request_chat_attributes(span: trace.Span, **kwargs: Any) -> None:
    _set_attribute(span, "gen_ai.system", "openai")
    _set_attribute(span, "gen_ai.request.model", kwargs.get("model"))
    _set_attribute(span, "gen_ai.request.max_tokens", kwargs.get("max_tokens"))
    _set_attribute(span, "gen_ai.request.temperature", kwargs.get("temperature"))
    _set_attribute(span, "gen_ai.request.top_p", kwargs.get("top_p"))


def _add_response_chat_message_event(span: trace.Span, result: ChatCompletion) -> None:
    for choice in result.choices:
        response = {
            "message.role": choice.message.role,
            "message.content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "index": choice.index,
        }
        if choice.message.tool_calls:
            response.update({"message.tool_calls": "".join([tool.to_json() for tool in choice.message.tool_calls])})
        span.add_event(name="gen_ai.response.message", attributes={"event.data": json.dumps(response, indent=2)})


def _add_response_chat_attributes(span: trace.Span, result: ChatCompletion) -> None:
    _set_attribute(span, "gen_ai.response.id", result.id)
    _set_attribute(span, "gen_ai.response.model", result.model)
    _set_attribute(span, "gen_ai.response.finish_reason", result.choices[0].finish_reason)
    if hasattr(result, "usage"):
        _set_attribute(span, "gen_ai.usage.completion_tokens", result.usage.completion_tokens if result.usage else None)
        _set_attribute(span, "gen_ai.usage.prompt_tokens", result.usage.prompt_tokens if result.usage else None)


def _traceable_stream(stream_obj: Stream[ChatCompletionChunk], span: trace.Span) -> Generator[ChatCompletionChunk, None, None]:
    try:
        accumulate: dict[str, Any] = {"role": ""}
        for chunk in stream_obj:
            for item in chunk.choices:
                if item.finish_reason:
                    accumulate["finish_reason"] = item.finish_reason
                if item.index:
                    accumulate["index"] = item.index
                if item.delta.role:
                    accumulate["role"] = item.delta.role
                if item.delta.content:
                    accumulate.setdefault("content", "")
                    accumulate["content"] += item.delta.content
                if item.delta.tool_calls:
                    accumulate.setdefault("tool_calls", [])
                    for tool_call in item.delta.tool_calls:
                        if tool_call.id:
                            accumulate["tool_calls"].append({"id": tool_call.id, "type": "", "function": {"name": "", "arguments": ""}})
                        if tool_call.type:
                            accumulate["tool_calls"][-1]["type"] = tool_call.type
                        if tool_call.function and tool_call.function.name:
                            accumulate["tool_calls"][-1]["function"]["name"] = tool_call.function.name
                        if tool_call.function and tool_call.function.arguments:
                            accumulate["tool_calls"][-1]["function"]["arguments"] += tool_call.function.arguments
            yield chunk

        span.add_event(name="gen_ai.response.message", attributes={"event.data": json.dumps(accumulate, indent=2)})
        _add_response_chat_attributes(span, chunk)

    finally:
        span.end()


def _wrapped_stream(stream_obj: Stream[ChatCompletionChunk], span: trace.Span) -> Stream[ChatCompletionChunk]:
    import wrapt

    class StreamWrapper(wrapt.ObjectProxy):
        def __iter__(self) -> Iterator[ChatCompletionChunk]:
            return _traceable_stream(stream_obj, span)

    return StreamWrapper(stream_obj)


def _add_request_span_attributes(span: trace.Span, span_name: str, kwargs: Any) -> None:
    if span_name.startswith("chat.completions.create"):
        _add_request_chat_attributes(span, **kwargs)
        _add_request_chat_message_event(span, **kwargs)
    # TODO add more models here


def _add_response_span_attributes(span: trace.Span, result: TracedModels) -> None:
    if result.object == "chat.completion":
        _add_response_chat_attributes(span, result)
        _add_response_chat_message_event(span, result)
    # TODO add more models here


def traceable(
    *, span_name: str
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    tracer = trace.get_tracer(__name__)
    enabled = has_tracing_enabled()

    def wrapper(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def inner(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            if not enabled:
                return func(*args, **kwargs)

            span = tracer.start_span(span_name, kind=trace.SpanKind.CLIENT)
            try:
                _add_request_span_attributes(span, span_name, kwargs)

                result = func(*args, **kwargs)

                if hasattr(result, "__stream__"):
                    return _wrapped_stream(result, span)

                _add_response_span_attributes(span, result)

            except Exception:
                span.end()
                raise

            span.end()
            return result

        return inner

    return wrapper
