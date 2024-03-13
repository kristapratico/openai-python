import functools
from typing import TypeVar, Callable

from typing_extensions import ParamSpec
import httpx
from opentelemetry import trace

P = ParamSpec("P")
T = TypeVar("T")

tracer = trace.get_tracer("openai")


def traceable(func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        with tracer.start_as_current_span(f"{func.__module__}"):
            return func(*args, **kwargs)
    return wrapper


def on_request(request: httpx.Request) -> None:
    with tracer.start_as_current_span("request") as req:
        req.set_attribute("headers", request.headers["user-agent"])
        # add more custom attrs

def on_response(response: httpx.Response) -> None:
    with tracer.start_as_current_span("response") as res:
        res.set_attribute("status_code", response.status_code)
        # add more custom attrs