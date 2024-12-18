import logging
from typing import TYPE_CHECKING

from .._exceptions import OpenAIError
from ..resources import beta
from ..resources.beta import realtime
from .._compat import cached_property
from .._types import Query, Headers
from ..types.websocket_connection_options import WebsocketConnectionOptions
from .._base_client import _merge_mappings

if TYPE_CHECKING:
    from .azure import AzureOpenAI, AsyncAzureOpenAI

log: logging.Logger = logging.getLogger(__name__)


class RealtimeConnectionManager(realtime.realtime.RealtimeConnectionManager):
    def __enter__(self) -> realtime.realtime.RealtimeConnection:
        """
        👋 If your application doesn't work well with the context manager approach then you
        can call this method directly to initiate a connection.

        **Warning**: You must remember to close the connection with `.close()`.

        ```py
        connection = client.beta.realtime.connect(...).enter()
        # ...
        connection.close()
        ```
        """
        try:
            from websockets.sync.client import connect
        except ImportError as exc:
            raise OpenAIError("You need to install `openai[realtime]` to use this method") from exc

        auth_headers = self.__client.auth_headers
        extra_query = self.__extra_query
        self.__client: AzureOpenAI
        extra_query = {
            **self.__extra_query,
            "api-version": self.__client._api_version,
            "deployment": self.__client._azure_deployment or self.__model
        }
        if self.__client.api_key != "<missing API key>":
            auth_headers = {"api-key": self.__client.api_key}
        else:
            token = self.__client._get_azure_ad_token()
            if token:
                auth_headers = {"Authorization": f"Bearer {token}"}

        url = self._prepare_url().copy_with(
            params={
                **self.__client.base_url.params,
                "model": self.__model,
                **extra_query,
            },
        )
        log.debug("Connecting to %s", url)
        if self.__websocket_connection_options:
            log.debug("Connection options: %s", self.__websocket_connection_options)

        self.__connection = realtime.realtime.RealtimeConnection(
            connect(
                str(url),
                user_agent_header=self.__client.user_agent,
                additional_headers=_merge_mappings(
                    {
                        **auth_headers,
                        "OpenAI-Beta": "realtime=v1",
                    },
                    self.__extra_headers,
                ),
                **self.__websocket_connection_options,
            )
        )

        return self.__connection

    enter = __enter__


class Realtime(realtime.Realtime):

    def connect(
        self,
        *,
        model: str,
        extra_query: Query = {},
        extra_headers: Headers = {},
        websocket_connection_options: WebsocketConnectionOptions = {},
    ) -> RealtimeConnectionManager:
        """
        The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as function calling.

        Some notable benefits of the API include:

        - Native speech-to-speech: Skipping an intermediate text format means low latency and nuanced output.
        - Natural, steerable voices: The models have natural inflection and can laugh, whisper, and adhere to tone direction.
        - Simultaneous multimodal output: Text is useful for moderation; faster-than-realtime audio ensures stable playback.

        The Realtime API is a stateful, event-based API that communicates over a WebSocket.
        """
        return RealtimeConnectionManager(
            client=self._client,
            extra_query=extra_query,
            extra_headers=extra_headers,
            websocket_connection_options=websocket_connection_options,
            model=model,
        )

class Beta(beta.Beta):
    @cached_property
    def realtime(self) -> Realtime:
        return Realtime(self._client)


class AsyncRealtimeConnectionManager(realtime.realtime.AsyncRealtimeConnectionManager):
    async def __aenter__(self) -> realtime.realtime.AsyncRealtimeConnection:
        """
        👋 If your application doesn't work well with the context manager approach then you
        can call this method directly to initiate a connection.

        **Warning**: You must remember to close the connection with `.close()`.

        ```py
        connection = client.beta.realtime.connect(...).enter()
        # ...
        connection.close()
        ```
        """
        try:
            from websockets.asyncio.client import connect
        except ImportError as exc:
            raise OpenAIError("You need to install `openai[realtime]` to use this method") from exc

        auth_headers = self.__client.auth_headers
        extra_query = self.__extra_query
        self.__client: AsyncAzureOpenAI
        extra_query = {
            **self.__extra_query,
            "api-version": self.__client._api_version,
            "deployment": self.__client._azure_deployment or self.__model
        }
        if self.__client.api_key != "<missing API key>":
            auth_headers = {"api-key": self.__client.api_key}
        else:
            token = await self.__client._get_azure_ad_token()
            if token:
                auth_headers = {"Authorization": f"Bearer {token}"}

        url = self._prepare_url().copy_with(
            params={
                **self.__client.base_url.params,
                "model": self.__model,
                **extra_query,
            },
        )
        log.debug("Connecting to %s", url)
        if self.__websocket_connection_options:
            log.debug("Connection options: %s", self.__websocket_connection_options)

        self.__connection = realtime.realtime.AsyncRealtimeConnection(
            await connect(
                str(url),
                user_agent_header=self.__client.user_agent,
                additional_headers=_merge_mappings(
                    {
                        **auth_headers,
                        "OpenAI-Beta": "realtime=v1",
                    },
                    self.__extra_headers,
                ),
                **self.__websocket_connection_options,
            )
        )

        return self.__connection

    enter = __aenter__


class AsyncRealtime(realtime.AsyncRealtime):

    def connect(
        self,
        *,
        model: str,
        extra_query: Query = {},
        extra_headers: Headers = {},
        websocket_connection_options: WebsocketConnectionOptions = {},
    ) -> AsyncRealtimeConnectionManager:
        """
        The Realtime API enables you to build low-latency, multi-modal conversational experiences. It currently supports text and audio as both input and output, as well as function calling.

        Some notable benefits of the API include:

        - Native speech-to-speech: Skipping an intermediate text format means low latency and nuanced output.
        - Natural, steerable voices: The models have natural inflection and can laugh, whisper, and adhere to tone direction.
        - Simultaneous multimodal output: Text is useful for moderation; faster-than-realtime audio ensures stable playback.

        The Realtime API is a stateful, event-based API that communicates over a WebSocket.
        """
        return AsyncRealtimeConnectionManager(
            client=self._client,
            extra_query=extra_query,
            extra_headers=extra_headers,
            websocket_connection_options=websocket_connection_options,
            model=model,
        )

class AsyncBeta(beta.AsyncBeta):
    @cached_property
    def realtime(self) -> AsyncRealtime:
        return AsyncRealtime(self._client)
