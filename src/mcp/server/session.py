"""
ServerSession Module

This module provides the ServerSession class, which manages communication between the
server and client in the MCP (Model Context Protocol) framework. It is most commonly
used in MCP servers to interact with the client.

Common usage pattern:
```
    server = Server(name)

    @server.call_tool()
    async def handle_tool_call(ctx: RequestContext, arguments: Dict[str, Any]) -> Any:
        # Check client capabilities before proceeding
        if ctx.session.check_client_capability(
            types.ClientCapabilities(experimental={"advanced_tools": dict()})
        ):
            # Perform advanced tool operations
            result = await perform_advanced_tool_operation(arguments)
        else:
            # Fall back to basic tool operations
            result = await perform_basic_tool_operation(arguments)

        return result

    @server.list_prompts()
    async def handle_list_prompts(ctx: RequestContext) -> List[types.Prompt]:
        # Access session for any necessary checks or operations
        if ctx.session.client_params:
            # Customize prompts based on client initialization parameters
            return generate_custom_prompts(ctx.session.client_params)
        else:
            return default_prompts
```

The ServerSession class is typically used internally by the Server class and should not
be instantiated directly by users of the MCP framework.
"""

from enum import Enum
from typing import Any, TypeVar, Union, Dict, List

import anyio
import anyio.lowlevel
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from pydantic import AnyUrl

import mcp.types as types
from mcp.server.models import InitializationOptions
from mcp.shared.session import (
    BaseSession,
    RequestResponder,
)


class InitializationState(Enum):
    NotInitialized = 1
    Initializing = 2
    Initialized = 3


ServerSessionT = TypeVar("ServerSessionT", bound="ServerSession")


class ServerSession(
    BaseSession[
        types.ServerRequest,
        types.ServerNotification,
        types.ServerResult,
        types.ClientRequest,
        types.ClientNotification,
    ]
):
    _initialized: InitializationState = InitializationState.NotInitialized
    _client_params: Union[types.InitializeRequestParams, None] = None

    def __init__(
        self,
        read_stream: MemoryObjectReceiveStream[Union[types.JSONRPCMessage, Exception]],
        write_stream: MemoryObjectSendStream[types.JSONRPCMessage],
        init_options: InitializationOptions,
    ) -> None:
        super().__init__(
            read_stream, write_stream, types.ClientRequest, types.ClientNotification
        )
        self._initialization_state = InitializationState.NotInitialized
        self._init_options = init_options

    @property
    def client_params(self) -> Union[types.InitializeRequestParams, None]:
        return self._client_params

    def check_client_capability(self, capability: types.ClientCapabilities) -> bool:
        """Check if the client supports a specific capability."""
        if self._client_params is None:
            return False

        # Get client capabilities from initialization params
        client_caps = self._client_params.capabilities

        # Check each specified capability in the passed in capability object
        if capability.roots is not None:
            if client_caps.roots is None:
                return False
            if capability.roots.listChanged and not client_caps.roots.listChanged:
                return False

        if capability.sampling is not None:
            if client_caps.sampling is None:
                return False

        if capability.experimental is not None:
            if client_caps.experimental is None:
                return False
            # Check each experimental capability
            for exp_key, exp_value in capability.experimental.items():
                if (
                    exp_key not in client_caps.experimental
                    or client_caps.experimental[exp_key] != exp_value
                ):
                    return False

        return True

    async def _received_request(
        self, responder: RequestResponder[types.ClientRequest, types.ServerResult]
    ):
        req = responder.request.root
        if isinstance(req, types.InitializeRequest):
            params = req.params
            self._initialization_state = InitializationState.Initializing
            self._client_params = params
            with responder:
                await responder.respond(
                    types.ServerResult(
                            types.InitializeResult(
                                protocolVersion=types.LATEST_PROTOCOL_VERSION,
                                capabilities=self._init_options.capabilities,
                                serverInfo=types.Implementation(
                                    name=self._init_options.server_name,
                                    version=self._init_options.server_version,
                                ),
                                instructions=self._init_options.instructions,
                            )
                        )
                    )
        else:
            if self._initialization_state != InitializationState.Initialized:
                raise RuntimeError(
                    "Received request before initialization was complete"
                )

    async def _received_notification(
        self, notification: types.ClientNotification
    ) -> None:
        # Need this to avoid ASYNC910
        await anyio.lowlevel.checkpoint()
        if isinstance(notification.root, types.InitializedNotification):
            self._initialization_state = InitializationState.Initialized
        else:
            if self._initialization_state != InitializationState.Initialized:
                raise RuntimeError(
                    "Received notification before initialization was complete"
                )

    async def send_log_message(
        self, level: types.LoggingLevel, data: Any, logger: Union[str, None] = None
    ) -> None:
        """Send a log message notification."""
        await self.send_notification(
            types.ServerNotification(
                types.LoggingMessageNotification(
                    method="notifications/message",
                    params=types.LoggingMessageNotificationParams(
                        level=level,
                        data=data,
                        logger=logger,
                    ),
                )
            )
        )

    async def send_resource_updated(self, uri: AnyUrl) -> None:
        """Send a resource updated notification."""
        await self.send_notification(
            types.ServerNotification(
                types.ResourceUpdatedNotification(
                    method="notifications/resources/updated",
                    params=types.ResourceUpdatedNotificationParams(uri=uri),
                )
            )
        )

    async def create_message(
        self,
        messages: List[types.SamplingMessage],
        *,
        max_tokens: int,
        system_prompt: Union[str, None] = None,
        include_context: Union[types.IncludeContext, None] = None,
        temperature: Union[float, None] = None,
        stop_sequences: Union[List[str], None] = None,
        metadata: Union[Dict[str, Any], None] = None,
        model_preferences: Union[types.ModelPreferences, None] = None,
    ) -> types.CreateMessageResult:
        """Send a sampling/create_message request."""
        return await self.send_request(
            types.ServerRequest(
                types.CreateMessageRequest(
                    method="sampling/createMessage",
                    params=types.CreateMessageRequestParams(
                        messages=messages,
                        systemPrompt=system_prompt,
                        includeContext=include_context,
                        temperature=temperature,
                        maxTokens=max_tokens,
                        stopSequences=stop_sequences,
                        metadata=metadata,
                        modelPreferences=model_preferences,
                    ),
                )
            ),
            types.CreateMessageResult,
        )

    async def list_roots(self) -> types.ListRootsResult:
        """Send a roots/list request."""
        return await self.send_request(
            types.ServerRequest(
                types.ListRootsRequest(
                    method="roots/list",
                )
            ),
            types.ListRootsResult,
        )

    async def send_ping(self) -> types.EmptyResult:
        """Send a ping request."""
        return await self.send_request(
            types.ServerRequest(
                types.PingRequest(
                    method="ping",
                )
            ),
            types.EmptyResult,
        )

    async def send_progress_notification(
        self, progress_token: Union[str, int], progress: float, total: Union[float, None] = None
    ) -> None:
        """Send a progress notification."""
        await self.send_notification(
            types.ServerNotification(
                types.ProgressNotification(
                    method="notifications/progress",
                    params=types.ProgressNotificationParams(
                        progressToken=progress_token,
                        progress=progress,
                        total=total,
                    ),
                )
            )
        )

    async def send_resource_list_changed(self) -> None:
        """Send a resource list changed notification."""
        await self.send_notification(
            types.ServerNotification(
                types.ResourceListChangedNotification(
                    method="notifications/resources/list_changed",
                )
            )
        )

    async def send_tool_list_changed(self) -> None:
        """Send a tool list changed notification."""
        await self.send_notification(
            types.ServerNotification(
                types.ToolListChangedNotification(
                    method="notifications/tools/list_changed",
                )
            )
        )

    async def send_prompt_list_changed(self) -> None:
        """Send a prompt list changed notification."""
        await self.send_notification(
            types.ServerNotification(
                types.PromptListChangedNotification(
                    method="notifications/prompts/list_changed",
                )
            )
        )