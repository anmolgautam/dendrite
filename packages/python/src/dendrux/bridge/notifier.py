"""Bridge notifiers for transport and composition.

TransportNotifier pushes events to an asyncio.Queue for SSE consumption.
"""

from __future__ import annotations

import asyncio  # noqa: TC003 — used at runtime in Queue type hints
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dendrux.loops.base import LoopNotifier
from dendrux.notifiers.composite import CompositeNotifier

if TYPE_CHECKING:
    from collections.abc import Callable

    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult


@dataclass
class ServerEvent:
    """An event pushed to the SSE queue."""

    event: str
    data: dict[str, Any] = field(default_factory=dict)


class TransportNotifier(LoopNotifier):
    """Pushes loop events onto an asyncio.Queue for SSE streaming.

    Args:
        queue: The asyncio.Queue to push events to.
        redact: Optional redaction function applied to message content.
    """

    def __init__(
        self,
        queue: asyncio.Queue[ServerEvent],
        *,
        redact: Callable[[str], str] | None = None,
    ) -> None:
        self._queue = queue
        self._redact = redact

    async def on_message_appended(self, message: Message, iteration: int) -> None:
        content = message.content[:500]
        if self._redact:
            content = self._redact(content)
        await self._queue.put(
            ServerEvent(
                event="run.step",
                data={
                    "role": message.role.value,
                    "content": content,
                    "iteration": iteration,
                },
            )
        )

    async def on_llm_call_completed(
        self,
        response: LLMResponse,
        iteration: int,
        *,
        semantic_messages: list[Message] | None = None,
        semantic_tools: list[ToolDef] | None = None,
        duration_ms: int | None = None,
    ) -> None:
        await self._queue.put(
            ServerEvent(
                event="run.llm_done",
                data={
                    "iteration": iteration,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            )
        )

    async def on_tool_completed(
        self, tool_call: ToolCall, tool_result: ToolResult, iteration: int
    ) -> None:
        await self._queue.put(
            ServerEvent(
                event="run.tool_done",
                data={
                    "tool_name": tool_call.name,
                    "success": tool_result.success,
                    "iteration": iteration,
                },
            )
        )

    async def on_governance_event(
        self,
        event_type: str,
        iteration: int,
        data: dict[str, Any],
        correlation_id: str | None = None,
    ) -> None:
        await self._queue.put(
            ServerEvent(
                event="run.governance",
                data={
                    "event_type": event_type,
                    "iteration": iteration,
                    **data,
                },
            )
        )


__all__ = ["CompositeNotifier", "ServerEvent", "TransportNotifier"]
