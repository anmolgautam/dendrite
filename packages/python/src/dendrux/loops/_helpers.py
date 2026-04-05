"""Shared notification helpers for loop implementations.

Two seams with different failure contracts:

  record_* — call LoopRecorder (internal persistence). Exceptions PROPAGATE.
             If persistence fails, the run stops.

  notify_* — call LoopObserver (best-effort notifications). Exceptions SWALLOWED.
             Console printing, Slack, SSE — if they fail, the run continues.

At each event point, the loop calls record first, then notify.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrux.loops.base import LoopObserver, LoopRecorder
    from dendrux.types import LLMResponse, Message, ToolCall, ToolDef, ToolResult

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Recorder helpers — fail-closed (exceptions propagate)
# ------------------------------------------------------------------


async def record_message(
    recorder: LoopRecorder | None,
    message: Message,
    iteration: int,
) -> None:
    """Record message to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_message_appended(message, iteration)


async def record_llm(
    recorder: LoopRecorder | None,
    response: LLMResponse,
    iteration: int,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
    duration_ms: int | None = None,
) -> None:
    """Record LLM completion to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_llm_call_completed(
        response,
        iteration,
        semantic_messages=semantic_messages,
        semantic_tools=semantic_tools,
        duration_ms=duration_ms,
    )


async def record_tool(
    recorder: LoopRecorder | None,
    tool_call: ToolCall,
    tool_result: ToolResult,
    iteration: int,
) -> None:
    """Record tool completion to authoritative persistence. Exceptions propagate."""
    if recorder is None:
        return
    await recorder.on_tool_completed(tool_call, tool_result, iteration)


# ------------------------------------------------------------------
# Observer helpers — best-effort (exceptions swallowed)
# ------------------------------------------------------------------


async def notify_message(
    observer: LoopObserver | None,
    message: Message,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify observer of a message append, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_message_appended(message, iteration)
    except Exception:
        logger.warning("Observer.on_message_appended failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_message_appended failed at iteration {iteration}")


async def notify_llm(
    observer: LoopObserver | None,
    response: LLMResponse,
    iteration: int,
    warnings: list[str] | None = None,
    *,
    semantic_messages: list[Message] | None = None,
    semantic_tools: list[ToolDef] | None = None,
    duration_ms: int | None = None,
) -> None:
    """Notify observer of an LLM call completion, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_llm_call_completed(
            response,
            iteration,
            semantic_messages=semantic_messages,
            semantic_tools=semantic_tools,
            duration_ms=duration_ms,
        )
    except Exception:
        logger.warning("Observer.on_llm_call_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_llm_call_completed failed at iteration {iteration}")


async def notify_tool(
    observer: LoopObserver | None,
    tool_call: ToolCall,
    tool_result: ToolResult,
    iteration: int,
    warnings: list[str] | None = None,
) -> None:
    """Notify observer of a tool completion, swallowing exceptions."""
    if observer is None:
        return
    try:
        await observer.on_tool_completed(tool_call, tool_result, iteration)
    except Exception:
        logger.warning("Observer.on_tool_completed failed", exc_info=True)
        if warnings is not None:
            warnings.append(f"on_tool_completed failed at iteration {iteration}")
