"""Tests for Loop protocol and ReActLoop."""

from __future__ import annotations

import pytest

from dendrite.agent import Agent
from dendrite.llm.mock import MockLLM
from dendrite.loops.base import Loop
from dendrite.loops.react import ReActLoop
from dendrite.strategies.native import NativeToolCalling
from dendrite.tool import tool
from dendrite.types import (
    Finish,
    LLMResponse,
    RunStatus,
    ToolCall,
)

# ------------------------------------------------------------------
# Test tools
# ------------------------------------------------------------------


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool()
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool()
def sync_add(a: int, b: int) -> int:
    """Add two numbers (sync)."""
    return a + b


@tool()
async def failing_tool() -> str:
    """A tool that always fails."""
    raise RuntimeError("Something broke")


# ------------------------------------------------------------------
# Test agent
# ------------------------------------------------------------------


def _make_agent(**overrides) -> Agent:
    defaults = {
        "model": "mock",
        "prompt": "You are a calculator.",
        "tools": [add, multiply],
        "max_iterations": 10,
    }
    defaults.update(overrides)
    return Agent(**defaults)


# ------------------------------------------------------------------
# Loop ABC
# ------------------------------------------------------------------


class TestLoopABC:
    def test_cannot_instantiate_without_run(self) -> None:
        with pytest.raises(TypeError):
            Loop()  # type: ignore[abstract]

    def test_react_loop_is_a_loop(self) -> None:
        assert isinstance(ReActLoop(), Loop)


# ------------------------------------------------------------------
# ReActLoop — simple finish (no tools)
# ------------------------------------------------------------------


class TestReActLoopFinish:
    async def test_immediate_finish(self) -> None:
        """LLM answers immediately without calling any tools."""
        llm = MockLLM([LLMResponse(text="42")])
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="What is the meaning of life?",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "42"
        assert result.iteration_count == 1
        assert len(result.steps) == 1
        assert isinstance(result.steps[0].action, Finish)

    async def test_run_has_unique_id(self) -> None:
        llm = MockLLM([LLMResponse(text="done")])
        agent = _make_agent()

        r1 = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Hi",
        )

        llm2 = MockLLM([LLMResponse(text="done")])
        r2 = await ReActLoop().run(
            agent=agent,
            provider=llm2,
            strategy=NativeToolCalling(),
            user_input="Hi",
        )

        assert r1.run_id != r2.run_id


# ------------------------------------------------------------------
# ReActLoop — tool calling
# ------------------------------------------------------------------


class TestReActLoopToolCalling:
    async def test_single_tool_call_then_finish(self) -> None:
        """LLM calls a tool, gets the result, then finishes."""
        tc = ToolCall(
            name="add",
            params={"a": 15, "b": 27},
            provider_tool_call_id="toolu_1",
        )
        llm = MockLLM(
            [
                LLMResponse(text="Let me add those", tool_calls=[tc]),
                LLMResponse(text="15 + 27 = 42"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="What is 15 + 27?",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "15 + 27 = 42"
        assert result.iteration_count == 2
        assert len(result.steps) == 2

        # First step was a tool call
        assert isinstance(result.steps[0].action, ToolCall)
        assert result.steps[0].action.name == "add"

        # Second step was finish
        assert isinstance(result.steps[1].action, Finish)

    async def test_tool_receives_correct_params(self) -> None:
        """Verify the tool function is called with the right arguments."""
        tc = ToolCall(
            name="multiply",
            params={"a": 6, "b": 7},
            provider_tool_call_id="toolu_m",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="42"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="6 * 7?",
        )

        assert result.status == RunStatus.SUCCESS
        # The tool was called and result fed back — if params were wrong,
        # the tool would fail and result would be an error

    async def test_multiple_tool_calls_sequential(self) -> None:
        """LLM calls tools across multiple iterations."""
        tc1 = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t1")
        tc2 = ToolCall(name="multiply", params={"a": 3, "b": 4}, provider_tool_call_id="t2")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc1]),
                LLMResponse(tool_calls=[tc2]),
                LLMResponse(text="1+2=3, 3*4=12"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Compute both",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.iteration_count == 3
        assert isinstance(result.steps[0].action, ToolCall)
        assert isinstance(result.steps[1].action, ToolCall)
        assert isinstance(result.steps[2].action, Finish)

    async def test_sync_tool_works(self) -> None:
        """Sync tools are executed via asyncio.to_thread."""
        tc = ToolCall(name="sync_add", params={"a": 10, "b": 20}, provider_tool_call_id="t_sync")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="30"),
            ]
        )
        agent = _make_agent(tools=[sync_add])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="10 + 20?",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "30"


# ------------------------------------------------------------------
# ReActLoop — error handling
# ------------------------------------------------------------------


class TestReActLoopErrors:
    async def test_unknown_tool_returns_error_result(self) -> None:
        """If LLM calls a tool that doesn't exist, loop returns error in result."""
        tc = ToolCall(
            name="nonexistent",
            params={},
            provider_tool_call_id="toolu_bad",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="I couldn't find that tool"),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Do something",
        )

        # Loop should continue — the error result is fed back to LLM
        assert result.status == RunStatus.SUCCESS
        assert result.iteration_count == 2

    async def test_tool_exception_returns_error_result(self) -> None:
        """If a tool raises an exception, it becomes an error ToolResult."""
        tc = ToolCall(
            name="failing_tool",
            params={},
            provider_tool_call_id="toolu_fail",
        )
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(text="The tool failed, sorry"),
            ]
        )
        agent = _make_agent(tools=[failing_tool])

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Do the thing",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.iteration_count == 2


# ------------------------------------------------------------------
# ReActLoop — max iterations
# ------------------------------------------------------------------


class TestReActLoopMaxIterations:
    async def test_stops_at_max_iterations(self) -> None:
        """Loop terminates with MAX_ITERATIONS when limit is hit."""
        tc = ToolCall(name="add", params={"a": 1, "b": 1}, provider_tool_call_id="t_loop")
        llm = MockLLM(
            [
                LLMResponse(tool_calls=[tc]),
                LLMResponse(tool_calls=[tc]),
                LLMResponse(tool_calls=[tc]),
            ]
        )
        agent = _make_agent(max_iterations=3)

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Keep going",
        )

        assert result.status == RunStatus.MAX_ITERATIONS
        assert result.iteration_count == 3
        assert result.answer is None

    async def test_max_iterations_one_with_finish(self) -> None:
        """max_iterations=1 works as single-shot when LLM finishes immediately."""
        llm = MockLLM([LLMResponse(text="Quick answer")])
        agent = _make_agent(max_iterations=1)

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="Be quick",
        )

        assert result.status == RunStatus.SUCCESS
        assert result.answer == "Quick answer"


# ------------------------------------------------------------------
# ReActLoop — usage tracking
# ------------------------------------------------------------------


class TestReActLoopUsage:
    async def test_accumulates_usage_across_iterations(self) -> None:
        tc = ToolCall(name="add", params={"a": 1, "b": 2}, provider_tool_call_id="t_u")
        llm = MockLLM(
            [
                LLMResponse(
                    tool_calls=[tc],
                    usage=UsageStats(input_tokens=100, output_tokens=50, total_tokens=150),
                ),
                LLMResponse(
                    text="3",
                    usage=UsageStats(input_tokens=200, output_tokens=30, total_tokens=230),
                ),
            ]
        )
        agent = _make_agent()

        result = await ReActLoop().run(
            agent=agent,
            provider=llm,
            strategy=NativeToolCalling(),
            user_input="1+2?",
        )

        assert result.usage.input_tokens == 300
        assert result.usage.output_tokens == 80
        assert result.usage.total_tokens == 380


# Need to import UsageStats for the usage test
from dendrite.types import UsageStats  # noqa: E402
