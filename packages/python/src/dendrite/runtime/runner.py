"""Agent runner — the entry point for executing agents.

Takes an Agent definition and runs it through the loop with a resolved
provider and strategy. This is the top-level API developers interact with.

Sprint 1: resolves provider from model string, uses NativeToolCalling
strategy and ReActLoop as defaults. Future sprints add provider registry,
strategy selection from agent config, and more loop types.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dendrite.loops.react import ReActLoop
from dendrite.strategies.native import NativeToolCalling

if TYPE_CHECKING:
    from dendrite.agent import Agent
    from dendrite.llm.base import LLMProvider
    from dendrite.loops.base import Loop
    from dendrite.strategies.base import Strategy
    from dendrite.types import RunResult


async def run(
    agent: Agent,
    *,
    provider: LLMProvider,
    input: str,
    strategy: Strategy | None = None,
    loop: Loop | None = None,
    **kwargs: Any,
) -> RunResult:
    """Run an agent to completion.

    This is the primary API for executing a Dendrite agent. It wires
    together the agent definition, LLM provider, strategy, and loop,
    then executes the loop until completion.

    Args:
        agent: Agent definition (model, tools, prompt, limits).
        provider: LLM provider to use for this run.
        input: The user's input to process.
        strategy: Communication strategy. Defaults to NativeToolCalling.
        loop: Execution loop. Defaults to ReActLoop.
        **kwargs: Reserved for future use (run config, callbacks, etc.).

    Returns:
        RunResult with status, answer, steps, and usage stats.

    Usage:
        from dendrite import Agent, tool, run
        from dendrite.llm import AnthropicProvider

        @tool()
        async def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(
            model="claude-sonnet-4-6",
            tools=[add],
            prompt="You are a calculator.",
        )
        provider = AnthropicProvider(api_key="sk-...", model="claude-sonnet-4-6")
        result = await run(agent, provider=provider, input="What is 15 + 27?")
        print(result.answer)
    """
    resolved_strategy = strategy or NativeToolCalling()
    resolved_loop = loop or ReActLoop()

    return await resolved_loop.run(
        agent=agent,
        provider=provider,
        strategy=resolved_strategy,
        user_input=input,
    )
