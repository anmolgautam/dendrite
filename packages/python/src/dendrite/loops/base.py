"""Loop protocol — how the agent iterates.

A loop orchestrates the cycle of calling the LLM, interpreting the response,
executing tools, and feeding results back. It uses a Strategy for LLM
communication and a Provider for actual LLM calls.

The loop never touches provider-specific APIs or prompt formatting — that's
the strategy's job. The loop is pure orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dendrite.agent import Agent
    from dendrite.llm.base import LLMProvider
    from dendrite.strategies.base import Strategy
    from dendrite.types import RunResult


class Loop(ABC):
    """Base class for agent execution loops.

    Subclasses implement the iteration pattern:
        ReActLoop      — think → act → observe → repeat
        SingleShot     — one LLM call, no tools (planned)
        PlanAndExecute — plan upfront, then execute steps (planned)
    """

    @abstractmethod
    async def run(
        self,
        *,
        agent: Agent,
        provider: LLMProvider,
        strategy: Strategy,
        user_input: str,
    ) -> RunResult:
        """Execute the agent loop until completion.

        Args:
            agent: Agent definition (tools, prompt, limits).
            provider: LLM provider to call.
            strategy: Strategy for message building and response parsing.
            user_input: The user's input to process.

        Returns:
            RunResult with status, answer, steps, and usage.
        """
