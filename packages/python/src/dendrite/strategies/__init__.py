"""Agent strategies — how the agent communicates with the LLM."""

from dendrite.strategies.base import Strategy
from dendrite.strategies.native import NativeToolCalling

__all__ = ["NativeToolCalling", "Strategy"]
