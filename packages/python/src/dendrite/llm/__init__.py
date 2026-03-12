"""LLM provider abstraction layer."""

from dendrite.llm.base import LLMProvider
from dendrite.llm.mock import MockLLM

__all__ = ["LLMProvider", "MockLLM"]
