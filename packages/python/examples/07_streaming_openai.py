"""Streaming with OpenAI Chat Completions — tool calls via gpt-4o-mini.

Run with:
    OPENAI_API_KEY=sk-... python examples/07_streaming_openai.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.openai import OpenAIProvider
from dendrux.types import RunEventType

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


@tool()
async def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    data = {
        "London": {"temp": 12, "condition": "cloudy", "humidity": 78},
        "Tokyo": {"temp": 24, "condition": "sunny", "humidity": 45},
        "New York": {"temp": 18, "condition": "partly cloudy", "humidity": 62},
    }
    return data.get(city, {"temp": 20, "condition": "unknown", "humidity": 50})


@tool()
async def convert_temp(celsius: float, to_unit: str) -> str:
    """Convert a temperature from Celsius to another unit."""
    if to_unit.lower() == "fahrenheit":
        return f"{celsius * 9 / 5 + 32:.1f}°F"
    if to_unit.lower() == "kelvin":
        return f"{celsius + 273.15:.1f}K"
    return f"{celsius}°C"


async def main() -> None:
    async with Agent(
        provider=OpenAIProvider(model="gpt-4o-mini"),
        prompt=(
            "You are a weather assistant. When asked about weather, "
            "use get_weather to fetch data, then convert temperatures "
            "if the user asks for a specific unit. Be concise."
        ),
        tools=[get_weather, convert_temp],
    ) as agent:
        stream = agent.stream(
            "What's the weather in Tokyo and London? Give me temps in Fahrenheit."
        )
        print(f"Run ID: {stream.run_id}\n")

        async with stream:
            async for event in stream:
                if event.type == RunEventType.TEXT_DELTA:
                    print(event.text, end="", flush=True)

                elif event.type == RunEventType.TOOL_USE_START:
                    print(f"\n  >> calling {event.tool_name}...", end="", flush=True)

                elif event.type == RunEventType.TOOL_USE_END:
                    params = event.tool_call.params if event.tool_call else {}
                    print(f" {params}")

                elif event.type == RunEventType.TOOL_RESULT:
                    status = "ok" if event.tool_result.success else "error"
                    print(f"  << {event.tool_call.name} [{status}]: {event.tool_result.payload}")

                elif event.type == RunEventType.RUN_COMPLETED:
                    r = event.run_result
                    iters, toks = r.iteration_count, r.usage.total_tokens
                    print(f"\n\n--- {iters} iterations, {toks} tokens ---")

                elif event.type == RunEventType.RUN_ERROR:
                    print(f"\n[error] {event.error}")


if __name__ == "__main__":
    asyncio.run(main())
