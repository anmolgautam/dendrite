"""Streaming research agent — shows real-time event handling.

Same research flow as main.py but uses agent.stream() to print
token-by-token text and tool events as they happen.

Usage:
    cd examples/04_research_agent
    ANTHROPIC_API_KEY=sk-... FIRECRAWL_API_KEY=fc-... python stream_test.py "your topic"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from agents.scrape_agent import run_scrape
from agents.search_agent import run_search
from dotenv import load_dotenv

from dendrux import Agent, tool
from dendrux.llm.anthropic import AnthropicProvider
from dendrux.types import RunEventType

load_dotenv()

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


@tool(max_calls_per_run=2, timeout_seconds=120)
async def research_topic(query: str) -> str:
    """Search the web for a query and get summarized findings."""
    print(f"\n  [search] {query}")
    return await run_search(query)


@tool(max_calls_per_run=1, timeout_seconds=120)
async def deep_read(url: str) -> str:
    """Read and summarize a specific web page in depth."""
    print(f"\n  [deep_read] {url}")
    return await run_scrape(url)


@tool()
async def save_report(filename: str, content: str) -> str:
    """Save the final research report as a markdown file."""
    filepath = OUTPUT_DIR / f"{filename}.md"
    filepath.write_text(content, encoding="utf-8")
    return f"Report saved to {filepath}"


PROMPT = """\
You are a research assistant. Produce a short research summary on the given topic.

Budget: 2 searches, 1 deep read. Be efficient.

Workflow:
1. Use research_topic for 1-2 focused queries
2. Optionally deep_read one high-value source
3. Save a concise markdown report via save_report

Keep the report under 500 words. Cite sources.
"""


async def main() -> None:
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not topic:
        print('Usage: python stream_test.py "your topic"')
        sys.exit(1)

    print(f"Researching: {topic}")
    print(f"{'=' * 60}\n")

    agent = Agent(
        name="StreamTest",
        provider=AnthropicProvider(model="claude-sonnet-4-6", timeout=120),
        prompt=PROMPT,
        tools=[research_topic, deep_read, save_report],
        max_iterations=10,
    )

    stream = agent.stream(f"Research: {topic}")
    print(f"Run ID: {stream.run_id}\n")

    async with stream:
        async for event in stream:
            if event.type == RunEventType.RUN_STARTED:
                print(f"[started] run_id={event.run_id}")

            elif event.type == RunEventType.TEXT_DELTA:
                print(event.text, end="", flush=True)

            elif event.type == RunEventType.TOOL_USE_START:
                print(f"\n[tool_start] {event.tool_name}")

            elif event.type == RunEventType.TOOL_USE_END:
                print(f"[tool_end] {event.tool_name}")

            elif event.type == RunEventType.TOOL_RESULT:
                status = "ok" if event.tool_result.success else "error"
                print(f"[tool_result] {event.tool_call.name} -> {status}")

            elif event.type == RunEventType.RUN_COMPLETED:
                r = event.run_result
                print(f"\n\n{'=' * 60}")
                print(f"[completed] status={r.status.value}")
                print(f"  iterations: {r.iteration_count}")
                print(f"  tokens: {r.usage.input_tokens} in / {r.usage.output_tokens} out")
                if r.answer:
                    print(f"  answer length: {len(r.answer)} chars")

            elif event.type == RunEventType.RUN_ERROR:
                print(f"\n[error] {event.error}")

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
