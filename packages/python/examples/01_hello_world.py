"""Hello World — minimal Dendrite agent with a tool.

Run with:
    dendrite run examples/01_hello_world.py -i "What is 15 + 27?"

Or programmatically:
    ANTHROPIC_API_KEY=sk-... python examples/01_hello_world.py
"""

from dendrite import Agent, tool


@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


agent = Agent(
    model="claude-sonnet-4-6",
    prompt="You are a helpful calculator. Use the add tool when asked to add numbers.",
    tools=[add],
)

if __name__ == "__main__":
    import asyncio
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    from dendrite import run
    from dendrite.llm.anthropic import AnthropicProvider

    # Load .env from repo root (three levels up: examples/ → python/ → packages/ → dendrite/)
    load_dotenv(Path(__file__).resolve().parents[3] / ".env")

    async def main() -> None:
        provider = AnthropicProvider(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            model=agent.model,
        )
        result = await run(agent, provider=provider, user_input="What is 15 + 27?")
        print(f"Answer: {result.answer}")
        print(f"Steps: {result.iteration_count}, Tokens: {result.usage.total_tokens}")

    asyncio.run(main())
