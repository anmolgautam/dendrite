# Dendrite

> The runtime for agents that act in the real world.

Dendrite is a Python framework for building AI agents with tool calling, state persistence, and full observability. Define an agent, give it tools, run it — every step is traced and stored.

**What's working today:** Agent loop, ReAct reasoning, tool calling (sync + async), Anthropic Claude integration, SQLite/Postgres persistence, CLI for running agents and inspecting traces, token usage tracking.

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/user/dendrite.git
cd dendrite/packages/python
pip install -e ".[dev,db,anthropic]"
```

### 2. Set up your API key

```bash
cp ../../.env.example ../../.env
# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Run the hello world example

```bash
# Via CLI
dendrite run examples/01_hello_world.py -i "What is 15 + 27?"

# Or directly
ANTHROPIC_API_KEY=sk-ant-... python examples/01_hello_world.py
```

### 4. Try the persistent agent (traces + DB)

```bash
ANTHROPIC_API_KEY=sk-ant-... python examples/02_persistent_agent.py

# Then inspect what happened:
dendrite runs                          # List all runs
dendrite traces <run_id> --tools       # See the full reasoning trace
```

## How It Works

```python
from dendrite import Agent, tool, run
from dendrite.llm.anthropic import AnthropicProvider

# 1. Define tools — plain Python functions
@tool()
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

# 2. Define an agent
agent = Agent(
    model="claude-sonnet-4-6",
    prompt="You are a helpful calculator.",
    tools=[add],
)

# 3. Run it
provider = AnthropicProvider(api_key="sk-ant-...", model="claude-sonnet-4-6")
result = await run(agent, provider=provider, user_input="What is 15 + 27?")
print(result.answer)  # "42"
```

### With persistence (traces saved to SQLite)

```python
from dendrite.db.session import get_engine
from dendrite.runtime.state import SQLAlchemyStateStore

engine = await get_engine()  # Auto-creates ./dendrite.db
store = SQLAlchemyStateStore(engine)

result = await run(
    agent,
    provider=provider,
    user_input="What is 15 + 27?",
    state_store=store,  # Every step is now persisted
)

# Inspect later:
# dendrite runs
# dendrite traces <run_id> --tools
```

### With redaction (scrub secrets before storing)

```python
result = await run(
    agent,
    provider=provider,
    user_input="My password is hunter2",
    state_store=store,
    redact=lambda s: s.replace("hunter2", "***"),  # Applied to all persisted data
)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `dendrite run <file> -i "prompt"` | Run an agent from a Python file |
| `dendrite runs` | List recent agent runs |
| `dendrite traces <run_id>` | Show the reasoning trace for a run |
| `dendrite traces <run_id> --tools` | Include tool call details |
| `dendrite db migrate` | Run database migrations (Postgres) |
| `dendrite db status` | Show current migration revision |
| `dendrite --version` | Show version |

## Project Structure

```
dendrite/
├── packages/python/
│   ├── src/dendrite/          # Source code
│   │   ├── agent.py           # Agent definition
│   │   ├── tool.py            # @tool decorator + schema generation
│   │   ├── types.py           # Core types (Message, ToolCall, RunResult, etc.)
│   │   ├── llm/               # LLM providers (Anthropic, Mock)
│   │   ├── loops/             # Execution loops (ReAct)
│   │   ├── strategies/        # Tool calling strategies (NativeToolCalling)
│   │   ├── runtime/           # Runner, observer, state store
│   │   ├── db/                # SQLAlchemy models, migrations
│   │   └── cli/               # CLI commands
│   ├── tests/                 # 258 tests (95%+ coverage)
│   └── examples/              # Working examples
├── .env.example               # API key template
└── Makefile                   # Dev commands
```

## Requirements

- Python 3.11+
- An Anthropic API key (for running agents with Claude)

Tests run without an API key — they use `MockLLM` for deterministic testing.

## Development

```bash
cd packages/python

# Install with all dev dependencies
pip install -e ".[dev,db,anthropic]"

# Run all checks (lint + typecheck + tests)
make ci

# Auto-fix formatting
make format

# Run tests only
make test
```

| Command | What it does |
|---------|-------------|
| `make ci` | Lint (ruff) + typecheck (mypy) + tests (pytest) — **run before every commit** |
| `make test` | Tests only |
| `make format` | Auto-fix formatting and lint issues |
| `make lint` | Check lint + formatting without fixing |
| `make typecheck` | Check type annotations |

## Current Status (v0.1.0a1)

This is an early alpha. What's shipped vs what's planned:

| Feature | Status |
|---------|--------|
| Agent loop + ReAct reasoning | Shipped |
| Tool calling (sync + async, timeouts) | Shipped |
| Anthropic Claude provider | Shipped |
| SQLite + Postgres persistence | Shipped |
| CLI (run, traces, runs, db) | Shipped |
| Token usage tracking | Shipped |
| Opt-in trace redaction | Shipped |
| Tool sandbox / isolation | Planned (Sprint 6) |
| Client tool bridge (pause/resume) | Planned (Sprint 3) |
| Worker pool / crash recovery | Planned (Sprint 4) |
| OpenAI + multi-provider support | Planned (Sprint 6) |
| TypeScript client SDK | Planned (Sprint 5) |

## Built with Claude Code

Dendrite is built through AI pair programming using [Claude Code](https://claude.ai/code). Every commit is co-authored, every design decision discussed before implementation. We use a 5-layer doc architecture to keep the LLM aligned across conversations — vision docs describe the destination, the status map describes reality.

## License

[Apache 2.0](LICENSE)
