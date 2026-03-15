"""Dendrite CLI entry point."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

app = typer.Typer(
    name="dendrite",
    help="Dendrite — The runtime for agents that act in the real world.",
    no_args_is_help=True,
)
console = Console()


def _register_subcommands() -> None:
    """Register subcommand groups. Called after app is created."""
    from dendrite.cli.db import app as db_app
    from dendrite.cli.runs import app as runs_app
    from dendrite.cli.traces import app as traces_app

    app.add_typer(db_app)
    app.add_typer(runs_app)
    app.add_typer(traces_app)


_register_subcommands()


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """Dendrite — The runtime for agents that act in the real world."""
    if version:
        from dendrite import __version__

        typer.echo(f"dendrite {__version__}")
        raise typer.Exit()


@app.command()
def run(
    file: str = typer.Argument(..., help="Path to a Python file defining an Agent."),
    input: str = typer.Option(..., "--input", "-i", help="User input to send to the agent."),
    agent_name: str = typer.Option(
        "", "--agent", "-a", help="Agent class/variable name. Auto-detected if omitted."
    ),
) -> None:
    """Run an agent from a Python file.

    This command executes the specified Python file to find and run an Agent.
    Only run files you trust — the file runs with full host privileges.
    """
    path = Path(file).resolve()
    if not path.exists():
        console.print(f"[red]File not found:[/red] {file}")
        raise typer.Exit(1)
    if path.suffix != ".py":
        console.print(f"[red]Only .py files are supported.[/red] Got: {path.suffix or '(none)'}")
        raise typer.Exit(1)

    module = _load_module(path)

    # Find the Agent in the module
    from dendrite.agent import Agent

    agent = _find_agent(module, Agent, agent_name)
    if agent is None:
        console.print(f"[red]No Agent found in {file}.[/red] Define one or use --agent.")
        raise typer.Exit(1)

    # Resolve the provider
    provider = _resolve_provider(agent.model)

    console.print(f"[bold]\\[Agent][/bold] Starting {agent.name}...")
    console.print()

    # Run the agent
    from dendrite.runtime.runner import run as agent_run

    result = asyncio.run(agent_run(agent, provider=provider, user_input=input))

    # Print results
    for i, step in enumerate(result.steps, 1):
        from dendrite.types import Finish, ToolCall

        if isinstance(step.action, ToolCall):
            params_str = ", ".join(f"{k}={v!r}" for k, v in step.action.params.items())
            console.print(f"[dim]\\[Step {i}][/dim] Calling tool: {step.action.name}({params_str})")
        elif isinstance(step.action, Finish):
            console.print(f'[dim]\\[Step {i}][/dim] Agent finished: "{step.action.answer}"')

    console.print()

    from dendrite.types import RunStatus

    status_display = {
        RunStatus.SUCCESS: ("[green]✓[/green]", "Completed"),
        RunStatus.MAX_ITERATIONS: ("[yellow]⚠[/yellow]", "Stopped (max iterations)"),
        RunStatus.WAITING_HUMAN_INPUT: ("[blue]⏸[/blue]", "Waiting for input"),
        RunStatus.ERROR: ("[red]✗[/red]", "Failed"),
    }
    icon, label = status_display.get(result.status, ("·", result.status.value))
    console.print(
        f"{icon} {label} in {result.iteration_count} iterations, {result.usage.total_tokens} tokens"
    )


def _load_module(path: Path) -> Any:
    """Load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location("__agent_module__", path)
    if spec is None or spec.loader is None:
        console.print(f"[red]Cannot load module from:[/red] {path}")
        raise typer.Exit(1)

    # Temporarily add parent directory to sys.path so relative imports work
    parent = str(path.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if added and parent in sys.path:
            sys.path.remove(parent)


def _find_agent(module: Any, agent_class: type, name: str) -> Any:
    """Find an Agent instance or subclass in a module."""
    # If name specified, look for it directly
    if name:
        obj = getattr(module, name, None)
        if obj is None:
            return None
        if isinstance(obj, agent_class):
            return obj
        if isinstance(obj, type) and issubclass(obj, agent_class) and obj is not agent_class:
            return obj()
        return None

    # Auto-detect: look for instances first, then subclasses
    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, agent_class):
            return obj

    for attr_name in dir(module):
        obj = getattr(module, attr_name)
        if isinstance(obj, type) and issubclass(obj, agent_class) and obj is not agent_class:
            return obj()

    return None


def _resolve_provider(model: str) -> Any:
    """Resolve an LLM provider from the model string."""
    import os

    # Sprint 1: Anthropic only. Future: parse model string for provider routing.
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        console.print(
            "[red]No API key found.[/red] Set the ANTHROPIC_API_KEY environment variable."
        )
        raise typer.Exit(1)

    try:
        from dendrite.llm.anthropic import AnthropicProvider
    except ImportError:
        console.print(
            "[red]Anthropic SDK not installed.[/red] Run: pip install dendrite[anthropic]"
        )
        raise typer.Exit(1) from None

    return AnthropicProvider(api_key=key, model=model)
