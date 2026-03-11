"""Dendrite CLI entry point."""

import typer

app = typer.Typer(
    name="dendrite",
    help="Dendrite — The runtime for agents that act in the real world.",
    no_args_is_help=True,
)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", "-v", help="Show version and exit."),
) -> None:
    """Dendrite — The runtime for agents that act in the real world."""
    if version:
        from dendrite import __version__

        typer.echo(f"dendrite {__version__}")
        raise typer.Exit()
