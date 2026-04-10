"""Command-line interface for BehaviorCI."""

import subprocess
from pathlib import Path

import typer

from .storage import get_storage, reset_all_storage

app = typer.Typer(
    help="BehaviorCI - Pytest-native behavioral regression testing for LLMs", no_args_is_help=True
)


def run_pytest(args: list[str]) -> None:
    """Helper to run pytest with BehaviorCI arguments."""
    cmd = ["pytest"] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(code=e.returncode)


@app.command()
def record(
    path: str = typer.Argument("tests/", help="Path to tests"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Record new behavioral snapshots (overwrites existing)."""
    typer.echo(f"Recording snapshots for {path}...")
    args = [path, "--behaviorci-record", "-v"]
    if db:
        args.extend(["--behaviorci-db", db])
    run_pytest(args)


@app.command()
def check(
    path: str = typer.Argument("tests/", help="Path to tests"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Check for behavioral regressions."""
    typer.echo(f"Checking for regressions in {path}...")
    args = [path, "--behaviorci", "-v"]
    if db:
        args.extend(["--behaviorci-db", db])
    run_pytest(args)


@app.command()
def update(
    path: str = typer.Argument("tests/", help="Path to tests"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Update failing snapshots with current output."""
    typer.echo(f"Updating failing snapshots in {path}...")
    args = [path, "--behaviorci-update", "-v"]
    if db:
        args.extend(["--behaviorci-db", db])
    run_pytest(args)


@app.command()
def record_missing(
    path: str = typer.Argument("tests/", help="Path to tests"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Record missing snapshots and check existing (CI workflow)."""
    typer.echo(f"Recording missing snapshots in {path}...")
    args = [path, "--behaviorci-record-missing", "-v"]
    if db:
        args.extend(["--behaviorci-db", db])
    run_pytest(args)


@app.command()
def stats(db: str = typer.Option(None, "--db", help="Custom database path")) -> None:
    """Show statistics about stored behaviors."""
    storage = get_storage(db)
    stats_dict = storage.get_stats()

    typer.echo("\n📊 BehaviorCI Statistics")
    typer.echo("=========================")
    typer.echo(f"Total Snapshots: {stats_dict['snapshots']}")
    typer.echo(f"Unique Behaviors: {stats_dict['behaviors']}")
    typer.echo(f"History Records: {stats_dict['history_records']}")
    typer.echo("")


@app.command()
def clear(
    force: bool = typer.Option(False, "--force", "-f", help="Force deletion without prompt"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Clear all stored snapshots (destructive)."""
    if not force:
        confirm = typer.confirm("Are you sure you want to delete ALL snapshots?")
        if not confirm:
            typer.echo("Operation cancelled.")
            raise typer.Exit()

    storage = get_storage(db)
    db_path = storage.db_path

    reset_all_storage()

    try:
        Path(db_path).unlink(missing_ok=True)
        typer.echo(f"Successfully deleted {db_path}")
    except Exception as e:
        typer.echo(f"Error deleting database: {e}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
