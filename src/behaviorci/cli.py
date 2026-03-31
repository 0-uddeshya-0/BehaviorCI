"""CLI for BehaviorCI - thin wrapper around pytest."""

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from .storage import get_storage

app = typer.Typer(
    name="behaviorci",
    help="Pytest-native behavioral regression testing for LLM applications",
    no_args_is_help=True,
)


def _run_pytest(
    behaviorci_args: list,
    pytest_args: Optional[list] = None,
    directory: Optional[str] = None
) -> int:
    """Run pytest with BehaviorCI options."""
    cmd = ["pytest"] + behaviorci_args + (pytest_args or [])

    if directory:
        cmd.append(directory)

    typer.echo(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        typer.echo("Error: pytest not found. Install with: pip install pytest", err=True)
        return 1
    except KeyboardInterrupt:
        typer.echo("\nInterrupted.")
        return 130


@app.command()
def record(
    directory: Annotated[Optional[str], typer.Argument(help="Test directory")] = None,
    pytest_args: Annotated[Optional[list[str]], typer.Argument(help="Additional pytest arguments")] = None,
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
    # TASK 5 (v0.2): Add --model flag for custom embedding models
    model: Annotated[Optional[str], typer.Option("--model", help="Embedding model name (e.g., sentence-transformers/all-mpnet-base-v2)")] = None,
):
    """Record new behavioral snapshots (creates or overwrites)."""
    behaviorci_args = ["--behaviorci-record"]
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    if model:
        behaviorci_args.extend(["--behaviorci-model", model])
    sys.exit(_run_pytest(behaviorci_args, pytest_args, directory))


@app.command()
def check(
    directory: Annotated[Optional[str], typer.Argument(help="Test directory")] = None,
    pytest_args: Annotated[Optional[list[str]], typer.Argument(help="Additional pytest arguments")] = None,
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
    # TASK 5 (v0.2): Add --model flag for custom embedding models
    model: Annotated[Optional[str], typer.Option("--model", help="Embedding model name (e.g., sentence-transformers/all-mpnet-base-v2)")] = None,
):
    """Run behavioral regression tests (fails on behavior change)."""
    behaviorci_args = ["--behaviorci"]
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    if model:
        behaviorci_args.extend(["--behaviorci-model", model])
    sys.exit(_run_pytest(behaviorci_args, pytest_args, directory))


@app.command()
def update(
    directory: Annotated[Optional[str], typer.Argument(help="Test directory")] = None,
    pytest_args: Annotated[Optional[list[str]], typer.Argument(help="Additional pytest arguments")] = None,
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
    # TASK 5 (v0.2): Add --model flag for custom embedding models
    model: Annotated[Optional[str], typer.Option("--model", help="Embedding model name (e.g., sentence-transformers/all-mpnet-base-v2)")] = None,
):
    """Update failing snapshots (accept new behavior)."""
    behaviorci_args = ["--behaviorci-update"]
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    if model:
        behaviorci_args.extend(["--behaviorci-model", model])
    sys.exit(_run_pytest(behaviorci_args, pytest_args, directory))


@app.command()
def record_missing(
    directory: Annotated[Optional[str], typer.Argument(help="Test directory")] = None,
    pytest_args: Annotated[Optional[list[str]], typer.Argument(help="Additional pytest arguments")] = None,
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
    # TASK 5 (v0.2): Add --model flag for custom embedding models
    model: Annotated[Optional[str], typer.Option("--model", help="Embedding model name (e.g., sentence-transformers/all-mpnet-base-v2)")] = None,
):
    """Record missing snapshots (CI workflow - checks existing, records new)."""
    behaviorci_args = ["--behaviorci-record-missing"]
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    if model:
        behaviorci_args.extend(["--behaviorci-model", model])
    sys.exit(_run_pytest(behaviorci_args, pytest_args, directory))


@app.command()
def stats(
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
):
    """Show database statistics."""
    storage = get_storage(db_path)
    stats_data = storage.get_stats()

    typer.echo("BehaviorCI Database Statistics")
    typer.echo("=" * 30)
    typer.echo(f"Total snapshots: {stats_data['snapshots']}")
    typer.echo(f"Unique behaviors: {stats_data['behaviors']}")
    typer.echo(f"History records: {stats_data['history_records']}")

    if stats_data['snapshots'] > 0:
        typer.echo("\nSnapshots per behavior:")
        # HIGH-002 FIX: Use Storage method instead of raw SQLite
        summary = storage.get_behavior_summary()
        for behavior_id, count, last_run in summary:
            typer.echo(f"  {behavior_id}: {count}")


@app.command()
def clear(
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
    force: Annotated[bool, typer.Option("--force", help="Skip confirmation")] = False,
):
    """Clear all snapshots (USE WITH CAUTION)."""
    storage = get_storage(db_path)
    stats_data = storage.get_stats()

    if stats_data['snapshots'] == 0:
        typer.echo("Database is already empty.")
        return

    if not force:
        confirm = typer.confirm(
            f"Delete {stats_data['snapshots']} snapshots and "
            f"{stats_data['history_records']} history records?"
        )
        if not confirm:
            typer.echo("Aborted.")
            return

    storage.clear_all()
    typer.echo("Database cleared.")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
