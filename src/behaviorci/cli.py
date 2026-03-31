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
    """Run pytest with BehaviorCI options.
    
    Args:
        behaviorci_args: BehaviorCI-specific pytest arguments
        pytest_args: Additional pytest arguments
        directory: Directory to run tests from
        
    Returns:
        Exit code from pytest
    """
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
):
    """Record new behavioral snapshots (creates or overwrites)."""
    behaviorci_args = ["--behaviorci-record"]
    
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    
    exit_code = _run_pytest(behaviorci_args, pytest_args, directory)
    sys.exit(exit_code)


@app.command()
def check(
    directory: Annotated[Optional[str], typer.Argument(help="Test directory")] = None,
    pytest_args: Annotated[Optional[list[str]], typer.Argument(help="Additional pytest arguments")] = None,
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
):
    """Run behavioral regression tests (fails on behavior change)."""
    behaviorci_args = ["--behaviorci"]
    
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    
    exit_code = _run_pytest(behaviorci_args, pytest_args, directory)
    sys.exit(exit_code)


@app.command()
def update(
    directory: Annotated[Optional[str], typer.Argument(help="Test directory")] = None,
    pytest_args: Annotated[Optional[list[str]], typer.Argument(help="Additional pytest arguments")] = None,
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
):
    """Update failing snapshots (accept new behavior)."""
    behaviorci_args = ["--behaviorci-update"]
    
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    
    exit_code = _run_pytest(behaviorci_args, pytest_args, directory)
    sys.exit(exit_code)


@app.command()
def record_missing(
    directory: Annotated[Optional[str], typer.Argument(help="Test directory")] = None,
    pytest_args: Annotated[Optional[list[str]], typer.Argument(help="Additional pytest arguments")] = None,
    db_path: Annotated[Optional[str], typer.Option("--db", help="Path to BehaviorCI database")] = None,
):
    """Record missing snapshots (CI workflow - checks existing, records new)."""
    behaviorci_args = ["--behaviorci-record-missing"]
    
    if db_path:
        behaviorci_args.extend(["--behaviorci-db", db_path])
    
    exit_code = _run_pytest(behaviorci_args, pytest_args, directory)
    sys.exit(exit_code)


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
        # Get all behaviors
        import sqlite3
        conn = sqlite3.connect(str(storage.db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT behavior_id, COUNT(*) as count FROM snapshots GROUP BY behavior_id"
        ).fetchall()
        for row in rows:
            typer.echo(f"  {row['behavior_id']}: {row['count']}")


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
            f"Delete {stats_data['snapshots']} snapshots and {stats_data['history_records']} history records?"
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