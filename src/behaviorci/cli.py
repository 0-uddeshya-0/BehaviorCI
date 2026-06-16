"""Command-line interface for BehaviorCI."""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer

from .storage import get_storage, reset_all_storage

app = typer.Typer(
    help="BehaviorCI - Pytest-native behavioral regression testing for LLMs",
    no_args_is_help=True,
)


def run_pytest(args: List[str]) -> None:
    """Run pytest with the given arguments, surfacing its exit code."""
    cmd = ["pytest"] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise typer.Exit(code=e.returncode)


def _format_time(timestamp: Optional[int]) -> str:
    if not timestamp:
        return "-"
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def _bar(value: float, width: int = 24) -> str:
    """Render a similarity score (0-1) as a simple ASCII meter."""
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


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
    typer.echo(f"Updating snapshots in {path}...")
    args = [path, "--behaviorci-update", "-v"]
    if db:
        args.extend(["--behaviorci-db", db])
    run_pytest(args)


@app.command()
def record_missing(
    path: str = typer.Argument("tests/", help="Path to tests"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Record missing snapshots and check existing ones (CI workflow)."""
    typer.echo(f"Recording missing snapshots in {path}...")
    args = [path, "--behaviorci-record-missing", "-v"]
    if db:
        args.extend(["--behaviorci-db", db])
    run_pytest(args)


@app.command()
def stats(db: str = typer.Option(None, "--db", help="Custom database path")) -> None:
    """Show statistics about stored behaviors."""
    storage = get_storage(db)
    totals = storage.get_stats()

    typer.echo("")
    typer.echo("BehaviorCI Statistics")
    typer.echo("=====================")
    typer.echo(f"Total Snapshots:  {totals['snapshots']}")
    typer.echo(f"Unique Behaviors: {totals['behaviors']}")
    typer.echo(f"History Records:  {totals['history_records']}")

    summary = storage.get_behavior_summary()
    if summary:
        typer.echo("")
        typer.echo(f"{'Behavior':<32}{'Snapshots':>10}   Last recorded")
        typer.echo("-" * 70)
        for behavior_id, count, last_run in summary:
            typer.echo(f"{behavior_id:<32}{count:>10}   {_format_time(last_run)}")
    typer.echo("")


@app.command()
def history(
    behavior_id: str = typer.Argument(..., help="Behavior ID to inspect"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max records to show per snapshot"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Show recent similarity history for a behavior (drift over time)."""
    storage = get_storage(db)
    snapshots = storage.get_all_snapshots_for_behavior(behavior_id)

    if not snapshots:
        typer.echo(f"No snapshots found for behavior '{behavior_id}'.")
        raise typer.Exit(code=1)

    for snapshot in snapshots:
        typer.echo("")
        typer.echo(f"Behavior: {behavior_id}   (snapshot {snapshot.id[:12]})")
        typer.echo(f"Input:    {snapshot.input_json}")
        rows = storage.get_similarity_history_with_timestamps(snapshot.id, limit=limit)
        if not rows:
            typer.echo("  No comparison history yet - run `behaviorci check`.")
            continue
        for similarity, timestamp in rows:
            typer.echo(f"  {_format_time(timestamp)}   {similarity:6.4f}  {_bar(similarity)}")
    typer.echo("")


@app.command()
def clear(
    force: bool = typer.Option(False, "--force", "-f", help="Skip the confirmation prompt"),
    db: str = typer.Option(None, "--db", help="Custom database path"),
) -> None:
    """Clear all stored snapshots (destructive)."""
    if not force:
        if not typer.confirm("Are you sure you want to delete ALL snapshots?"):
            typer.echo("Operation cancelled.")
            raise typer.Exit()

    storage = get_storage(db)
    db_path = Path(storage.db_path)

    reset_all_storage()

    try:
        for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
            path.unlink(missing_ok=True)
        typer.echo(f"Deleted {db_path}")
    except OSError as e:
        typer.echo(f"Error deleting database: {e}", err=True)
        raise typer.Exit(code=1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
