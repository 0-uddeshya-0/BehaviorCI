"""pytest integration for BehaviorCI.

The plugin runs each ``@behavior`` test the normal way and then, in
``pytest_runtest_makereport``, reads the value the decorator stashed on the
function and either records a baseline or compares against the stored one.
Doing the work in ``makereport`` (rather than a custom call hook) keeps the
test executing exactly once.
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest

from .comparator import Comparator
from .embedder import DEFAULT_MODEL_NAME, get_embedder
from .exceptions import BehaviorCIError, ConfigurationError
from .storage import get_storage

# Stash key used to pass per-item config from collection to the report hook.
CONFIG_KEY = pytest.StashKey[dict]()


def pytest_addoption(parser: Any) -> None:
    """Register BehaviorCI's command-line options."""
    group = parser.getgroup("behaviorci", "Behavioral regression testing for LLMs")
    group.addoption(
        "--behaviorci",
        action="store_true",
        default=False,
        help="Enable BehaviorCI regression testing",
    )
    group.addoption(
        "--behaviorci-record",
        action="store_true",
        default=False,
        help="Record new snapshots (overwrites existing)",
    )
    group.addoption(
        "--behaviorci-update",
        action="store_true",
        default=False,
        help="Update failing snapshots",
    )
    group.addoption(
        "--behaviorci-record-missing",
        action="store_true",
        default=False,
        help="Record snapshots that don't exist yet instead of failing (CI workflow)",
    )
    group.addoption(
        "--behaviorci-db",
        action="store",
        default=None,
        help="Path to the BehaviorCI database (default: .behaviorci/behaviorci.db)",
    )
    group.addoption(
        "--behaviorci-model",
        action="store",
        default=DEFAULT_MODEL_NAME,
        help=f"Embedding model name (default: {DEFAULT_MODEL_NAME})",
    )
    group.addoption(
        "--behaviorci-report",
        action="store",
        default=None,
        metavar="PATH",
        help="Write a machine-readable JSON report of results to PATH (for CI/automation)",
    )


def pytest_configure(config: Any) -> None:
    """Resolve the chosen options into flags on the pytest config object."""
    config.behaviorci_enabled = (
        config.getoption("--behaviorci")
        or config.getoption("--behaviorci-record")
        or config.getoption("--behaviorci-update")
        or config.getoption("--behaviorci-record-missing")
    )
    config.behaviorci_record = config.getoption("--behaviorci-record")
    config.behaviorci_update = config.getoption("--behaviorci-update")
    config.behaviorci_record_missing = config.getoption("--behaviorci-record-missing")
    config.behaviorci_db_path = config.getoption("--behaviorci-db")
    config.behaviorci_model = config.getoption("--behaviorci-model")
    config.behaviorci_report = config.getoption("--behaviorci-report")
    # Per-test outcomes collected during the run for the optional JSON report.
    config._behaviorci_results = []


def pytest_collection_modifyitems(config: Any, items: Any) -> None:
    """Validate behavior ids and stash each item's config before tests run."""
    if not config.behaviorci_enabled:
        return

    seen_ids: Dict[str, tuple] = {}

    for item in items:
        if not hasattr(item.obj, "_behavior_config"):
            continue

        config_obj = item.obj._behavior_config
        behavior_id = config_obj.behavior_id

        # Parametrized tests share a function, so key uniqueness on the function
        # location rather than the parametrized node id.
        func_path = item.location[0]
        func_name = item.location[2]
        if "[" in func_name:
            func_name = func_name.split("[")[0]
        func_key = (func_path, func_name)

        if behavior_id in seen_ids and seen_ids[behavior_id] != func_key:
            other_item = next(
                (
                    i
                    for i in items
                    if hasattr(i.obj, "_behavior_config")
                    and i.obj._behavior_config.behavior_id == behavior_id
                ),
                None,
            )
            raise ConfigurationError(
                f"Duplicate behavior_id '{behavior_id}' detected:\n"
                f"  - {item.nodeid}\n"
                f"  - {other_item.nodeid if other_item else 'unknown'}\n"
                f"Each @behavior decorator must have a unique behavior_id."
            )
        seen_ids.setdefault(behavior_id, func_key)

        item.stash[CONFIG_KEY] = {
            "behavior_id": behavior_id,
            "threshold": config_obj.threshold,
            "must_contain": config_obj.must_contain,
            "must_not_contain": config_obj.must_not_contain,
        }


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Any, call: Any) -> Any:
    """Record or compare a behavior snapshot after the test's call phase."""
    outcome = yield

    if CONFIG_KEY not in item.stash:
        return

    if call.when != "call":
        return

    report = outcome.get_result()

    if report.failed:
        return

    config = item.stash.get(CONFIG_KEY, None)
    if config is None:
        return

    output_text = getattr(item.obj, "_behaviorci_result", None)
    input_json = getattr(item.obj, "_behaviorci_input_json", None)

    if output_text is None or input_json is None:
        report.outcome = "failed"
        report.longrepr = (
            f"BehaviorCI: Test '{item.name}' did not capture output. "
            f"Ensure it returns the LLM output string and is decorated with @behavior."
        )
        return

    storage = get_storage(item.config.behaviorci_db_path)
    embedder = get_embedder(item.config.behaviorci_model)
    comparator = Comparator(storage, embedder)

    behavior_id = config["behavior_id"]
    threshold = config["threshold"]
    must_contain = config["must_contain"]
    must_not_contain = config["must_not_contain"]

    record_mode = item.config.behaviorci_record or item.config.behaviorci_update
    record_missing = item.config.behaviorci_record_missing

    try:
        # For centroid baselines the captured value is a list of samples; show
        # the first sample in reports and note how many were averaged.
        display_text = output_text[0] if isinstance(output_text, list) else output_text
        centroid_msg = (
            f" (Centroid of {len(output_text)} samples)" if isinstance(output_text, list) else ""
        )

        if record_mode:
            git_commit = _get_git_commit()
            snapshot_id = comparator.record_snapshot(
                behavior_id=behavior_id,
                input_json=input_json,
                output_text=output_text,
                git_commit=git_commit,
            )

            report.sections.append(
                (
                    "BehaviorCI",
                    f"Recorded snapshot: {behavior_id}{centroid_msg}\n"
                    f"Snapshot ID: {snapshot_id[:16]}...\n\n"
                    f"Review the captured output below to make sure it is correct -- "
                    f"it becomes the baseline for future runs.\n"
                    f"{'=' * 50}\n"
                    f"{display_text}\n"
                    f"{'=' * 50}",
                )
            )
            _collect_result(
                item,
                behavior_id,
                snapshot_id,
                "recorded",
                True,
                output_text,
                model=embedder.model_name,
            )
        else:
            result = comparator.compare(
                behavior_id=behavior_id,
                input_json=input_json,
                output_text=output_text,
                base_threshold=threshold,
                must_contain=must_contain,
                must_not_contain=must_not_contain,
                record_mode=False,
            )

            if not result.passed and record_missing and "No snapshot found" in result.message:
                git_commit = _get_git_commit()
                snapshot_id = comparator.record_snapshot(
                    behavior_id=behavior_id,
                    input_json=input_json,
                    output_text=output_text,
                    git_commit=git_commit,
                )

                report.sections.append(
                    (
                        "BehaviorCI",
                        f"Auto-recorded missing snapshot: {behavior_id}{centroid_msg}\n"
                        f"Snapshot ID: {snapshot_id[:16]}...\n\n"
                        f"Review the captured output below to make sure it is correct -- "
                        f"it becomes the baseline for future runs.\n"
                        f"{'=' * 50}\n"
                        f"{display_text}\n"
                        f"{'=' * 50}\n"
                        f"(Use --behaviorci-record to re-record all, --behaviorci for strict mode)",
                    )
                )
                _collect_result(
                    item,
                    behavior_id,
                    snapshot_id,
                    "recorded_missing",
                    True,
                    output_text,
                    model=embedder.model_name,
                )
            else:
                report_lines = [
                    f"Behavior: {result.behavior_id}{centroid_msg}",
                    f"Snapshot ID: {result.snapshot_id[:16]}...",
                    f"Similarity: {result.similarity:.4f}",
                    f"Threshold: {result.effective_threshold:.4f}",
                ]

                if result.base_threshold != result.effective_threshold:
                    report_lines.append(f"Base threshold: {result.base_threshold:.4f}")

                if result.model_mismatch:
                    report_lines.append(
                        f"WARNING: Model mismatch (stored: {result.stored_model}, "
                        f"current: {result.current_model})"
                    )

                report.sections.append(("BehaviorCI", "\n".join(report_lines)))
                _collect_result(
                    item,
                    result.behavior_id,
                    result.snapshot_id,
                    "checked",
                    result.passed,
                    output_text,
                    model=embedder.model_name,
                    similarity=result.similarity,
                    base_threshold=result.base_threshold,
                    effective_threshold=result.effective_threshold,
                    model_mismatch=result.model_mismatch,
                )

                if not result.passed:
                    report.outcome = "failed"
                    report.longrepr = f"BehaviorCI: {result.message}"

                    if result.lexical_passed and result.similarity < result.effective_threshold:
                        snapshot = storage.find_snapshot(behavior_id, input_json)
                        if snapshot:
                            diff_text = _generate_diff(
                                snapshot.output_text, output_text, result.similarity
                            )
                            report.longrepr += f"\n\n{diff_text}"

    except BehaviorCIError as e:
        report.outcome = "failed"
        report.longrepr = f"BehaviorCI Error: {e.message}"
        _collect_result(
            item,
            behavior_id,
            None,
            "error",
            False,
            output_text,
            model=embedder.model_name,
            error=e.message,
        )


def _collect_result(
    item: Any,
    behavior_id: str,
    snapshot_id: Optional[str],
    action: str,
    passed: bool,
    output_text: Union[str, List[str]],
    model: str,
    similarity: Optional[float] = None,
    base_threshold: Optional[float] = None,
    effective_threshold: Optional[float] = None,
    model_mismatch: bool = False,
    error: Optional[str] = None,
) -> None:
    """Record one test's outcome for the optional JSON report."""
    results = getattr(item.config, "_behaviorci_results", None)
    if results is None:
        return
    entry: Dict[str, Any] = {
        "behavior_id": behavior_id,
        "snapshot_id": snapshot_id,
        "action": action,
        "passed": passed,
        "similarity": similarity,
        "base_threshold": base_threshold,
        "effective_threshold": effective_threshold,
        "model_mismatch": model_mismatch,
        "samples": len(output_text) if isinstance(output_text, list) else 1,
        "model": model,
        "nodeid": item.nodeid,
    }
    if error is not None:
        entry["error"] = error
    results.append(entry)


def _write_json_report(config: Any) -> None:
    """Write the collected results to ``--behaviorci-report`` as JSON."""
    results = getattr(config, "_behaviorci_results", [])

    if config.behaviorci_record:
        mode = "record"
    elif config.behaviorci_update:
        mode = "update"
    elif config.behaviorci_record_missing:
        mode = "record-missing"
    else:
        mode = "check"

    recorded = sum(1 for r in results if r["action"] in ("recorded", "recorded_missing"))
    # Report the model actually used (which may be an injected embedder) rather
    # than the configured default.
    model = results[0]["model"] if results else config.behaviorci_model
    report = {
        "schema": "behaviorci/report/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model": model,
        "summary": {
            "total": len(results),
            "passed": sum(1 for r in results if r["passed"]),
            "failed": sum(1 for r in results if not r["passed"]),
            "recorded": recorded,
            "checked": sum(1 for r in results if r["action"] == "checked"),
        },
        "results": results,
    }

    path = Path(config.behaviorci_report)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def _get_git_commit() -> Optional[str]:
    """Return the current commit hash, or ``None`` outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _generate_diff(
    stored_output: str, current_output: Union[str, List[str]], similarity: float
) -> str:
    """Build a human-readable before/after block for a failed comparison."""
    current_display = current_output[0] if isinstance(current_output, list) else current_output

    # The stored output may be a JSON-encoded list of centroid samples.
    try:
        stored_parsed = json.loads(stored_output)
        stored_display = stored_parsed[0] if isinstance(stored_parsed, list) else stored_output
    except (json.JSONDecodeError, TypeError):
        stored_display = stored_output

    lines = [
        "=" * 50,
        "BEHAVIORAL REGRESSION DETECTED",
        "=" * 50,
        "",
        f"Semantic similarity: {similarity:.4f}",
        "",
        "--- STORED OUTPUT (Primary Sample) ---",
        stored_display[:500],
    ]

    if len(stored_display) > 500:
        lines.append("... (truncated)")

    lines.extend(
        [
            "",
            "--- CURRENT OUTPUT (Primary Sample) ---",
            current_display[:500],
        ]
    )

    if len(current_display) > 500:
        lines.append("... (truncated)")

    lines.extend(
        [
            "",
            "=" * 50,
            "Run with --behaviorci-update to accept the new behavior",
            "=" * 50,
        ]
    )

    return "\n".join(lines)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: Any, config: Any) -> None:
    """Print a short BehaviorCI summary at the end of the run."""
    if not config.behaviorci_enabled:
        return

    storage = get_storage(config.behaviorci_db_path)
    stats = storage.get_stats()

    terminalreporter.write_sep("=", "BehaviorCI Summary")
    terminalreporter.write_line(f"Snapshots: {stats['snapshots']}")
    terminalreporter.write_line(f"Behaviors: {stats['behaviors']}")
    terminalreporter.write_line(f"History records: {stats['history_records']}")

    # Show the model that was actually exercised this run when we know it.
    results = getattr(config, "_behaviorci_results", [])
    model = results[0]["model"] if results else config.behaviorci_model
    if model:
        terminalreporter.write_line(f"Model: {model}")

    if config.behaviorci_record:
        terminalreporter.write_line("Mode: RECORD (snapshots created/updated)")
    elif config.behaviorci_update:
        terminalreporter.write_line("Mode: UPDATE (failing snapshots updated)")
    elif config.behaviorci_record_missing:
        terminalreporter.write_line("Mode: RECORD-MISSING (auto-recording new snapshots)")
    else:
        terminalreporter.write_line("Mode: CHECK (regression testing)")

    if getattr(config, "behaviorci_report", None):
        _write_json_report(config)
        terminalreporter.write_line(f"Report: {config.behaviorci_report}")
