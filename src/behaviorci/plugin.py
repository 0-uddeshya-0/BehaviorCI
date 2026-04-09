"""Pytest plugin for BehaviorCI - core engine.

CRITICAL: Uses item.stash to pass data between hooks.
Implements return value capture pattern from api.py.

BUG FIXES APPLIED:
- BUG-003: Uses get_storage() singleton instead of direct Storage() instantiation
- FIX-004: Added --behaviorci-record-missing flag for CI workflows
- FIX-005: Validates behavior_id uniqueness at collection time
- FIX-006: Removed pytest_runtest_call hook to prevent double-execution.
           The previous implementation had a hook that called the test runner,
           which conflicted with pytest's default runner causing duplicate executions.
- FIX-009: Added mandatory output review block on snapshot recording to prevent blind baselines.
"""

import pytest
import os
import subprocess
from typing import Optional, Dict

from .api import get_behavior_config, serialize_inputs
from .storage import get_storage, reset_all_storage
from .embedder import get_embedder, DEFAULT_MODEL_NAME
from .comparator import Comparator
from .exceptions import BehaviorCIError, SerializationError, ConfigurationError


# Stash keys for passing data between hooks
CONFIG_KEY = pytest.StashKey[dict]()


def pytest_addoption(parser):
    """Add BehaviorCI command-line options."""
    group = parser.getgroup("behaviorci", "Behavioral regression testing for LLMs")
    group.addoption(
        "--behaviorci",
        action="store_true",
        default=False,
        help="Enable BehaviorCI regression testing"
    )
    group.addoption(
        "--behaviorci-record",
        action="store_true",
        default=False,
        help="Record new snapshots (overwrites existing)"
    )
    group.addoption(
        "--behaviorci-update",
        action="store_true",
        default=False,
        help="Update failing snapshots"
    )
    group.addoption(
        "--behaviorci-record-missing",
        action="store_true",
        default=False,
        help="Record missing snapshots instead of failing (CI workflow)"
    )
    group.addoption(
        "--behaviorci-db",
        action="store",
        default=None,
        help="Path to BehaviorCI database (default: .behaviorci/behaviorci.db)"
    )
    group.addoption(
        "--behaviorci-model",
        action="store",
        default='sentence-transformers/all-MiniLM-L6-v2',  # FIX: Default model
        help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )


def pytest_configure(config):
    """Configure BehaviorCI based on options."""
    config.behaviorci_enabled = (
        config.getoption("--behaviorci") or
        config.getoption("--behaviorci-record") or
        config.getoption("--behaviorci-update") or
        config.getoption("--behaviorci-record-missing")
    )
    config.behaviorci_record = config.getoption("--behaviorci-record")
    config.behaviorci_update = config.getoption("--behaviorci-update")
    config.behaviorci_record_missing = config.getoption("--behaviorci-record-missing")
    config.behaviorci_db_path = config.getoption("--behaviorci-db")
    # FIX: Ensure model name is never None
    config.behaviorci_model = config.getoption("--behaviorci-model")


def pytest_collection_modifyitems(config, items):
    """Validate behavior tests during collection."""
    if not config.behaviorci_enabled:
        return

    seen_ids: Dict[str, tuple] = {}

    for item in items:
        if hasattr(item.obj, '_behavior_config'):
            config_obj = item.obj._behavior_config
            behavior_id = config_obj.behavior_id
            func_path = item.location[0]
            func_name = item.location[2]
            if '[' in func_name:
                func_name = func_name.split('[')[0]
            func_key = (func_path, func_name)

            if behavior_id in seen_ids:
                other_func_key = seen_ids[behavior_id]
                if other_func_key != func_key:
                    other_item = next(
                        (i for i in items
                         if hasattr(i.obj, '_behavior_config')
                         and i.obj._behavior_config.behavior_id == behavior_id),
                        None
                    )
                    raise ConfigurationError(
                        f"Duplicate behavior_id '{behavior_id}' detected:\n"
                        f"  - {item.nodeid}\n"
                        f"  - {other_item.nodeid if other_item else 'unknown'}\n"
                        f"Each @behavior decorator must have a unique behavior_id."
                    )
            else:
                seen_ids[behavior_id] = func_key

            item.stash[CONFIG_KEY] = {
                'behavior_id': behavior_id,
                'threshold': config_obj.threshold,
                'must_contain': config_obj.must_contain,
                'must_not_contain': config_obj.must_not_contain,
            }


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Generate report for behavior tests."""
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

    output_text = getattr(item.obj, '_behaviorci_result', None)
    input_json = getattr(item.obj, '_behaviorci_input_json', None)

    if output_text is None or input_json is None:
        report.outcome = "failed"
        report.longrepr = (
            f"BehaviorCI: Test '{item.name}' did not capture output. "
            f"Ensure it returns the LLM output string and is decorated with @behavior."
        )
        return

    storage = get_storage(item.config.behaviorci_db_path)
    # FIX: Now safe to pass None (will use default), but config ensures it's set
    embedder = get_embedder(item.config.behaviorci_model)
    comparator = Comparator(storage, embedder)

    behavior_id = config['behavior_id']
    threshold = config['threshold']
    must_contain = config['must_contain']
    must_not_contain = config['must_not_contain']

    record_mode = item.config.behaviorci_record or item.config.behaviorci_update
    record_missing = item.config.behaviorci_record_missing

    try:
        if record_mode:
            git_commit = _get_git_commit()
            snapshot_id = comparator.record_snapshot(
                behavior_id=behavior_id,
                input_json=input_json,
                output_text=output_text,
                git_commit=git_commit
            )

            report.sections.append((
                "BehaviorCI",
                f"✅ Recorded snapshot: {behavior_id}\n"
                f"Snapshot ID: {snapshot_id[:16]}...\n\n"
                f"⚠️ URGENT: Review the captured output below to ensure it is correct.\n"
                f"This will be your new ground truth for future tests.\n"
                f"{'='*50}\n"
                f"{output_text}\n"
                f"{'='*50}"
            ))
        else:
            result = comparator.compare(
                behavior_id=behavior_id,
                input_json=input_json,
                output_text=output_text,
                base_threshold=threshold,
                must_contain=must_contain,
                must_not_contain=must_not_contain,
                record_mode=False
            )

            if not result.passed and record_missing and "No snapshot found" in result.message:
                git_commit = _get_git_commit()
                snapshot_id = comparator.record_snapshot(
                    behavior_id=behavior_id,
                    input_json=input_json,
                    output_text=output_text,
                    git_commit=git_commit
                )

                report.sections.append((
                    "BehaviorCI",
                    f"✅ Auto-recorded missing snapshot: {behavior_id}\n"
                    f"Snapshot ID: {snapshot_id[:16]}...\n\n"
                    f"⚠️ URGENT: Review the captured output below to ensure it is correct.\n"
                    f"This will be your new ground truth for future tests.\n"
                    f"{'='*50}\n"
                    f"{output_text}\n"
                    f"{'='*50}\n"
                    f"(Use --behaviorci-record to record all, --behaviorci for strict mode)"
                ))
            else:
                report_lines = [
                    f"Behavior: {result.behavior_id}",
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

                if not result.passed:
                    report.outcome = "failed"
                    report.longrepr = f"BehaviorCI: {result.message}"

                    if result.lexical_passed and result.similarity < result.effective_threshold:
                        snapshot = storage.find_snapshot(behavior_id, input_json)
                        if snapshot:
                            diff_text = _generate_diff(
                                snapshot.output_text,
                                output_text,
                                result.similarity
                            )
                            report.longrepr += f"\n\n{diff_text}"

    except BehaviorCIError as e:
        report.outcome = "failed"
        report.longrepr = f"BehaviorCI Error: {e.message}"


def _get_git_commit() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


def _generate_diff(stored_output: str, current_output: str, similarity: float) -> str:
    """Generate a readable diff between stored and current output."""
    lines = [
        "=" * 50,
        "BEHAVIORAL REGRESSION DETECTED",
        "=" * 50,
        "",
        f"Semantic similarity: {similarity:.4f}",
        "",
        "--- STORED OUTPUT ---",
        stored_output[:500],
    ]

    if len(stored_output) > 500:
        lines.append("... (truncated)")

    lines.extend([
        "",
        "--- CURRENT OUTPUT ---",
        current_output[:500],
    ])

    if len(current_output) > 500:
        lines.append("... (truncated)")

    lines.extend([
        "",
        "=" * 50,
        "Run with --behaviorci-update to accept new behavior",
        "=" * 50,
    ])

    return "\n".join(lines)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print BehaviorCI summary at end of test run."""
    if not config.behaviorci_enabled:
        return

    storage = get_storage(config.behaviorci_db_path)
    stats = storage.get_stats()

    terminalreporter.write_sep("=", "BehaviorCI Summary")
    terminalreporter.write_line(f"Snapshots: {stats['snapshots']}")
    terminalreporter.write_line(f"Behaviors: {stats['behaviors']}")
    terminalreporter.write_line(f"History records: {stats['history_records']}")

    if config.behaviorci_model:
        terminalreporter.write_line(f"Model: {config.behaviorci_model}")

    if config.behaviorci_record:
        terminalreporter.write_line("Mode: RECORD (snapshots created/updated)")
    elif config.behaviorci_update:
        terminalreporter.write_line("Mode: UPDATE (failing snapshots updated)")
    elif config.behaviorci_record_missing:
        terminalreporter.write_line("Mode: RECORD-MISSING (auto-recording new snapshots)")
    else:
        terminalreporter.write_line("Mode: CHECK (regression testing)")


def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    pass
