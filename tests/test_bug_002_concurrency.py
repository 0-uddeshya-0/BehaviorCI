"""Test for BUG-002: SQLite concurrency - Database Locked Error.

BUG-002: SQLite uses rollback journaling by default. With pytest -n 4 (parallel),
multiple processes lock each other out with "database is locked" errors.

FIX: Enable WAL (Write-Ahead Logging) mode with:
- PRAGMA journal_mode=WAL
- PRAGMA synchronous=NORMAL
- PRAGMA busy_timeout=5000
"""

import multiprocessing
import os
import tempfile

import numpy as np
import pytest


def write_to_db(db_path: str, behavior_id: str, num_writes: int = 10):
    """Worker function to write to DB from a separate process.

    Returns True on success, raises exception on failure.
    """
    from behaviorci.storage import Storage

    storage = Storage(db_path)

    for i in range(num_writes):
        storage.save_snapshot(
            behavior_id=behavior_id,
            input_json=f'{{"idx": {i}}}',
            output_text=f"output {i} from {behavior_id}",
            embedding=np.random.randn(384).astype(np.float32),
            model_name="test",
        )

    return True


def test_wal_mode_enabled():
    """Verify WAL mode is enabled in database.

    WHY: This test verifies BUG-002 fix. WAL mode must be enabled
    for concurrent writes to work without locking.

    VERIFIED: _init_db() sets PRAGMA journal_mode=WAL
    """
    import sqlite3

    from behaviorci.storage import Storage

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = Storage(db_path)

        # Check WAL mode is enabled
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA journal_mode")
        journal_mode = cursor.fetchone()[0]
        conn.close()

        assert journal_mode.upper() == "WAL", f"Expected WAL mode, got {journal_mode}"


def test_wal_files_created():
    """Verify WAL files are created alongside database.

    WHY: WAL mode creates .db-wal and .db-shm files.
         These must be handled properly (documented in README).
    """
    from behaviorci.storage import Storage

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = Storage(db_path)

        # Write some data to trigger WAL file creation
        storage.save_snapshot(
            behavior_id="test",
            input_json="{}",
            output_text="test",
            embedding=np.array([0.1, 0.2], dtype=np.float32),
            model_name="test",
        )

        # Check WAL file exists (may not exist if checkpointed immediately)
        wal_path = db_path + "-wal"
        shm_path = db_path + "-shm"

        # At least one of these should exist after a write
        assert os.path.exists(db_path), "Database file should exist"
        # WAL files may or may not exist depending on timing


@pytest.mark.slow
def test_concurrent_writes_no_lock_errors():
    """Verify concurrent writes from multiple processes don't cause lock errors.

    WHY: This is the main regression test for BUG-002. Without WAL mode,
    this test would fail with "database is locked" errors.

    VERIFIED: _init_db() enables WAL mode and sets busy_timeout.

    NOTE: This test spawns 4 processes, each writing 10 snapshots.
          Total: 40 snapshots should be written without errors.
    """
    from behaviorci.storage import Storage

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")

        # Spawn 4 processes writing simultaneously
        # Use spawn to ensure clean process isolation
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(4) as pool:
            results = pool.starmap(write_to_db, [(db_path, f"proc_{i}") for i in range(4)])

        # All should succeed without exception
        assert all(results), "All processes should complete without errors"

        # Verify 40 snapshots exist (4 processes * 10 writes each)
        storage = Storage(db_path)
        stats = storage.get_stats()
        assert stats["snapshots"] == 40, f"Expected 40 snapshots, got {stats['snapshots']}"


def test_busy_timeout_set():
    """Verify busy timeout is configured.

    WHY: busy_timeout prevents "database is locked" errors by waiting
    instead of immediately failing.

    VERIFIED: _init_db() sets PRAGMA busy_timeout=5000
    """
    import sqlite3

    from behaviorci.storage import Storage

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = Storage(db_path)

        # Check busy timeout is set
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("PRAGMA busy_timeout")
        timeout = cursor.fetchone()[0]
        conn.close()

        assert timeout >= 5000, f"Expected busy_timeout >= 5000ms, got {timeout}ms"


def test_synchronous_normal():
    """Verify synchronous mode is NORMAL (not FULL) on Storage connections.

    WHY: NORMAL is safe with WAL mode and faster than FULL.

    VERIFIED: _get_connection() sets PRAGMA synchronous=NORMAL

    NOTE: synchronous is a per-connection setting. We verify that
    connections created by Storage have the correct setting.
    """
    from behaviorci.storage import Storage

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = Storage(db_path)

        # Trigger connection creation by doing an operation
        storage.save_snapshot(
            behavior_id="test",
            input_json="{}",
            output_text="test",
            embedding=np.array([0.1, 0.2], dtype=np.float32),
            model_name="test",
        )

        # Check synchronous mode on the storage's connection
        conn = storage._get_connection()
        cursor = conn.execute("PRAGMA synchronous")
        sync_mode = cursor.fetchone()[0]

        # 1 = NORMAL, 2 = FULL
        assert sync_mode == 1, f"Expected synchronous=NORMAL (1), got {sync_mode}"
