"""SQLite storage layer for BehaviorCI snapshots and history.

BUG FIXES APPLIED:
- BUG-001: datetime import moved to top (was at bottom causing NameError)
- BUG-002: WAL mode enabled for concurrent writes (prevents "database is locked")
- BUG-003: Singleton pattern with thread-local connections (prevents connection leak)
- FIX-007: busy_timeout now set on every thread-local connection in _get_connection().
           Previously it was set only on the init connection (which is immediately
           closed), so thread-local connections had no timeout and could still
           produce "database is locked" errors under concurrent load.
- FIX-008: :memory: path is now handled as a true SQLite in-memory database.
           Previously, Storage(":memory:") would call Path(":memory:").parent.mkdir()
           and sqlite3.connect(":memory:") under WAL mode produced literal files
           named ":memory:" and ":memory:-shm" on disk. Now the path is detected
           and directory creation + WAL init are skipped for in-memory databases.
"""

import sqlite3
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import numpy as np

from .models import Snapshot, SimilarityRecord
from .exceptions import StorageError, SnapshotNotFoundError

SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id TEXT PRIMARY KEY,
    behavior_id TEXT NOT NULL,
    input_json TEXT NOT NULL,
    output_text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    created_at INTEGER,
    git_commit TEXT
);

CREATE TABLE IF NOT EXISTS similarity_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL,
    similarity REAL NOT NULL,
    timestamp INTEGER,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id)
);

CREATE INDEX IF NOT EXISTS idx_behavior ON snapshots(behavior_id);
CREATE INDEX IF NOT EXISTS idx_snapshot_history ON similarity_history(snapshot_id);

-- TASK 2 (v0.2): Index for timestamp-ordered queries (e.g., behaviorci history command)
CREATE INDEX IF NOT EXISTS idx_similarity_timestamp 
ON similarity_history(timestamp DESC);
"""

# Sentinel value recognised by SQLite as a true in-memory database
_MEMORY_PATH = ":memory:"

# Module-level singleton storage
# WHY: Prevents connection exhaustion with 1000+ tests (BUG-003)
# APPROACH: Dict of db_path -> Storage instance. Rejected: connection pooling (too complex)
# RISKS: Must call reset_storage() in tests to avoid state leakage
# VERIFIED BY: tests/test_bug_003_singleton.py
_storage_lock = threading.Lock()
_storage_instances: Dict[str, 'Storage'] = {}


def get_storage(db_path: Optional[str] = None) -> 'Storage':
    """Get or create Storage singleton for given path.

    WHY: BUG-003 - Each Storage() instantiation opens a new SQLite connection.
         With 1000 tests, this hits OS file descriptor limit.

    APPROACH: Singleton per db_path with thread-safe creation.
              Rejected: global single instance (doesn't support multiple DB paths)

    RISKS: Tests must reset singleton state. Use reset_storage() in test fixtures.

    VERIFIED BY: tests/test_bug_003_singleton.py

    Args:
        db_path: Path to SQLite database. Defaults to .behaviorci/behaviorci.db
                 Pass ":memory:" for a true in-memory database (tests only).

    Returns:
        Storage singleton instance for the given path
    """
    if db_path is None:
        db_path = os.path.join('.behaviorci', 'behaviorci.db')

    # Normalise path for consistent lookup, but leave :memory: unchanged —
    # Path(":memory:").resolve() produces an incorrect filesystem path.
    if db_path != _MEMORY_PATH:
        db_path = str(Path(db_path).resolve())

    with _storage_lock:
        if db_path not in _storage_instances:
            _storage_instances[db_path] = Storage(db_path)
        return _storage_instances[db_path]


def reset_storage(db_path: Optional[str] = None) -> None:
    """Reset storage singleton for given path (useful for testing).

    Args:
        db_path: Path to SQLite database. If None, resets default path.
    """
    if db_path is None:
        db_path = os.path.join('.behaviorci', 'behaviorci.db')

    if db_path != _MEMORY_PATH:
        db_path = str(Path(db_path).resolve())

    with _storage_lock:
        if db_path in _storage_instances:
            del _storage_instances[db_path]


def reset_all_storage() -> None:
    """Reset all storage singletons (nuclear option for testing)."""
    with _storage_lock:
        _storage_instances.clear()


def compute_snapshot_id(behavior_id: str, input_json: str) -> str:
    """Compute unique snapshot ID from behavior_id and canonical input JSON."""
    data = f"{behavior_id}:{input_json}"
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class Storage:
    """SQLite storage for behavioral snapshots.

    WHY: BUG-003 - Uses thread-local connections for thread safety.
         SQLite connections are NOT thread-safe.

    APPROACH: threading.local() stores connection per thread.
              Rejected: connection pool (overkill for SQLite)

    RISKS: Each thread opens its own connection. With many threads,
           could hit SQLite connection limit (default 1000).

    VERIFIED BY: tests/test_bug_003_singleton.py
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage with database path.

        NOTE: Use get_storage() instead of direct instantiation to get singleton.

        Args:
            db_path: Path to SQLite database. Defaults to .behaviorci/behaviorci.db
                     Pass ":memory:" for a true in-memory database (tests only).
        """
        if db_path is None:
            db_path = os.path.join('.behaviorci', 'behaviorci.db')

        self._is_memory = (db_path == _MEMORY_PATH)
        self.db_path = Path(db_path)

        # FIX-008: Skip directory creation for :memory: — it is not a real path.
        # Previously this called Path(":memory:").parent.mkdir() which created
        # a "." directory entry, and sqlite3.connect(":memory:") under WAL mode
        # produced literal ":memory:" and ":memory:-shm" files on disk.
        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Thread-local storage for connections (BUG-003)
        self._local = threading.local()

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.

        WHY: BUG-003 - SQLite connections are NOT thread-safe.
              Each thread must have its own connection.

        FIX-007: busy_timeout is a per-connection setting and is NOT persisted
                 by WAL mode. It must be set on every new connection, not just
                 the init connection. Without this, thread-local connections had
                 no timeout and could still deadlock under concurrent writes.

        NOTE: For :memory: databases each thread gets its own independent
              in-memory database. This is intentional — it matches test
              isolation expectations and is consistent with SQLite's behaviour.

        Returns:
            Thread-local SQLite connection
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                uri=False,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._local.connection.row_factory = sqlite3.Row
            # FIX-007: These PRAGMAs must be set on every new connection.
            # journal_mode=WAL persists in the DB file so it is already active,
            # but synchronous and busy_timeout are per-connection and must be
            # re-applied every time a new thread-local connection is created.
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA busy_timeout=5000")

            # For in-memory databases the schema must also be created per
            # connection because each thread gets a fresh in-memory DB.
            if self._is_memory:
                self._local.connection.executescript(SCHEMA)
                self._local.connection.commit()

        # MYPY FIX: explicitly ignore upstream sqlite3 connection Any return
        return self._local.connection  # type: ignore[no-any-return]

    def _init_db(self) -> None:
        """Initialize database schema with WAL mode and retry for concurrency.
        WHY: BUG-002 - SQLite rollback journal causes "database is locked"
             errors with concurrent writes from pytest-xdist workers.

        APPROACH: WAL (Write-Ahead Logging) mode allows concurrent reads/writes.
                  - journal_mode=WAL: Enables WAL mode (persisted in DB file)
                  - synchronous=NORMAL: Safe with WAL, faster than FULL
                  - busy_timeout=5000: Set here too, for the init connection

        NOTE: journal_mode=WAL is a database-level persistent setting.
              synchronous and busy_timeout are per-connection (see _get_connection).
              WAL mode is a no-op for :memory: databases and is skipped.

        RISKS: WAL creates .db-wal and .db-shm files that must be handled in CI.

        VERIFIED BY: tests/test_bug_002_concurrency.py
        
        # FIX-008: :memory: databases don't support WAL mode and don't need it —
        # they are single-process by definition. Schema is created per-connection
        # in _get_connection() instead.
        """
        if self._is_memory:
            return

        retries = 5
        for attempt in range(retries):
            try:
                conn = sqlite3.connect(str(self.db_path))
                conn.row_factory = sqlite3.Row
                try:
                    conn.execute("PRAGMA journal_mode=WAL")
                    conn.execute("PRAGMA synchronous=NORMAL")
                    conn.execute("PRAGMA busy_timeout=5000")
                    conn.executescript(SCHEMA)
                    conn.commit()
                    return
                finally:
                    conn.close()
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < retries - 1:
                    import time
                    time.sleep(0.1 * (attempt + 1))  # exponential-ish backoff
                    continue
                raise StorageError(f"Failed to initialize database: {e}")
            except sqlite3.Error as e:
                raise StorageError(f"Failed to initialize database: {e}")
       
    def save_snapshot(
        self,
        behavior_id: str,
        input_json: str,
        output_text: str,
        embedding: np.ndarray,
        model_name: str,
        git_commit: Optional[str] = None
    ) -> str:
        """Save a new snapshot, overwriting any existing one.

        Args:
            behavior_id: Logical behavior identifier
            input_json: JSON-serialized input arguments
            output_text: LLM output text
            embedding: Normalized embedding vector (numpy array)
            model_name: Name of embedding model used
            git_commit: Optional git commit hash

        Returns:
            snapshot_id: The computed snapshot ID
        """
        snapshot_id = compute_snapshot_id(behavior_id, input_json)

        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-6:
            embedding = embedding / norm

        embedding_blob = embedding.tobytes()
        created_at = int(datetime.now().timestamp())

        try:
            conn = self._get_connection()
            conn.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
            conn.execute("DELETE FROM similarity_history WHERE snapshot_id = ?", (snapshot_id,))
            conn.execute(
                """
                INSERT INTO snapshots
                (id, behavior_id, input_json, output_text, embedding, model_name, created_at, git_commit)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (snapshot_id, behavior_id, input_json, output_text, embedding_blob,
                 model_name, created_at, git_commit)
            )
            conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save snapshot: {e}")

        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Snapshot:
        """Retrieve a snapshot by ID.

        Args:
            snapshot_id: The snapshot ID

        Returns:
            Snapshot object

        Raises:
            SnapshotNotFoundError: If snapshot doesn't exist
        """
        try:
            conn = self._get_connection()
            row = conn.execute(
                "SELECT * FROM snapshots WHERE id = ?",
                (snapshot_id,)
            ).fetchone()

            if row is None:
                raise SnapshotNotFoundError(snapshot_id, "unknown")

            return Snapshot(
                id=row['id'],
                behavior_id=row['behavior_id'],
                input_json=row['input_json'],
                output_text=row['output_text'],
                embedding=row['embedding'],
                model_name=row['model_name'],
                created_at=row['created_at'],
                git_commit=row['git_commit']
            )
        except sqlite3.Error as e:
            raise StorageError(f"Failed to retrieve snapshot: {e}")

    def find_snapshot(self, behavior_id: str, input_json: str) -> Optional[Snapshot]:
        """Find snapshot by behavior_id and input.

        Args:
            behavior_id: Logical behavior identifier
            input_json: JSON-serialized input arguments

        Returns:
            Snapshot if found, None otherwise
        """
        snapshot_id = compute_snapshot_id(behavior_id, input_json)
        try:
            return self.get_snapshot(snapshot_id)
        except SnapshotNotFoundError:
            return None

    def record_similarity(self, snapshot_id: str, similarity: float) -> None:
        """Record a similarity comparison for variance tracking.

        Args:
            snapshot_id: The snapshot ID
            similarity: Similarity score (0-1)
        """
        timestamp = int(datetime.now().timestamp())

        try:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO similarity_history (snapshot_id, similarity, timestamp)
                VALUES (?, ?, ?)
                """,
                (snapshot_id, similarity, timestamp)
            )
            conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to record similarity: {e}")

    def get_similarity_history(self, snapshot_id: str, limit: int = 10) -> List[float]:
        """Get recent similarity scores for variance tracking.

        Args:
            snapshot_id: The snapshot ID
            limit: Maximum number of records to return

        Returns:
            List of similarity scores
        """
        try:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT similarity FROM similarity_history
                WHERE snapshot_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (snapshot_id, limit)
            ).fetchall()

            # MYPY FIX: Ignores SQLite returning lists of Any
            return [row['similarity'] for row in rows]  # type: ignore[no-any-return]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get similarity history: {e}")

    def get_all_snapshots_for_behavior(self, behavior_id: str) -> List[Snapshot]:
        """Get all snapshots for a behavior ID.

        Args:
            behavior_id: Logical behavior identifier

        Returns:
            List of snapshots
        """
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE behavior_id = ?",
                (behavior_id,)
            ).fetchall()

            return [
                Snapshot(
                    id=row['id'],
                    behavior_id=row['behavior_id'],
                    input_json=row['input_json'],
                    output_text=row['output_text'],
                    embedding=row['embedding'],
                    model_name=row['model_name'],
                    created_at=row['created_at'],
                    git_commit=row['git_commit']
                )
                for row in rows
            ]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get snapshots: {e}")

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot and its history.

        Args:
            snapshot_id: The snapshot ID

        Returns:
            True if deleted, False if not found
        """
        try:
            conn = self._get_connection()
            cursor = conn.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
            conn.execute("DELETE FROM similarity_history WHERE snapshot_id = ?", (snapshot_id,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to delete snapshot: {e}")

    def clear_all(self) -> None:
        """Clear all snapshots and history. USE WITH CAUTION."""
        try:
            conn = self._get_connection()
            conn.execute("DELETE FROM similarity_history")
            conn.execute("DELETE FROM snapshots")
            conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to clear database: {e}")

    def get_stats(self) -> dict:
        """Get database statistics.

        Returns:
            Dictionary with snapshot count and history count
        """
        try:
            conn = self._get_connection()
            snapshot_count = conn.execute(
                "SELECT COUNT(*) FROM snapshots"
            ).fetchone()[0]
            history_count = conn.execute(
                "SELECT COUNT(*) FROM similarity_history"
            ).fetchone()[0]
            behavior_count = conn.execute(
                "SELECT COUNT(DISTINCT behavior_id) FROM snapshots"
            ).fetchone()[0]

            return {
                'snapshots': snapshot_count,
                'history_records': history_count,
                'behaviors': behavior_count
            }
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get stats: {e}")

    def get_behavior_summary(self) -> List[tuple]:
        """Get summary of behaviors for CLI stats command.

        HIGH-002 FIX: Added this method to avoid raw SQLite in CLI.

        WHY: CLI stats was bypassing the Storage abstraction with raw
        sqlite3.connect() calls, which:
        1. Bypassed singleton pattern (extra connections)
        2. Bypassed WAL mode settings (potential locking issues)
        3. Violated abstraction layer

        Returns:
            List of tuples: (behavior_id, count, last_run_timestamp)

        VERIFIED BY: CLI stats command uses this method
        """
        try:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT 
                    behavior_id, 
                    COUNT(*) as count, 
                    MAX(created_at) as last_run
                FROM snapshots
                GROUP BY behavior_id
                ORDER BY count DESC
                """
            ).fetchall()

            # MYPY FIX: Ignores SQLite returning lists of Any
            return [
                (row['behavior_id'], row['count'], row['last_run'])
                for row in rows
            ]  # type: ignore[no-any-return]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get behavior summary: {e}")

    def get_similarity_history_with_timestamps(
        self,
        snapshot_id: str,
        limit: int = 10
    ) -> List[tuple]:
        """Get similarity history with timestamps for the history command.

        TASK 2 (v0.2): This method uses the idx_similarity_timestamp index
        for efficient timestamp-ordered queries.

        Args:
            snapshot_id: The snapshot ID
            limit: Maximum number of records to return

        Returns:
            List of tuples: (similarity, timestamp)
        """
        try:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT similarity, timestamp FROM similarity_history
                WHERE snapshot_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (snapshot_id, limit)
            ).fetchall()

            # MYPY FIX: Ignores SQLite returning lists of Any
            return [(row['similarity'], row['timestamp']) for row in rows]  # type: ignore[no-any-return]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get similarity history: {e}")
