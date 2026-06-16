"""SQLite storage for behavioral snapshots and their similarity history.

The database keeps two tables: ``snapshots`` holds one baseline per
(behavior_id, input) pair, and ``similarity_history`` records every score we
measure against a snapshot so the comparator can reason about variance.

Connections are opened per-thread and the database runs in WAL mode so that
``pytest-xdist`` workers can read and write concurrently without tripping over
``database is locked`` errors.
"""

import hashlib
import os
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .exceptions import SnapshotNotFoundError, StorageError
from .models import Snapshot

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
CREATE INDEX IF NOT EXISTS idx_similarity_timestamp
ON similarity_history(timestamp DESC);
"""

# Sentinel SQLite recognises as a true in-memory database.
_MEMORY_PATH = ":memory:"

# One Storage instance per database path. Reusing a single instance keeps the
# number of open SQLite connections bounded even across very large test suites.
_storage_lock = threading.Lock()
_storage_instances: Dict[str, "Storage"] = {}


def get_storage(db_path: Optional[str] = None) -> "Storage":
    """Return the shared :class:`Storage` for ``db_path``, creating it once.

    Args:
        db_path: Path to the SQLite database. Defaults to
            ``.behaviorci/behaviorci.db``. Pass ``":memory:"`` for an in-memory
            database (handy in tests).
    """
    if db_path is None:
        db_path = os.path.join(".behaviorci", "behaviorci.db")

    # Normalise real paths so different spellings map to the same instance, but
    # leave ":memory:" untouched -- resolving it would point at a real file.
    if db_path != _MEMORY_PATH:
        db_path = str(Path(db_path).resolve())

    with _storage_lock:
        if db_path not in _storage_instances:
            _storage_instances[db_path] = Storage(db_path)
        return _storage_instances[db_path]


def reset_storage(db_path: Optional[str] = None) -> None:
    """Drop the cached :class:`Storage` for ``db_path`` (used by test fixtures)."""
    if db_path is None:
        db_path = os.path.join(".behaviorci", "behaviorci.db")

    if db_path != _MEMORY_PATH:
        db_path = str(Path(db_path).resolve())

    with _storage_lock:
        instance = _storage_instances.pop(db_path, None)
    if instance is not None:
        instance.close()


def reset_all_storage() -> None:
    """Drop every cached :class:`Storage` instance."""
    with _storage_lock:
        instances = list(_storage_instances.values())
        _storage_instances.clear()
    for instance in instances:
        instance.close()


def compute_snapshot_id(behavior_id: str, input_json: str) -> str:
    """Derive a stable snapshot id from a behavior id and its canonical input."""
    data = f"{behavior_id}:{input_json}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class Storage:
    """SQLite-backed store for snapshots and similarity history.

    Prefer :func:`get_storage` over instantiating this directly so connections
    are reused. Each thread gets its own connection because SQLite connection
    objects are not safe to share across threads.
    """

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = os.path.join(".behaviorci", "behaviorci.db")

        self._is_memory = db_path == _MEMORY_PATH
        self.db_path = Path(db_path)

        # An in-memory database is not a real path, so there is nothing to create.
        if not self._is_memory:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Return this thread's connection, opening it on first use.

        ``synchronous`` and ``busy_timeout`` are per-connection PRAGMAs, so they
        have to be re-applied for every new connection rather than relying on
        the WAL setting persisted in the database file. In-memory databases are
        private to each connection, so they also need the schema created here.
        """
        if not hasattr(self._local, "connection") or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path), uri=False, detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA busy_timeout=5000")

            if self._is_memory:
                self._local.connection.executescript(SCHEMA)
                self._local.connection.commit()

        return self._local.connection  # type: ignore[no-any-return]

    def close(self) -> None:
        """Close the current thread's connection if one is open.

        Long-lived processes can simply let the connection live for their
        lifetime; this exists so tests and the ``clear`` command can release the
        file handle (which matters on Windows, where open files can't be
        deleted).
        """
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            conn.close()
            self._local.connection = None

    def _init_db(self) -> None:
        """Create the schema and enable WAL mode for concurrent access.

        ``journal_mode=WAL`` is stored in the database file, so enabling it once
        here is enough. In-memory databases are single-process and set up their
        schema in :meth:`_get_connection` instead.
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
                    time.sleep(0.1 * (attempt + 1))
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
        git_commit: Optional[str] = None,
    ) -> str:
        """Insert a snapshot, replacing any existing one for the same input.

        Args:
            behavior_id: Logical behavior identifier.
            input_json: Canonical JSON of the test's input arguments.
            output_text: Captured output text.
            embedding: Embedding vector; normalised to unit length before storing.
            model_name: Name of the embedding model that produced ``embedding``.
            git_commit: Commit hash the snapshot was recorded at, if known.

        Returns:
            The computed snapshot id.
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
                INSERT INTO snapshots (
                    id, behavior_id, input_json, output_text,
                    embedding, model_name, created_at, git_commit
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    behavior_id,
                    input_json,
                    output_text,
                    embedding_blob,
                    model_name,
                    created_at,
                    git_commit,
                ),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to save snapshot: {e}")

        return snapshot_id

    def get_snapshot(self, snapshot_id: str) -> Snapshot:
        """Return the snapshot with ``snapshot_id``.

        Raises:
            SnapshotNotFoundError: If no such snapshot exists.
        """
        try:
            conn = self._get_connection()
            row = conn.execute("SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)).fetchone()

            if row is None:
                raise SnapshotNotFoundError(snapshot_id, "unknown")

            return Snapshot(
                id=row["id"],
                behavior_id=row["behavior_id"],
                input_json=row["input_json"],
                output_text=row["output_text"],
                embedding=row["embedding"],
                model_name=row["model_name"],
                created_at=row["created_at"],
                git_commit=row["git_commit"],
            )
        except sqlite3.Error as e:
            raise StorageError(f"Failed to retrieve snapshot: {e}")

    def find_snapshot(self, behavior_id: str, input_json: str) -> Optional[Snapshot]:
        """Return the snapshot for ``(behavior_id, input_json)`` or ``None``."""
        snapshot_id = compute_snapshot_id(behavior_id, input_json)
        try:
            return self.get_snapshot(snapshot_id)
        except SnapshotNotFoundError:
            return None

    def record_similarity(self, snapshot_id: str, similarity: float) -> None:
        """Append a measured similarity score for variance tracking."""
        timestamp = int(datetime.now().timestamp())

        try:
            conn = self._get_connection()
            conn.execute(
                """
                INSERT INTO similarity_history (snapshot_id, similarity, timestamp)
                VALUES (?, ?, ?)
                """,
                (snapshot_id, similarity, timestamp),
            )
            conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to record similarity: {e}")

    def get_similarity_history(self, snapshot_id: str, limit: int = 10) -> List[float]:
        """Return the most recent similarity scores for a snapshot, newest first."""
        try:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT similarity FROM similarity_history
                WHERE snapshot_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (snapshot_id, limit),
            ).fetchall()

            return [row["similarity"] for row in rows]  # type: ignore[no-any-return]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get similarity history: {e}")

    def get_all_snapshots_for_behavior(self, behavior_id: str) -> List[Snapshot]:
        """Return every snapshot recorded under ``behavior_id``."""
        try:
            conn = self._get_connection()
            rows = conn.execute(
                "SELECT * FROM snapshots WHERE behavior_id = ?", (behavior_id,)
            ).fetchall()

            return [
                Snapshot(
                    id=row["id"],
                    behavior_id=row["behavior_id"],
                    input_json=row["input_json"],
                    output_text=row["output_text"],
                    embedding=row["embedding"],
                    model_name=row["model_name"],
                    created_at=row["created_at"],
                    git_commit=row["git_commit"],
                )
                for row in rows
            ]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get snapshots: {e}")

    def delete_snapshot(self, snapshot_id: str) -> bool:
        """Delete a snapshot and its history. Returns ``True`` if one was removed."""
        try:
            conn = self._get_connection()
            cursor = conn.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
            conn.execute("DELETE FROM similarity_history WHERE snapshot_id = ?", (snapshot_id,))
            conn.commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to delete snapshot: {e}")

    def clear_all(self) -> None:
        """Remove all snapshots and history. Irreversible."""
        try:
            conn = self._get_connection()
            conn.execute("DELETE FROM similarity_history")
            conn.execute("DELETE FROM snapshots")
            conn.commit()
        except sqlite3.Error as e:
            raise StorageError(f"Failed to clear database: {e}")

    def get_stats(self) -> dict:
        """Return counts of snapshots, distinct behaviors, and history rows."""
        try:
            conn = self._get_connection()
            snapshot_count = conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
            history_count = conn.execute("SELECT COUNT(*) FROM similarity_history").fetchone()[0]
            behavior_count = conn.execute(
                "SELECT COUNT(DISTINCT behavior_id) FROM snapshots"
            ).fetchone()[0]

            return {
                "snapshots": snapshot_count,
                "history_records": history_count,
                "behaviors": behavior_count,
            }
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get stats: {e}")

    def get_behavior_summary(self) -> List[tuple]:
        """Return ``(behavior_id, snapshot_count, last_recorded_at)`` per behavior."""
        try:
            conn = self._get_connection()
            rows = conn.execute("""
                SELECT
                    behavior_id,
                    COUNT(*) as count,
                    MAX(created_at) as last_run
                FROM snapshots
                GROUP BY behavior_id
                ORDER BY count DESC
                """).fetchall()

            return [(row["behavior_id"], row["count"], row["last_run"]) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get behavior summary: {e}")

    def get_similarity_history_with_timestamps(
        self, snapshot_id: str, limit: int = 10
    ) -> List[tuple]:
        """Return ``(similarity, timestamp)`` pairs for a snapshot, newest first."""
        try:
            conn = self._get_connection()
            rows = conn.execute(
                """
                SELECT similarity, timestamp FROM similarity_history
                WHERE snapshot_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (snapshot_id, limit),
            ).fetchall()

            return [(row["similarity"], row["timestamp"]) for row in rows]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get similarity history: {e}")
