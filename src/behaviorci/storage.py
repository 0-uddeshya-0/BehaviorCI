"""SQLite storage layer for BehaviorCI snapshots and history.

BUG FIXES APPLIED:
- BUG-001: datetime import moved to top (was at bottom causing NameError)
- BUG-002: WAL mode enabled for concurrent writes (prevents "database is locked")
- BUG-003: Singleton pattern with thread-local connections (prevents connection leak)
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
"""


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
        
    Returns:
        Storage singleton instance for the given path
    """
    if db_path is None:
        db_path = os.path.join('.behaviorci', 'behaviorci.db')
    
    # Normalize path for consistent lookup
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
    db_path = str(Path(db_path).resolve())
    
    with _storage_lock:
        if db_path in _storage_instances:
            # Close any thread-local connections
            storage = _storage_instances[db_path]
            # Note: We can't easily close thread-local connections,
            # but they'll be garbage collected when the threads exit
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
        """
        if db_path is None:
            db_path = os.path.join('.behaviorci', 'behaviorci.db')
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread-local storage for connections (BUG-003)
        self._local = threading.local()
        
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection.
        
        WHY: BUG-003 - SQLite connections are NOT thread-safe.
             Each thread must have its own connection.
        
        VERIFIED BY: tests/test_bug_003_singleton.py
        
        Returns:
            Thread-local SQLite connection
        """
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(
                str(self.db_path),
                # Enable URI mode for additional options if needed
                uri=False,
                # Detect types for better compatibility
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            self._local.connection.row_factory = sqlite3.Row
            # Set synchronous mode for each new connection (BUG-002)
            # WAL mode settings persist, but we set this for consistency
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
        return self._local.connection
    
    def _init_db(self) -> None:
        """Initialize database schema with WAL mode for concurrency.
        
        WHY: BUG-002 - SQLite rollback journal causes "database is locked" 
             errors with concurrent writes from pytest-xdist workers.
        
        APPROACH: WAL (Write-Ahead Logging) mode allows concurrent reads/writes.
                  - journal_mode=WAL: Enables WAL mode
                  - synchronous=NORMAL: Safe with WAL, faster than FULL
                  - busy_timeout=5000: Prevents "database is locked" errors
        
        RISKS: WAL creates .db-wal and .db-shm files that must be:
               - Committed to CI cache, OR
               - Removed before git commit (documented in README)
        
        VERIFIED BY: tests/test_bug_002_concurrency.py
        """
        try:
            # For WAL mode, we need a persistent connection
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            try:
                # Enable WAL mode for concurrent reads/writes (BUG-002)
                conn.execute("PRAGMA journal_mode=WAL")
                # NORMAL synchronous is safe with WAL and faster than FULL
                conn.execute("PRAGMA synchronous=NORMAL")
                # Busy timeout prevents "database is locked" errors (5 seconds)
                conn.execute("PRAGMA busy_timeout=5000")
                # Create schema
                conn.executescript(SCHEMA)
                conn.commit()
            finally:
                conn.close()
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
        
        # Ensure embedding is float32 and normalized
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # Normalize if not already (L2 norm)
        norm = np.linalg.norm(embedding)
        if norm > 0 and abs(norm - 1.0) > 1e-6:
            embedding = embedding / norm
        
        embedding_blob = embedding.tobytes()
        created_at = int(datetime.now().timestamp())
        
        try:
            conn = self._get_connection()
            # Delete existing snapshot and its history
            conn.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
            conn.execute("DELETE FROM similarity_history WHERE snapshot_id = ?", (snapshot_id,))
            
            # Insert new snapshot
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
            
            return [row['similarity'] for row in rows]
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
            
            return [
                (row['behavior_id'], row['count'], row['last_run'])
                for row in rows
            ]
        except sqlite3.Error as e:
            raise StorageError(f"Failed to get behavior summary: {e}")
