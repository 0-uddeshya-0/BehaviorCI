"""SQLite storage layer for BehaviorCI snapshots and history."""

import sqlite3
import hashlib
import json
import os
from pathlib import Path
from typing import Optional, List
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


def compute_snapshot_id(behavior_id: str, input_json: str) -> str:
    """Compute unique snapshot ID from behavior_id and canonical input JSON."""
    data = f"{behavior_id}:{input_json}"
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class Storage:
    """SQLite storage for behavioral snapshots."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage with database path.
        
        Args:
            db_path: Path to SQLite database. Defaults to .behaviorci/behaviorci.db
        """
        if db_path is None:
            db_path = os.path.join('.behaviorci', 'behaviorci.db')
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        try:
            with self._get_connection() as conn:
                conn.executescript(SCHEMA)
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
            with self._get_connection() as conn:
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
            with self._get_connection() as conn:
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
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO similarity_history (snapshot_id, similarity, timestamp)
                    VALUES (?, ?, ?)
                    """,
                    (snapshot_id, similarity, timestamp)
                )
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
            with self._get_connection() as conn:
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
            with self._get_connection() as conn:
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
            with self._get_connection() as conn:
                cursor = conn.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))
                conn.execute("DELETE FROM similarity_history WHERE snapshot_id = ?", (snapshot_id,))
                return cursor.rowcount > 0
        except sqlite3.Error as e:
            raise StorageError(f"Failed to delete snapshot: {e}")
    
    def clear_all(self) -> None:
        """Clear all snapshots and history. USE WITH CAUTION."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM similarity_history")
                conn.execute("DELETE FROM snapshots")
        except sqlite3.Error as e:
            raise StorageError(f"Failed to clear database: {e}")
    
    def get_stats(self) -> dict:
        """Get database statistics.
        
        Returns:
            Dictionary with snapshot count and history count
        """
        try:
            with self._get_connection() as conn:
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


# Import at bottom to avoid circular import
from datetime import datetime