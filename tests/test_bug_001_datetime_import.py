"""Test for BUG-001: datetime import location.

BUG-001: datetime import was at bottom of storage.py, causing NameError
when save_snapshot() was called (uses datetime at line 110).

FIX: Moved `from datetime import datetime` to top of file.
"""

import numpy as np
import tempfile
import os


def test_datetime_import_at_module_load():
    """Verify datetime is available when Storage module is loaded.
    
    WHY: This test verifies BUG-001 is fixed. Previously, datetime was imported
    at the bottom of the file, causing NameError on first use.
    
    VERIFIED: datetime import moved to top of storage.py
    """
    # This import should succeed without NameError
    from behaviorci.storage import Storage
    
    # Creating a Storage instance should work
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = Storage(db_path)
        
        # save_snapshot uses datetime.now() - this should not raise NameError
        storage.save_snapshot(
            behavior_id="test",
            input_json='{}',
            output_text="test output",
            embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            model_name="test-model",
            git_commit=None
        )
        
        # Verify the snapshot was saved with a valid timestamp
        snapshot = storage.get_all_snapshots_for_behavior("test")[0]
        assert snapshot.created_at > 0  # Valid Unix timestamp


def test_datetime_import_in_record_similarity():
    """Verify datetime works in record_similarity too.
    
    WHY: record_similarity also uses datetime at line 191.
    """
    from behaviorci.storage import Storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = Storage(db_path)
        
        # First save a snapshot
        storage.save_snapshot(
            behavior_id="test",
            input_json='{}',
            output_text="test",
            embedding=np.array([0.1, 0.2], dtype=np.float32),
            model_name="test"
        )
        
        # Get the snapshot ID
        snapshot = storage.get_all_snapshots_for_behavior("test")[0]
        
        # record_similarity uses datetime - should not raise NameError
        storage.record_similarity(snapshot.id, 0.95)
        
        # Verify it was recorded
        history = storage.get_similarity_history(snapshot.id)
        assert len(history) == 1
