"""Test for BUG-003: Connection Leak - Resource Exhaustion.

BUG-003: Each Storage() instantiation opens a new SQLite connection.
With 1000 tests, this hits OS file descriptor limit.

FIX: Implement Storage as singleton per database path with:
- Module-level _storage_instances dict
- get_storage() function for singleton access
- Thread-local connections for thread safety
"""

import tempfile
import os
import threading
import numpy as np
import pytest


def test_storage_singleton_same_path():
    """Verify Storage is singleton for same path.
    
    WHY: This is the core fix for BUG-003. Multiple calls to get_storage()
    with the same path should return the same instance.
    
    VERIFIED: get_storage() returns cached instance from _storage_instances
    """
    from behaviorci.storage import get_storage, reset_storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        # Get storage multiple times
        s1 = get_storage(db_path)
        s2 = get_storage(db_path)
        s3 = get_storage(db_path)
        
        # Should be the same instance
        assert s1 is s2, "Same path should return same instance"
        assert s2 is s3, "Same path should return same instance"
        
        # Cleanup
        reset_storage(db_path)


def test_storage_singleton_different_paths():
    """Verify different paths get different instances.
    
    WHY: Each database should have its own Storage instance.
    
    VERIFIED: get_storage() uses path as dict key
    """
    from behaviorci.storage import get_storage, reset_storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path1 = os.path.join(tmpdir, "test1.db")
        db_path2 = os.path.join(tmpdir, "test2.db")
        
        s1 = get_storage(db_path1)
        s2 = get_storage(db_path2)
        
        # Should be different instances
        assert s1 is not s2, "Different paths should return different instances"
        
        # Cleanup
        reset_storage(db_path1)
        reset_storage(db_path2)


def test_storage_singleton_default_path():
    """Verify default path uses singleton.
    
    WHY: Default path (.behaviorci/behaviorci.db) should also be singleton.
    """
    from behaviorci.storage import get_storage, reset_all_storage
    
    # Use default path (None)
    s1 = get_storage(None)
    s2 = get_storage(None)
    
    # Should be the same instance
    assert s1 is s2, "Default path should return same instance"
    
    # Cleanup
    reset_all_storage()


def test_thread_local_connections():
    """Verify each thread gets its own connection.
    
    WHY: SQLite connections are NOT thread-safe. Each thread must have
    its own connection to avoid corruption.
    
    VERIFIED: _get_connection() uses threading.local()
    """
    from behaviorci.storage import get_storage, reset_storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        storage = get_storage(db_path)
        
        results = []
        
        def write_from_thread(thread_id: int):
            """Write from a thread."""
            for i in range(5):
                storage.save_snapshot(
                    behavior_id=f"thread_{thread_id}",
                    input_json=f'{{"idx": {i}}}',
                    output_text=f"output {i}",
                    embedding=np.array([0.1, 0.2], dtype=np.float32),
                    model_name="test"
                )
            results.append(thread_id)
        
        # Spawn multiple threads
        threads = [
            threading.Thread(target=write_from_thread, args=(i,))
            for i in range(4)
        ]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # All threads should complete
        assert len(results) == 4, "All threads should complete"
        
        # Verify 20 snapshots (4 threads * 5 writes each)
        stats = storage.get_stats()
        assert stats['snapshots'] == 20, f"Expected 20 snapshots, got {stats['snapshots']}"
        
        # Cleanup
        reset_storage(db_path)


def test_reset_storage():
    """Verify reset_storage removes singleton instance.
    
    WHY: Tests need to reset singleton state for isolation.
    
    VERIFIED: reset_storage() removes from _storage_instances
    """
    from behaviorci.storage import get_storage, reset_storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        # Get storage
        s1 = get_storage(db_path)
        
        # Reset
        reset_storage(db_path)
        
        # Get again - should be new instance
        s2 = get_storage(db_path)
        
        # Should be different instances after reset
        assert s1 is not s2, "After reset, should get new instance"
        
        # Cleanup
        reset_storage(db_path)


def test_reset_all_storage():
    """Verify reset_all_storage clears all instances.
    
    WHY: Nuclear option for test cleanup.
    """
    from behaviorci.storage import get_storage, reset_all_storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path1 = os.path.join(tmpdir, "test1.db")
        db_path2 = os.path.join(tmpdir, "test2.db")
        
        # Get storages
        s1 = get_storage(db_path1)
        s2 = get_storage(db_path2)
        
        # Reset all
        reset_all_storage()
        
        # Get again - should be new instances
        s3 = get_storage(db_path1)
        s4 = get_storage(db_path2)
        
        # Should be different instances
        assert s1 is not s3, "After reset_all, should get new instance"
        assert s2 is not s4, "After reset_all, should get new instance"
        
        # Cleanup
        reset_all_storage()


def test_no_connection_leak():
    """Verify singleton prevents connection exhaustion.
    
    WHY: This is the main regression test for BUG-003. Without singleton,
    1000 get_storage() calls would create 1000 connections.
    
    VERIFIED: get_storage() returns cached instance
    """
    from behaviorci.storage import get_storage, reset_storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        # Get storage many times
        instances = [get_storage(db_path) for _ in range(100)]
        
        # All should be the same instance
        first = instances[0]
        for instance in instances[1:]:
            assert instance is first, "All calls should return same instance"
        
        # Cleanup
        reset_storage(db_path)


def test_concurrent_singleton_access():
    """Verify singleton is thread-safe.
    
    WHY: Multiple threads may call get_storage() simultaneously.
    
    VERIFIED: get_storage() uses _storage_lock
    """
    from behaviorci.storage import get_storage, reset_storage
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        
        instances = []
        
        def get_and_store():
            s = get_storage(db_path)
            instances.append(s)
        
        # Spawn threads that all call get_storage
        threads = [threading.Thread(target=get_and_store) for _ in range(10)]
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        # All should have gotten the same instance
        first = instances[0]
        for instance in instances[1:]:
            assert instance is first, "All threads should get same instance"
        
        # Cleanup
        reset_storage(db_path)
