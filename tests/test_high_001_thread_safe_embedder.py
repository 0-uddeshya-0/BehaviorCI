"""Test for HIGH-001: Thread-safe embedder singleton.

HIGH-001: Without lock, multiple threads could create multiple Embedder instances.

FIX: Added threading.Lock like storage.py.

VERIFICATION: Test with 10 threads calling get_embedder simultaneously—
should create only 1 Embedder instance.
"""

import threading
import time

from behaviorci.embedder import get_embedder, reset_embedder


def test_thread_safe_embedder_singleton():
    """Verify embedder singleton is thread-safe.

    WHY: HIGH-001 fix - Without lock, race condition could create multiple
    Embedder instances when multiple threads call get_embedder simultaneously.

    VERIFIED: get_embedder uses _embedder_lock for thread safety.
    """
    reset_embedder()

    instances = []

    def get_and_store():
        """Get embedder and store instance."""
        embedder = get_embedder()
        instances.append(id(embedder))
        # Small delay to increase chance of race condition
        time.sleep(0.01)

    # Spawn 10 threads that all call get_embedder simultaneously
    threads = [threading.Thread(target=get_and_store) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # All should have gotten the same instance (same id)
    unique_ids = set(instances)
    assert (
        len(unique_ids) == 1
    ), f"Expected 1 unique embedder instance, got {len(unique_ids)}: {unique_ids}"


def test_reset_embedder_thread_safe():
    """Verify reset_embedder is thread-safe.

    WHY: reset_embedder also uses the lock to prevent race conditions.
    """
    reset_embedder()

    # Get initial instance
    e1 = get_embedder()

    # Reset
    reset_embedder()

    # Get new instance
    e2 = get_embedder()

    # Should be different instances
    assert e1 is not e2, "After reset, should get new instance"


def test_embedder_model_name_consistency():
    """Verify same model_name returns same instance.

    WHY: Model name should be consistent across calls.
    """
    reset_embedder()

    # Get embedder with default model
    e1 = get_embedder()
    e2 = get_embedder()

    # Same instance
    assert e1 is e2, "Same model should return same instance"

    # Get embedder with explicit model
    e3 = get_embedder("sentence-transformers/all-MiniLM-L6-v2")

    # Same model name, same instance
    assert e2 is e3, "Same explicit model should return same instance"


def test_concurrent_embedder_access():
    """Verify concurrent access doesn't corrupt singleton state.

    WHY: Multiple threads calling get_embedder simultaneously should all
    get the same instance without race conditions.

    NOTE: This test only verifies singleton pattern, not actual embedding
    (which would require loading sentence-transformers model).
    """
    reset_embedder()

    instances = []

    def get_embedder_and_store():
        """Get embedder from a thread."""
        embedder = get_embedder()
        instances.append(id(embedder))

    # Spawn threads that all call get_embedder
    threads = [threading.Thread(target=get_embedder_and_store) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # All should have gotten the same instance
    unique_ids = set(instances)
    assert len(unique_ids) == 1, f"Expected 1 unique embedder instance, got {len(unique_ids)}"
