"""Shared fixtures for the BehaviorCI test suite."""

import pytest


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset cached storage and embedder state around every test.

    BehaviorCI caches one storage instance per database path and one embedder
    per model, so tests must start from a clean slate to stay isolated.
    """
    from behaviorci.embedder import reset_embedder
    from behaviorci.storage import reset_all_storage

    reset_all_storage()
    reset_embedder()
    yield
    reset_all_storage()
    reset_embedder()
