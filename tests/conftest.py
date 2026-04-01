"""Pytest configuration for BehaviorCI tests."""

import pytest
import sys
import os

# Add tests/examples to path for mock_embedder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons before each test.
    
    WHY: BUG-003 - Singleton state must be reset between tests
    to ensure test isolation.
    """
    from behaviorci.storage import reset_all_storage
    from behaviorci.embedder import reset_embedder
    
    # Reset before test
    reset_all_storage()
    reset_embedder()
    
    yield
    
    # Reset after test
    reset_all_storage()
    reset_embedder()
