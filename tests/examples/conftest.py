"""Pytest configuration for BehaviorCI example tests."""

import pytest
from unittest.mock import patch


@pytest.fixture(scope="session", autouse=True)
def mock_embedder():
    """Replace real embedder with mock for fast testing."""
    from mock_embedder import MockEmbedder
    
    mock = MockEmbedder()
    
    # Patch the embedder module
    with patch('behaviorci.embedder._global_embedder', mock):
        with patch('behaviorci.embedder.Embedder', MockEmbedder):
            with patch('behaviorci.comparator.get_embedder', lambda: mock):
                yield mock