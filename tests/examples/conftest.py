import pytest
from unittest.mock import patch, MagicMock
import numpy as np

@pytest.fixture(autouse=True)
def mock_embedder():
    """Mock embedder for all tests to avoid model download."""
    mock = MagicMock()
    mock.model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    
    # Return predictable embeddings
    def mock_embed(texts):
        if isinstance(texts, str):
            # Single text - return normalized vector
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = hash(texts) % 1000 / 1000.0
            # Normalize
            vec = vec / np.linalg.norm(vec)
            return vec
        else:
            # Multiple texts
            return np.array([mock_embed(t) for t in texts])
    
    mock.embed = mock_embed
    mock.embed_single = mock_embed
    
    # Mock similarity
    def mock_similarity(a, b):
        return float(np.dot(a, b))
    
    mock.compute_similarity = mock_similarity
    
    # Patch the cache directly
    with patch('behaviorci.embedder._embedder_cache', 
              {'sentence-transformers/all-MiniLM-L6-v2': mock}):
        yield mock
    
    # Cleanup after test
    from behaviorci.embedder import reset_embedder
    reset_embedder()
