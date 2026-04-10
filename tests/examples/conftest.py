from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def mock_embedder():
    """Mock embedder for all tests to avoid model download."""
    mock = MagicMock()
    mock.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    # Return predictable embeddings
    def mock_embed(texts):
        if isinstance(texts, str):
            vec = np.zeros(384, dtype=np.float32)
            vec[0] = hash(texts) % 1000 / 1000.0
            vec = vec / np.linalg.norm(vec)
            return vec
        else:
            return np.array([mock_embed(t) for t in texts])

    mock.embed = mock_embed
    mock.embed_single = mock_embed

    def mock_similarity(a, b):
        return float(np.dot(a, b))

    mock.compute_similarity = mock_similarity

    # FIX: Patch _embedder_cache instead of _global_embedder
    with patch(
        "behaviorci.embedder._embedder_cache", {"sentence-transformers/all-MiniLM-L6-v2": mock}
    ):
        yield mock

    # Cleanup
    from behaviorci.embedder import reset_embedder

    reset_embedder()
