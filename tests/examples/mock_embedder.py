"""Mock embedder for testing without loading sentence-transformers model."""

import hashlib
from typing import List, Union

import numpy as np


class MockEmbedder:
    """Deterministic mock embedder for fast testing.

    This embedder generates deterministic embeddings based on text hash,
    allowing tests to run without loading the full sentence-transformers model.
    """

    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = "mock-model"):
        """Initialize mock embedder."""
        self.model_name = model_name
        self._embedding_dim = self.EMBEDDING_DIM

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Compute deterministic embeddings for text(s).

        Args:
            texts: Single text or list of texts

        Returns:
            Normalized embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        embeddings = []
        for text in texts:
            # Generate deterministic embedding from text hash
            hash_bytes = hashlib.sha256(text.encode()).digest()

            # Create embedding vector from hash
            embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            for i in range(self.EMBEDDING_DIM):
                # Use bytes from hash to fill embedding
                hash_idx = i % len(hash_bytes)
                embedding[i] = (hash_bytes[hash_idx] / 128.0) - 1.0  # Range: -1 to 1

            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            embeddings.append(embedding)

        result = np.array(embeddings, dtype=np.float32)
        if single_input:
            return result[0]
        return result

    def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        return self.embed(text)

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = float(np.dot(a, b))
        return max(-1.0, min(1.0, similarity))

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return True


# Global mock embedder instance
_mock_embedder = MockEmbedder()


def get_mock_embedder() -> MockEmbedder:
    """Get global mock embedder instance."""
    return _mock_embedder


def reset_mock_embedder() -> None:
    """Reset global mock embedder."""
    global _mock_embedder
    _mock_embedder = MockEmbedder()
