"""Shared helpers for the test suite.

``MockEmbedder`` is a deterministic, dependency-free stand-in for the local
sentence-transformers model. The same text always maps to the same unit vector
(even across processes), which keeps record/compare tests fast and stable.
"""

import hashlib
from typing import List, Union

import numpy as np


class MockEmbedder:
    """Deterministic embedder for tests; no model download required."""

    EMBEDDING_DIM = 384

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self._embedding_dim = self.EMBEDDING_DIM

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]  # type: ignore[list-item]

        embeddings = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
            for i in range(self.EMBEDDING_DIM):
                vec[i] = (digest[i % len(digest)] / 128.0) - 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)

        result = np.array(embeddings, dtype=np.float32)
        return result[0] if single_input else result

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed(text)

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return max(-1.0, min(1.0, float(np.dot(a, b))))

    def get_dimension(self) -> int:
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return True
