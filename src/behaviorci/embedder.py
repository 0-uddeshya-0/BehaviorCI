"""Embedding computation using sentence-transformers."""

import numpy as np
from typing import List, Union
import warnings

from .exceptions import EmbeddingError


class Embedder:
    """Local embedding model wrapper using sentence-transformers."""
    
    DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = None):
        """Initialize embedder with specified model.
        
        Args:
            model_name: Name of sentence-transformers model. 
                       Defaults to all-MiniLM-L6-v2 (384 dim)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._embedding_dim = None
    
    def _load_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                self._embedding_dim = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load model '{self.model_name}': {e}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Compute embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            
        Returns:
            Normalized embedding vectors as float32 numpy array
            Shape: (embedding_dim,) for single text, (n, embedding_dim) for list
        """
        self._load_model()
        
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        if not texts:
            raise EmbeddingError("Cannot embed empty text list")
        
        try:
            # Compute embeddings
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization
            )
            
            # Ensure float32
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            # Return single vector if single input
            if single_input:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Embedding computation failed: {e}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector (384-dim float32)
        """
        return self.embed(text)
    
    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings.
        
        Args:
            a: First embedding vector (normalized)
            b: Second embedding vector (normalized)
            
        Returns:
            Cosine similarity score (-1 to 1, typically 0 to 1 for similar texts)
        """
        # Both vectors should be normalized, so dot product = cosine similarity
        similarity = float(np.dot(a, b))
        
        # Clamp to valid range due to floating point errors
        similarity = max(-1.0, min(1.0, similarity))
        
        return similarity
    
    def get_dimension(self) -> int:
        """Get embedding dimension.
        
        Returns:
            Dimension of embedding vectors
        """
        self._load_model()
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


# Global embedder instance for reuse
_global_embedder: Embedder = None


def get_embedder(model_name: str = None) -> Embedder:
    """Get or create global embedder instance.
    
    Args:
        model_name: Model name (uses default if not specified)
        
    Returns:
        Embedder instance
    """
    global _global_embedder
    if _global_embedder is None or (model_name and _global_embedder.model_name != model_name):
        _global_embedder = Embedder(model_name)
    return _global_embedder


def reset_embedder() -> None:
    """Reset global embedder (useful for testing)."""
    global _global_embedder
    _global_embedder = None