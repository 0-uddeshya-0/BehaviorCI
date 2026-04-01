"""Embedding computation using sentence-transformers."""

import numpy as np
import threading
from typing import List, Union, Dict
import warnings

from .exceptions import EmbeddingError


# HIGH-001 FIX: Thread lock for embedder singleton cache
# WHY: Multiple threads could create multiple Embedder instances
# APPROACH: Dictionary cache with thread-safe access (supports multiple models)
# VERIFIED BY: tests/test_high_001_thread_safe_embedder.py
_embedder_lock = threading.Lock()
_embedder_cache: Dict[str, 'Embedder'] = {}
DEFAULT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


class Embedder:
    """Local embedding model wrapper using sentence-transformers."""
    
    DEFAULT_MODEL = DEFAULT_MODEL_NAME
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
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)
            
            if single_input:
                return embeddings[0]
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Embedding computation failed: {e}")
    
    def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text."""
        return self.embed(text)
    
    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings."""
        similarity = float(np.dot(a, b))
        similarity = max(-1.0, min(1.0, similarity))
        return similarity
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


def get_embedder(model_name: str = None) -> Embedder:
    """Get or create cached embedder instance for specific model.
    
    HIGH-001 FIX: Dictionary cache ensures same model returns same instance.
    
    Args:
        model_name: Model name (uses default if not specified)
        
    Returns:
        Embedder instance (cached per model)
    """
    model_name = model_name or DEFAULT_MODEL_NAME
    
    with _embedder_lock:
        if model_name not in _embedder_cache:
            _embedder_cache[model_name] = Embedder(model_name)
        return _embedder_cache[model_name]


def reset_embedder(model_name: str = None) -> None:
    """Reset embedder cache (useful for testing)."""
    with _embedder_lock:
        if model_name:
            _embedder_cache.pop(model_name, None)
        else:
            _embedder_cache.clear()
