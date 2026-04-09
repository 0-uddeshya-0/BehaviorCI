"""Embedding computation using sentence-transformers or injected APIs."""

import numpy as np
import threading
from typing import List, Union, Dict, Optional
from abc import ABC, abstractmethod

from .exceptions import EmbeddingError

# HIGH-001 FIX: Thread lock for embedder singleton cache
_embedder_lock = threading.Lock()
_embedder_cache: Dict[str, 'Embedder'] = {}
_injected_embedder: Optional['Embedder'] = None

DEFAULT_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'


class Embedder(ABC):
    """Abstract base class for all embedding providers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        
    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Compute embedding for a single text.
        
        MUST return a normalized float32 numpy array.
        """
        pass
        
    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two normalized embeddings."""
        similarity = float(np.dot(a, b))
        similarity = max(-1.0, min(1.0, similarity))
        return similarity


class LocalEmbedder(Embedder):
    """Local embedding model wrapper using sentence-transformers."""
    
    DEFAULT_MODEL = DEFAULT_MODEL_NAME
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: str = None):
        super().__init__(model_name or self.DEFAULT_MODEL)
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
                    "sentence-transformers is not installed. "
                    "Install with: pip install behaviorci[local] "
                    "or inject a custom API embedder via set_embedder()."
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load model '{self.model_name}': {e}")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Compute embeddings for text(s)."""
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
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._embedding_dim
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


def set_embedder(embedder: Embedder) -> None:
    """Inject a custom embedder globally (e.g., OpenAI, Gemini, Cohere).
    
    This overrides the default local sentence-transformers embedder.
    Call this in your conftest.py before tests run.
    """
    global _injected_embedder
    _injected_embedder = embedder


def get_embedder(model_name: str = None) -> Embedder:
    """Get the injected embedder or create a cached local instance.
    
    Args:
        model_name: Model name (uses default if not specified)
        
    Returns:
        Embedder instance
    """
    if _injected_embedder is not None:
        return _injected_embedder
        
    model_name = model_name or DEFAULT_MODEL_NAME
    
    with _embedder_lock:
        if model_name not in _embedder_cache:
            _embedder_cache[model_name] = LocalEmbedder(model_name)
        return _embedder_cache[model_name]


def reset_embedder(model_name: str = None) -> None:
    """Reset embedder cache and injected embedder (useful for testing)."""
    global _injected_embedder
    with _embedder_lock:
        _injected_embedder = None
        if model_name:
            _embedder_cache.pop(model_name, None)
        else:
            _embedder_cache.clear()
