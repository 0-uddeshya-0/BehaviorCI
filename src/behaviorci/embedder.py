"""Embedding computation using sentence-transformers or injected APIs."""

import numpy as np
import threading
from typing import List, Union, Dict, Optional, Any
from abc import ABC, abstractmethod
import warnings

from .exceptions import EmbeddingError

_embedder_lock = threading.Lock()
_embedder_cache: Dict[str, "BaseEmbedder"] = {}
_injected_embedder: Optional["BaseEmbedder"] = None

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class BaseEmbedder(ABC):
    """Abstract Base Class for all Embedders."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        pass

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        similarity = float(np.dot(a, b))
        similarity = max(-1.0, min(1.0, similarity))
        return similarity


class Embedder(BaseEmbedder):
    """Concrete local embedding model wrapper."""

    DEFAULT_MODEL = DEFAULT_MODEL_NAME
    EMBEDDING_DIM = 384

    def __init__(self, model_name: Optional[str] = None) -> None:
        super().__init__(model_name or self.DEFAULT_MODEL)
        self._model: Any = None
        self._embedding_dim: Optional[int] = None

    def _load_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
                # Suppress the FutureWarning by dynamically fetching the new method if available
                get_dim = getattr(self._model, "get_embedding_dimension", None)
                if get_dim is None:
                    get_dim = self._model.get_sentence_embedding_dimension
                self._embedding_dim = get_dim()
            except ImportError:
                raise EmbeddingError(
                    "sentence-transformers is not installed. "
                    "Install with: pip install behaviorci[local] "
                    "or inject a custom API embedder via set_embedder()."
                )
            except Exception as e:
                raise EmbeddingError(f"Failed to load model '{self.model_name}': {e}")

    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        self._load_model()

        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False

        if not texts:
            raise EmbeddingError("Cannot embed empty text list")

        try:
            assert self._model is not None
            embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32)

            if single_input:
                return embeddings[0]  # type: ignore[no-any-return]
            return embeddings  # type: ignore[no-any-return]

        except Exception as e:
            raise EmbeddingError(f"Embedding computation failed: {e}")

    def embed_single(self, text: str) -> np.ndarray:
        # MYPY FIX: Ensure strict return checking ignores numpy inheritance
        return self.embed(text)  # type: ignore[no-any-return]

    def get_dimension(self) -> int:
        self._load_model()
        assert self._embedding_dim is not None
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


def set_embedder(embedder: BaseEmbedder) -> None:
    global _injected_embedder
    _injected_embedder = embedder


def get_embedder(model_name: Optional[str] = None) -> BaseEmbedder:
    if _injected_embedder is not None:
        return _injected_embedder

    model_name = model_name or DEFAULT_MODEL_NAME

    with _embedder_lock:
        if model_name not in _embedder_cache:
            _embedder_cache[model_name] = Embedder(model_name)
        return _embedder_cache[model_name]


def reset_embedder(model_name: Optional[str] = None) -> None:
    global _injected_embedder
    with _embedder_lock:
        _injected_embedder = None
        if model_name:
            _embedder_cache.pop(model_name, None)
        else:
            _embedder_cache.clear()
