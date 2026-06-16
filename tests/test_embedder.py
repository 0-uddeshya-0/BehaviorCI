"""Tests for embedder selection, injection, and the local model wrapper."""

import threading

import numpy as np
import pytest

from behaviorci.embedder import get_embedder, reset_embedder, set_embedder
from tests.support import MockEmbedder


class TestSelection:
    def test_same_model_returns_same_instance(self):
        reset_embedder()
        assert get_embedder() is get_embedder()
        assert get_embedder() is get_embedder("sentence-transformers/all-MiniLM-L6-v2")

    def test_reset_creates_new_instance(self):
        reset_embedder()
        first = get_embedder()
        reset_embedder()
        assert get_embedder() is not first

    def test_thread_safe_singleton(self):
        reset_embedder()
        seen = []

        def grab():
            seen.append(id(get_embedder()))

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(seen)) == 1


class TestInjection:
    def test_injected_embedder_takes_priority(self):
        reset_embedder()
        mock = MockEmbedder("injected")
        set_embedder(mock)
        try:
            assert get_embedder() is mock
            assert get_embedder("some-other-model") is mock
        finally:
            reset_embedder()

    def test_reset_clears_injection(self):
        set_embedder(MockEmbedder())
        reset_embedder()
        assert get_embedder() is not None  # falls back to a real Embedder lazily


class TestSimilarity:
    def test_identical_text_similarity_is_one(self):
        embedder = MockEmbedder()
        emb = embedder.embed_single("hello world")
        assert embedder.compute_similarity(emb, emb) == pytest.approx(1.0)

    def test_similarity_is_clamped(self):
        embedder = MockEmbedder()
        a = np.array([1.0, 0.0], dtype=np.float32)
        # Deliberately un-normalized vectors would exceed 1.0 without clamping.
        b = np.array([2.0, 0.0], dtype=np.float32)
        assert embedder.compute_similarity(a, b) == 1.0


class TestLocalModel:
    """Exercises the real sentence-transformers model when it is installed."""

    @pytest.fixture(autouse=True)
    def _require_local(self):
        pytest.importorskip("sentence_transformers")
        reset_embedder()

    def test_embed_shape_and_dtype(self):
        from behaviorci.embedder import Embedder

        emb = Embedder().embed_single("hello world")
        assert emb.dtype == np.float32
        assert emb.shape == (384,)
        assert abs(np.linalg.norm(emb) - 1.0) < 1e-5

    def test_similar_texts_score_high(self):
        from behaviorci.embedder import Embedder

        embedder = Embedder()
        a = embedder.embed_single("I love machine learning")
        b = embedder.embed_single("I enjoy machine learning")
        assert embedder.compute_similarity(a, b) > 0.9

    def test_different_texts_score_lower(self):
        from behaviorci.embedder import Embedder

        embedder = Embedder()
        a = embedder.embed_single("machine learning")
        b = embedder.embed_single("pizza recipe")
        assert embedder.compute_similarity(a, b) < 0.8
