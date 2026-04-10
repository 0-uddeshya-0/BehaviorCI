"""Test for TASK 1 (v0.2): Model Mismatch Hard Fail.

TASK 1: Comparing embeddings from different models is mathematically invalid
because they exist in different vector spaces. This should raise a hard error.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))

import numpy as np
import pytest

from behaviorci.comparator import Comparator
from behaviorci.exceptions import ModelMismatchError
from behaviorci.storage import get_storage, reset_all_storage


class MockEmbedderA:
    """Mock embedder with model_name 'model-A'."""

    EMBEDDING_DIM = 384
    model_name = "model-A"

    def embed_single(self, text: str) -> np.ndarray:
        # Deterministic embedding based on text hash
        import hashlib

        hash_bytes = hashlib.sha256(text.encode()).digest()
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        for i in range(self.EMBEDDING_DIM):
            hash_idx = i % len(hash_bytes)
            embedding[i] = (hash_bytes[hash_idx] / 128.0) - 1.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        similarity = float(np.dot(a, b))
        return max(-1.0, min(1.0, similarity))


class MockEmbedderB:
    """Mock embedder with model_name 'model-B'."""

    EMBEDDING_DIM = 384
    model_name = "model-B"

    def embed_single(self, text: str) -> np.ndarray:
        # Different deterministic embedding
        import hashlib

        hash_bytes = hashlib.sha256((text + "_B").encode()).digest()
        embedding = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)
        for i in range(self.EMBEDDING_DIM):
            hash_idx = i % len(hash_bytes)
            embedding[i] = (hash_bytes[hash_idx] / 128.0) - 1.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def compute_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        similarity = float(np.dot(a, b))
        return max(-1.0, min(1.0, similarity))


def test_model_mismatch_raises_error():
    """Verify that comparing snapshots from different models raises ModelMismatchError.

    WHY: TASK 1 (v0.2) - Comparing embeddings from different models is
    mathematically invalid because they exist in different vector spaces.

    VERIFIED: Comparator.compare() raises ModelMismatchError when models differ.
    """
    reset_all_storage()
    storage = get_storage(":memory:")

    # Record snapshot with model-A
    embedder_a = MockEmbedderA()
    comparator_a = Comparator(storage, embedder_a)

    snapshot_id = comparator_a.record_snapshot(
        behavior_id="test_behavior",
        input_json='{"args": [], "kwargs": {}}',
        output_text="test output",
    )

    # Verify snapshot was recorded with model-A
    snapshot = storage.get_snapshot(snapshot_id)
    assert snapshot.model_name == "model-A"

    # Try to compare with model-B - should raise ModelMismatchError
    embedder_b = MockEmbedderB()
    comparator_b = Comparator(storage, embedder_b)

    with pytest.raises(ModelMismatchError) as exc_info:
        comparator_b.compare(
            behavior_id="test_behavior",
            input_json='{"args": [], "kwargs": {}}',
            output_text="test output",
            base_threshold=0.85,
        )

    # Verify error message contains both model names
    error_message = str(exc_info.value)
    assert (
        "model-A" in error_message
    ), f"Error should mention stored model 'model-A': {error_message}"
    assert (
        "model-B" in error_message
    ), f"Error should mention current model 'model-B': {error_message}"
    assert (
        "incomparable" in error_message.lower() or "different vector" in error_message.lower()
    ), f"Error should explain why models are incomparable: {error_message}"


def test_same_model_no_error():
    """Verify that comparing snapshots with the same model works correctly.

    WHY: Same model should allow normal comparison.
    """
    reset_all_storage()
    storage = get_storage(":memory:")

    # Record and compare with same embedder
    embedder = MockEmbedderA()
    comparator = Comparator(storage, embedder)

    comparator.record_snapshot(
        behavior_id="test_same_model",
        input_json='{"args": [], "kwargs": {}}',
        output_text="test output",
    )

    # Should not raise any error
    result = comparator.compare(
        behavior_id="test_same_model",
        input_json='{"args": [], "kwargs": {}}',
        output_text="test output",
        base_threshold=0.85,
    )

    assert result.passed is True
    assert result.similarity > 0.99  # Same text should have very high similarity


def test_model_mismatch_error_details():
    """Verify ModelMismatchError contains correct details.

    WHY: Error details help users understand and fix the issue.
    """
    reset_all_storage()
    storage = get_storage(":memory:")

    # Record with model-A
    embedder_a = MockEmbedderA()
    comparator_a = Comparator(storage, embedder_a)
    comparator_a.record_snapshot(behavior_id="test_details", input_json="{}", output_text="test")

    # Compare with model-B
    embedder_b = MockEmbedderB()
    comparator_b = Comparator(storage, embedder_b)

    with pytest.raises(ModelMismatchError) as exc_info:
        comparator_b.compare(
            behavior_id="test_details", input_json="{}", output_text="test", base_threshold=0.85
        )

    error = exc_info.value
    assert error.stored_model == "model-A"
    assert error.current_model == "model-B"
    assert "--behaviorci-update" in error.message or "re-record" in error.message.lower()
    assert (
        "--behaviorci-model" in error.message
        or "specify the original model" in error.message.lower()
    )
