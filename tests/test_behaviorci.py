"""Self-tests for BehaviorCI framework.

These tests validate the core functionality of BehaviorCI itself.
"""

import json
import pytest
import numpy as np
from datetime import datetime

import behaviorci
from behaviorci import behavior
from behaviorci.api import serialize_inputs, get_behavior_config, is_behavior_test
from behaviorci.storage import Storage, compute_snapshot_id
from behaviorci.embedder import Embedder, get_embedder, reset_embedder
from behaviorci.comparator import Comparator
from behaviorci.exceptions import (
    BehaviorCIError,
    SerializationError,
    ConfigurationError,
)


# ============================================================================
# Serialization Tests
# ============================================================================

class TestSerialization:
    """Test input serialization (STRICT - no default=str)."""
    
    def test_serializes_simple_types(self):
        """Test serialization of simple JSON types."""
        result = serialize_inputs(("hello", 42, True), {"key": "value"})
        data = json.loads(result)
        assert data["args"] == ["hello", 42, True]
        assert data["kwargs"] == {"key": "value"}
    
    def test_serializes_nested_dicts(self):
        """Test serialization of nested structures."""
        result = serialize_inputs(
            ({"nested": [1, 2, 3]},),
            {"outer": {"inner": "value"}}
        )
        data = json.loads(result)
        assert data["args"][0]["nested"] == [1, 2, 3]
    
    def test_fails_on_datetime(self):
        """CRITICAL: datetime must raise SerializationError (not silent conversion)."""
        with pytest.raises(SerializationError) as exc_info:
            serialize_inputs((datetime.now(),), {})
        
        assert "datetime" in str(exc_info.value).lower() or "not json-serializable" in str(exc_info.value).lower()
    
    def test_fails_on_custom_objects(self):
        """Custom objects must raise SerializationError."""
        class CustomClass:
            pass
        
        with pytest.raises(SerializationError):
            serialize_inputs((CustomClass(),), {})
    
    def test_canonical_form(self):
        """Same inputs produce same JSON (sorted keys)."""
        result1 = serialize_inputs(("a", "b"), {"z": 1, "a": 2})
        result2 = serialize_inputs(("a", "b"), {"a": 2, "z": 1})
        assert result1 == result2


# ============================================================================
# Decorator Tests
# ============================================================================

class TestDecorator:
    """Test @behavior decorator and return value capture."""
    
    def test_decorator_attaches_config(self):
        """Decorator attaches configuration to function."""
        @behavior("test_behavior", threshold=0.9)
        def my_test():
            return "output"
        
        config = get_behavior_config(my_test)
        assert config is not None
        assert config.behavior_id == "test_behavior"
        assert config.threshold == 0.9
    
    def test_is_behavior_test_detects_decorated(self):
        """is_behavior_test returns True for decorated functions."""
        @behavior("test")
        def my_test():
            return "output"
        
        assert is_behavior_test(my_test) is True
    
    def test_is_behavior_test_false_for_undecorated(self):
        """is_behavior_test returns False for regular functions."""
        def regular_func():
            return "output"
        
        assert is_behavior_test(regular_func) is False
    
    def test_decorator_captures_return_value(self):
        """CRITICAL: Decorator captures return value in function attribute."""
        @behavior("capture_test")
        def my_test():
            return "captured output"
        
        # Execute the function
        result = my_test()
        
        # Check that return value was captured
        assert hasattr(my_test, '_behaviorci_result')
        assert my_test._behaviorci_result == "captured output"
    
    def test_decorator_captures_inputs(self):
        """Decorator captures input arguments."""
        @behavior("input_test")
        def my_test(arg1, arg2, kwarg1=None):
            return f"{arg1} {arg2} {kwarg1}"
        
        my_test("hello", "world", kwarg1="test")
        
        assert hasattr(my_test, '_behaviorci_input')
        args, kwargs = my_test._behaviorci_input
        assert args == ("hello", "world")
        assert kwargs == {"kwarg1": "test"}
    
    def test_decorator_fails_on_none_return(self):
        """Decorator fails if function returns None."""
        @behavior("none_test")
        def my_test():
            return None
        
        with pytest.raises(ConfigurationError):
            my_test()
    
    def test_decorator_fails_on_non_string_return(self):
        """Decorator fails if function returns non-string."""
        @behavior("type_test")
        def my_test():
            return 123
        
        with pytest.raises(ConfigurationError):
            my_test()
    
    def test_decorator_fails_on_non_serializable_input(self):
        """Decorator fails fast on non-serializable inputs."""
        @behavior("serial_test")
        def my_test(dt):
            return "output"
        
        with pytest.raises(SerializationError):
            my_test(datetime.now())


# ============================================================================
# Storage Tests
# ============================================================================

class TestStorage:
    """Test SQLite storage layer."""
    
    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage for testing."""
        db_path = tmp_path / "test.db"
        return Storage(str(db_path))
    
    def test_compute_snapshot_id_deterministic(self):
        """Snapshot ID is deterministic."""
        id1 = compute_snapshot_id("behavior1", '{"args": [], "kwargs": {}}')
        id2 = compute_snapshot_id("behavior1", '{"args": [], "kwargs": {}}')
        assert id1 == id2
        assert len(id1) == 64  # SHA256 hex
    
    def test_compute_snapshot_id_different_inputs(self):
        """Different inputs produce different IDs."""
        id1 = compute_snapshot_id("behavior1", '{"args": [1]}')
        id2 = compute_snapshot_id("behavior1", '{"args": [2]}')
        assert id1 != id2
    
    def test_save_and_get_snapshot(self, temp_storage):
        """Save and retrieve snapshot."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        snapshot_id = temp_storage.save_snapshot(
            behavior_id="test_behavior",
            input_json='{"args": [], "kwargs": {}}',
            output_text="test output",
            embedding=embedding,
            model_name="test-model"
        )
        
        snapshot = temp_storage.get_snapshot(snapshot_id)
        assert snapshot.behavior_id == "test_behavior"
        assert snapshot.output_text == "test output"
        assert snapshot.model_name == "test-model"
    
    def test_snapshot_embedding_normalized(self, temp_storage):
        """Storage normalizes embeddings."""
        # Unnormalized embedding
        embedding = np.array([3.0, 4.0], dtype=np.float32)  # Norm = 5
        
        snapshot_id = temp_storage.save_snapshot(
            behavior_id="test",
            input_json='{}',
            output_text="test",
            embedding=embedding,
            model_name="test"
        )
        
        snapshot = temp_storage.get_snapshot(snapshot_id)
        stored = snapshot.get_embedding_array()
        
        # Should be normalized (L2 norm = 1)
        norm = np.linalg.norm(stored)
        assert abs(norm - 1.0) < 1e-5
    
    def test_overwrite_existing_snapshot(self, temp_storage):
        """Saving overwrites existing snapshot."""
        input_json = '{"args": [], "kwargs": {}}'
        
        id1 = temp_storage.save_snapshot(
            behavior_id="test",
            input_json=input_json,
            output_text="original",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="test"
        )
        
        id2 = temp_storage.save_snapshot(
            behavior_id="test",
            input_json=input_json,
            output_text="updated",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            model_name="test"
        )
        
        assert id1 == id2  # Same ID
        
        snapshot = temp_storage.get_snapshot(id1)
        assert snapshot.output_text == "updated"
    
    def test_similarity_history(self, temp_storage):
        """Record and retrieve similarity history."""
        # Create snapshot first
        snapshot_id = temp_storage.save_snapshot(
            behavior_id="test",
            input_json='{}',
            output_text="test",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="test"
        )
        
        # Record similarities
        temp_storage.record_similarity(snapshot_id, 0.95)
        temp_storage.record_similarity(snapshot_id, 0.92)
        temp_storage.record_similarity(snapshot_id, 0.94)
        
        history = temp_storage.get_similarity_history(snapshot_id)
        assert len(history) == 3
        assert 0.95 in history
        assert 0.92 in history
        assert 0.94 in history
    
    def test_per_snapshot_history_isolation(self, temp_storage):
        """History is isolated per snapshot (not per behavior)."""
        # Two different inputs under same behavior
        id1 = temp_storage.save_snapshot(
            behavior_id="same_behavior",
            input_json='{"args": [1]}',
            output_text="out1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="test"
        )
        
        id2 = temp_storage.save_snapshot(
            behavior_id="same_behavior",
            input_json='{"args": [2]}',
            output_text="out2",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            model_name="test"
        )
        
        # Record different similarities
        temp_storage.record_similarity(id1, 0.99)
        temp_storage.record_similarity(id2, 0.50)
        
        history1 = temp_storage.get_similarity_history(id1)
        history2 = temp_storage.get_similarity_history(id2)
        
        assert history1 == [0.99]
        assert history2 == [0.50]
    
    def test_stats(self, temp_storage):
        """Get database statistics."""
        # Empty initially
        stats = temp_storage.get_stats()
        assert stats["snapshots"] == 0
        assert stats["behaviors"] == 0
        
        # Add snapshots
        temp_storage.save_snapshot(
            behavior_id="behavior1",
            input_json='{"args": [1]}',
            output_text="out1",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
            model_name="test"
        )
        
        temp_storage.save_snapshot(
            behavior_id="behavior1",
            input_json='{"args": [2]}',
            output_text="out2",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
            model_name="test"
        )
        
        temp_storage.save_snapshot(
            behavior_id="behavior2",
            input_json='{}',
            output_text="out3",
            embedding=np.array([0.5, 0.5], dtype=np.float32),
            model_name="test"
        )
        
        stats = temp_storage.get_stats()
        assert stats["snapshots"] == 3
        assert stats["behaviors"] == 2


# ============================================================================
# Embedder Tests
# ============================================================================

class TestEmbedder:
    """Test embedding computation."""
    
    @pytest.fixture(autouse=True)
    def reset_global_embedder(self):
        """Reset global embedder before each test."""
        reset_embedder()
        yield
    
    def test_embed_single_text(self):
        """Embed single text returns vector."""
        embedder = Embedder()
        embedding = embedder.embed_single("hello world")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.dtype == np.float32
        assert embedding.shape == (384,)  # all-MiniLM-L6-v2 dimension
    
    def test_embedding_normalized(self):
        """Embeddings are L2 normalized."""
        embedder = Embedder()
        embedding = embedder.embed_single("test text")
        
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-5
    
    def test_embed_multiple_texts(self):
        """Embed multiple texts returns matrix."""
        embedder = Embedder()
        embeddings = embedder.embed(["text one", "text two", "text three"])
        
        assert embeddings.shape == (3, 384)
    
    def test_similarity_same_text(self):
        """Same text has similarity ~1.0."""
        embedder = Embedder()
        emb1 = embedder.embed_single("identical text")
        emb2 = embedder.embed_single("identical text")
        
        similarity = embedder.compute_similarity(emb1, emb2)
        assert similarity > 0.999
    
    def test_similarity_different_texts(self):
        """Different texts have lower similarity."""
        embedder = Embedder()
        emb1 = embedder.embed_single("machine learning")
        emb2 = embedder.embed_single("pizza recipe")
        
        similarity = embedder.compute_similarity(emb1, emb2)
        assert similarity < 0.8  # Should be quite different
    
    def test_similarity_similar_texts(self):
        """Similar texts have high similarity."""
        embedder = Embedder()
        emb1 = embedder.embed_single("I love machine learning")
        emb2 = embedder.embed_single("I enjoy machine learning")
        
        similarity = embedder.compute_similarity(emb1, emb2)
        assert similarity > 0.9  # Should be very similar


# ============================================================================
# Comparator Tests
# ============================================================================

class TestComparator:
    """Test comparison logic."""
    
    @pytest.fixture
    def temp_comparator(self, tmp_path):
        """Create comparator with temporary storage."""
        reset_embedder()
        db_path = tmp_path / "test.db"
        storage = Storage(str(db_path))
        embedder = get_embedder()
        return Comparator(storage, embedder)
    
    def test_lexical_must_contain_pass(self, temp_comparator):
        """Lexical check passes when must_contain present."""
        passed, missing, forbidden = temp_comparator.check_lexical(
            "This is a refund request",
            must_contain=["refund"],
            must_not_contain=[]
        )
        assert passed is True
        assert missing == []
        assert forbidden == []
    
    def test_lexical_must_contain_fail(self, temp_comparator):
        """Lexical check fails when must_contain missing."""
        passed, missing, forbidden = temp_comparator.check_lexical(
            "This is a request",
            must_contain=["refund"],
            must_not_contain=[]
        )
        assert passed is False
        assert missing == ["refund"]
    
    def test_lexical_must_not_contain_fail(self, temp_comparator):
        """Lexical check fails when must_not_contain present."""
        passed, missing, forbidden = temp_comparator.check_lexical(
            "This contains error message",
            must_contain=[],
            must_not_contain=["error"]
        )
        assert passed is False
        assert forbidden == ["error"]
    
    def test_lexical_case_insensitive(self, temp_comparator):
        """Lexical checks are case-insensitive."""
        passed, missing, forbidden = temp_comparator.check_lexical(
            "This is a REFUND request",
            must_contain=["refund"],
            must_not_contain=[]
        )
        assert passed is True
    
    def test_record_and_compare_same(self, temp_comparator):
        """Compare identical outputs passes."""
        # Record snapshot
        temp_comparator.record_snapshot(
            behavior_id="test",
            input_json='{"args": []}',
            output_text="test output"
        )
        
        # Compare same output
        result = temp_comparator.compare(
            behavior_id="test",
            input_json='{"args": []}',
            output_text="test output",
            base_threshold=0.85
        )
        
        assert result.passed is True
        assert result.similarity > 0.99
    
    def test_compare_different_output_fails(self, temp_comparator):
        """Compare different outputs fails."""
        # Record snapshot
        temp_comparator.record_snapshot(
            behavior_id="test",
            input_json='{"args": []}',
            output_text="original output"
        )
        
        # Compare different output
        result = temp_comparator.compare(
            behavior_id="test",
            input_json='{"args": []}',
            output_text="completely different output that means something else",
            base_threshold=0.85
        )
        
        assert result.passed is False
        assert result.similarity < 0.85
    
    def test_lexical_override_high_similarity(self, temp_comparator):
        """CRITICAL: High similarity but missing must_contain = FAIL."""
        # Record snapshot
        temp_comparator.record_snapshot(
            behavior_id="test",
            input_json='{"args": []}',
            output_text="The refund has been processed successfully"
        )
        
        # Compare with high similarity but missing must_contain
        result = temp_comparator.compare(
            behavior_id="test",
            input_json='{"args": []}',
            output_text="The refund has been processed successfully",  # Same text
            base_threshold=0.85,
            must_contain=["guarantee"]  # Not in output
        )
        
        assert result.passed is False
        assert "Missing required" in result.message or "guarantee" in str(result.missing_must_contain)
    
    def test_variance_aware_threshold(self, temp_comparator):
        """Variance-aware threshold adjusts based on history."""
        from behaviorci.storage import compute_snapshot_id
        
        input_json = '{"args": []}'
        behavior_id = "variance_test"
        snapshot_id = compute_snapshot_id(behavior_id, input_json)
        
        # Record snapshot
        temp_comparator.record_snapshot(
            behavior_id=behavior_id,
            input_json=input_json,
            output_text="test output"
        )
        
        # Record similarity history (simulating previous runs)
        for _ in range(5):
            temp_comparator.storage.record_similarity(snapshot_id, 0.92)
        
        # Compute effective threshold
        effective = temp_comparator.compute_effective_threshold(snapshot_id, 0.85)
        
        # With history of 0.92, effective threshold should be ~0.92
        assert effective > 0.85


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def setup(self, tmp_path):
        """Set up full test environment."""
        reset_embedder()
        db_path = tmp_path / "integration.db"
        storage = Storage(str(db_path))
        embedder = get_embedder()
        comparator = Comparator(storage, embedder)
        
        return {
            "storage": storage,
            "embedder": embedder,
            "comparator": comparator,
            "db_path": db_path
        }
    
    def test_full_workflow_record_then_check(self, setup):
        """Full workflow: record snapshot, then check passes."""
        comparator = setup["comparator"]
        
        # Step 1: Record
        comparator.record_snapshot(
            behavior_id="integration_test",
            input_json='{"args": ["hello"]}',
            output_text="Hello! How can I help you?"
        )
        
        # Step 2: Check (same output)
        result = comparator.compare(
            behavior_id="integration_test",
            input_json='{"args": ["hello"]}',
            output_text="Hello! How can I help you?",
            base_threshold=0.85
        )
        
        assert result.passed is True
    
    def test_full_workflow_detects_regression(self, setup):
        """Full workflow: detects behavioral regression."""
        comparator = setup["comparator"]
        
        # Step 1: Record baseline
        comparator.record_snapshot(
            behavior_id="regression_test",
            input_json='{"args": []}',
            output_text="I'm happy to help with your refund request."
        )
        
        # Step 2: Check with changed output (regression)
        result = comparator.compare(
            behavior_id="regression_test",
            input_json='{"args": []}',
            output_text="I cannot help with refunds. Contact support.",
            base_threshold=0.85
        )
        
        assert result.passed is False
        assert result.similarity < 0.85


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_behaviorci_error_has_message(self):
        """BehaviorCIError stores message."""
        error = BehaviorCIError("test message", {"key": "value"})
        assert error.message == "test message"
        assert error.details == {"key": "value"}
    
    def test_serialization_error_includes_type(self):
        """SerializationError includes object type."""
        error = SerializationError("datetime")
        assert "datetime" in error.message
        assert "JSON-serializable" in error.message