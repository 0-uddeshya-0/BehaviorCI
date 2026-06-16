"""Tests for the comparison engine."""

import pytest

from behaviorci.comparator import Comparator
from behaviorci.exceptions import ModelMismatchError
from behaviorci.storage import Storage
from tests.support import MockEmbedder


@pytest.fixture
def comparator(tmp_path):
    storage = Storage(str(tmp_path / "c.db"))
    yield Comparator(storage, MockEmbedder())
    storage.close()


class TestLexical:
    def test_must_contain_pass(self, comparator):
        passed, missing, forbidden = comparator.check_lexical(
            "This is a refund request", must_contain=["refund"]
        )
        assert passed and not missing and not forbidden

    def test_must_contain_fail(self, comparator):
        passed, missing, _ = comparator.check_lexical("a request", must_contain=["refund"])
        assert not passed and missing == ["refund"]

    def test_must_not_contain_fail(self, comparator):
        passed, _, forbidden = comparator.check_lexical(
            "contains error", must_not_contain=["error"]
        )
        assert not passed and forbidden == ["error"]

    def test_case_insensitive(self, comparator):
        passed, _, _ = comparator.check_lexical("a REFUND", must_contain=["refund"])
        assert passed


class TestRecordCompare:
    def test_identical_output_passes(self, comparator):
        comparator.record_snapshot("b", '{"args": []}', "hello there")
        result = comparator.compare("b", '{"args": []}', "hello there", base_threshold=0.85)
        assert result.passed
        assert result.similarity > 0.99

    def test_drifted_output_fails(self, comparator):
        comparator.record_snapshot("b", '{"args": []}', "I will gladly help with your refund.")
        result = comparator.compare(
            "b", '{"args": []}', "Refunds are not possible. Goodbye.", base_threshold=0.85
        )
        assert not result.passed
        assert result.similarity < 0.85

    def test_missing_snapshot_fails_with_hint(self, comparator):
        result = comparator.compare("missing", "{}", "out", base_threshold=0.85)
        assert not result.passed
        assert "No snapshot found" in result.message

    def test_lexical_failure_overrides_high_similarity(self, comparator):
        comparator.record_snapshot("b", "{}", "The refund has been processed.")
        result = comparator.compare(
            "b",
            "{}",
            "The refund has been processed.",
            base_threshold=0.85,
            must_contain=["guarantee"],
        )
        assert not result.passed
        assert result.missing_must_contain == ["guarantee"]


class TestVarianceThreshold:
    def _seed(self, comparator, key, scores):
        for score in scores:
            comparator.storage.record_similarity(key, score)

    def test_high_variance_lowers_threshold(self, comparator):
        self._seed(comparator, "k", [0.70, 0.90, 0.75, 0.85, 0.80])
        effective = comparator.compute_effective_threshold("k", 0.85)
        assert effective < 0.85
        assert effective >= 0.5

    def test_low_variance_keeps_base(self, comparator):
        self._seed(comparator, "k", [0.88, 0.92, 0.89, 0.91, 0.90])
        assert comparator.compute_effective_threshold("k", 0.85) == 0.85

    def test_zero_variance_floor_equals_mean(self, comparator):
        self._seed(comparator, "k", [0.80, 0.80, 0.80])
        assert comparator.compute_effective_threshold("k", 0.85) == pytest.approx(0.80)

    def test_never_below_floor(self, comparator):
        self._seed(comparator, "k", [0.20, 0.40, 0.30])
        assert comparator.compute_effective_threshold("k", 0.85) == 0.5

    def test_insufficient_history_uses_base(self, comparator):
        self._seed(comparator, "k", [0.90, 0.80])
        assert comparator.compute_effective_threshold("k", 0.85) == pytest.approx(0.85)


class TestModelMismatch:
    def test_mismatch_raises(self, tmp_path):
        storage = Storage(str(tmp_path / "m.db"))
        Comparator(storage, MockEmbedder("model-a")).record_snapshot("b", "{}", "out")

        try:
            with pytest.raises(ModelMismatchError) as exc_info:
                Comparator(storage, MockEmbedder("model-b")).compare("b", "{}", "out", 0.85)

            error = exc_info.value
            assert error.stored_model == "model-a"
            assert error.current_model == "model-b"
            assert "different vector spaces" in error.message.lower()
            assert "--behaviorci-update" in error.message
        finally:
            storage.close()

    def test_same_model_compares_normally(self, tmp_path):
        storage = Storage(str(tmp_path / "m.db"))
        comparator = Comparator(storage, MockEmbedder("model-a"))
        comparator.record_snapshot("b", "{}", "out")
        assert comparator.compare("b", "{}", "out", 0.85).passed
        storage.close()


class TestCentroidBaseline:
    def test_centroid_record_and_compare(self, comparator):
        samples = [
            "A sunny day at the beach.",
            "A bright morning by the sea.",
            "Sunshine on the coast.",
        ]
        comparator.record_snapshot("story", "{}", samples)
        # Comparing a fresh batch of related samples stays above a relaxed threshold.
        result = comparator.compare("story", "{}", samples, base_threshold=0.75)
        assert result.passed
