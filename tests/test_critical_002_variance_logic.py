"""Test for CRITICAL-002: Correct variance threshold logic.

CRITICAL-002: Previous implementation used max(base_threshold, mean - 2*std)
which could only RAISE threshold. This is backwards:
- High variance (creative writing) needs LOWER thresholds (more tolerant)
- Low variance (structured data) should stay strict

FIX: Changed to max(0.5, min(base_threshold, variance_floor))

VERIFICATION: These tests prove high variance lowers threshold, low variance keeps base.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "examples"))

import numpy as np
import tempfile

from behaviorci.storage import get_storage, reset_all_storage
from behaviorci.comparator import Comparator
from mock_embedder import MockEmbedder


def test_high_variance_lowers_threshold():
    """High variance outputs should get lower threshold (more tolerant).

    WHY: CRITICAL-002 fix - High variance (std=0.08, mean=0.85) should result
    in threshold < 0.85, not >= 0.85.

    VERIFIED: compute_effective_threshold uses min(base, floor) not max().
    """
    reset_all_storage()
    storage = get_storage(":memory:")
    embedder = MockEmbedder()
    comparator = Comparator(storage, embedder)

    # Record 5 snapshots with HIGH variance (0.70 to 0.90, std ~0.08)
    # mean = 0.82, std ~0.077
    high_variance_sims = [0.70, 0.90, 0.75, 0.85, 0.80]
    for sim in high_variance_sims:
        comparator.storage.record_similarity("test_high", sim)

    base_threshold = 0.85
    threshold = comparator.compute_effective_threshold("test_high", base_threshold)

    # High variance should LOWER threshold below base
    assert (
        threshold < base_threshold
    ), f"High variance should lower threshold below {base_threshold}, got {threshold}"

    # But never below 0.5
    assert threshold >= 0.5, f"Threshold should not go below 0.5, got {threshold}"


def test_low_variance_keeps_base():
    """Low variance outputs should keep base threshold (strict).

    WHY: CRITICAL-002 fix - Low variance (std=0.01, mean=0.92) should keep
    base threshold at 0.85 because variance_floor (0.90) > base (0.85).

    VERIFIED: min(0.85, 0.90) = 0.85, so threshold stays at base.
    """
    reset_all_storage()
    storage = get_storage(":memory:")
    embedder = MockEmbedder()
    comparator = Comparator(storage, embedder)

    # Record 5 snapshots with LOW variance (0.88 to 0.92, std ~0.016)
    # mean = 0.90, std ~0.016
    low_variance_sims = [0.88, 0.92, 0.89, 0.91, 0.90]
    for sim in low_variance_sims:
        comparator.storage.record_similarity("test_low", sim)

    base_threshold = 0.85
    threshold = comparator.compute_effective_threshold("test_low", base_threshold)

    # Low variance should keep base threshold
    assert (
        threshold == base_threshold
    ), f"Low variance should keep base threshold {base_threshold}, got {threshold}"


def test_variance_floor_calculation():
    """Verify variance floor is calculated correctly (mean - 2*std).

    WHY: The variance floor is the key calculation for adaptive thresholds.
    """
    reset_all_storage()
    storage = get_storage(":memory:")
    embedder = MockEmbedder()
    comparator = Comparator(storage, embedder)

    # Use exact values for predictable calculation
    sims = [0.80, 0.80, 0.80]  # mean=0.80, std=0.0
    for sim in sims:
        comparator.storage.record_similarity("test_exact", sim)

    base_threshold = 0.85
    threshold = comparator.compute_effective_threshold("test_exact", base_threshold)

    # With std=0, variance_floor = 0.80 - 0 = 0.80
    # min(0.85, 0.80) = 0.80
    # max(0.5, 0.80) = 0.80
    assert (
        abs(threshold - 0.80) < 0.001
    ), f"Expected threshold ~0.80 for mean=0.80, std=0, got {threshold}"


def test_minimum_threshold_floor():
    """Threshold should never go below 0.5.

    WHY: Even with extreme variance, we need a minimum threshold to prevent
    accepting completely different outputs.
    """
    reset_all_storage()
    storage = get_storage(":memory:")
    embedder = MockEmbedder()
    comparator = Comparator(storage, embedder)

    # Record very low similarities (would produce negative floor)
    # mean = 0.30, std ~0.14, floor = 0.30 - 0.28 = 0.02
    very_low_sims = [0.20, 0.40, 0.30]
    for sim in very_low_sims:
        comparator.storage.record_similarity("test_very_low", sim)

    base_threshold = 0.85
    threshold = comparator.compute_effective_threshold("test_very_low", base_threshold)

    # Should be clamped to 0.5 minimum
    assert threshold == 0.5, f"Threshold should be clamped to 0.5 minimum, got {threshold}"


def test_insufficient_history_uses_base():
    """With fewer than 3 history points, use base threshold.

    WHY: Need enough data to calculate meaningful variance.
    """
    reset_all_storage()
    storage = get_storage(":memory:")
    embedder = MockEmbedder()
    comparator = Comparator(storage, embedder)

    # Use a unique snapshot_id to ensure no history from other tests
    unique_id = "test_few_isolated_" + str(id(test_insufficient_history_uses_base))

    # Record only 2 snapshots (not enough for variance calculation)
    comparator.storage.record_similarity(unique_id, 0.90)
    comparator.storage.record_similarity(unique_id, 0.80)

    base_threshold = 0.85
    threshold = comparator.compute_effective_threshold(unique_id, base_threshold)

    # Should use base threshold (not enough history)
    assert (
        abs(threshold - base_threshold) < 0.001
    ), f"With <3 history points, should use base {base_threshold}, got {threshold}"


def test_variance_vs_max_bug():
    """Prove the old max() bug is fixed.

    WHY: Old code used max(base, floor) which could only RAISE threshold.
    This test proves the new min() logic works correctly.
    """
    reset_all_storage()
    storage = get_storage(":memory:")
    embedder = MockEmbedder()
    comparator = Comparator(storage, embedder)

    # High variance scenario: mean=0.82, std~0.077, floor=0.82-0.154=0.666
    # Old (buggy): max(0.85, 0.666) = 0.85 (no change - WRONG)
    # New (fixed): max(0.5, min(0.85, 0.666)) = 0.666 (lowered - CORRECT)

    high_variance_sims = [0.70, 0.90, 0.75, 0.85, 0.80]
    for sim in high_variance_sims:
        comparator.storage.record_similarity("test_bug_proof", sim)

    base_threshold = 0.85
    threshold = comparator.compute_effective_threshold("test_bug_proof", base_threshold)

    # Prove it's NOT using max() logic (which would give 0.85)
    assert threshold < 0.80, f"Threshold {threshold} should be < 0.80 (proves max() bug is fixed)"
