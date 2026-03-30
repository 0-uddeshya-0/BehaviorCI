"""Test for regression detection demonstration."""

import pytest
from behaviorci import behavior
from fake_llm import get_llm, reset_llm


@pytest.fixture(autouse=True)
def fresh_llm():
    """Reset LLM before each test."""
    reset_llm()
    yield


# This test uses the same behavior_id as test_refund_classification
# but returns different output to simulate a regression
@behavior("refund_classifier", threshold=0.85)
def test_refund_classification_regression():
    """This test simulates a behavioral regression.
    
    The original test returns 'REFUND_REQUEST' but this one returns
    'BILLING_QUESTION' to simulate a regression in the LLM behavior.
    """
    llm = get_llm()
    # Simulate regression: return different classification
    result = "BILLING_QUESTION"  # Changed from REFUND_REQUEST
    return result