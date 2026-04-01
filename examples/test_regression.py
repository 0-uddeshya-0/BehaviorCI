"""Test for regression detection demonstration."""

import pytest
from behaviorci import behavior
from fake_llm import get_llm, reset_llm


@pytest.fixture(autouse=True)
def fresh_llm():
    """Reset LLM before each test."""
    reset_llm()
    yield


# HIGH-003 FIX: Use unique behavior_id to avoid conflict with test_app.py
# The duplicate behavior_id would cause ConfigurationError with FIX-005
@behavior("refund_classifier_regression_demo", threshold=0.85)
def test_refund_classification_regression():
    """This test simulates a behavioral regression.
    
    The original test in test_app.py returns 'REFUND_REQUEST' but this 
    one returns 'BILLING_QUESTION' to simulate a regression in the LLM behavior.
    
    NOTE: Uses unique behavior_id to avoid conflict with FIX-005 validation.
    """
    llm = get_llm()
    # Simulate regression: return different classification
    result = "BILLING_QUESTION"  # Changed from REFUND_REQUEST
    return result