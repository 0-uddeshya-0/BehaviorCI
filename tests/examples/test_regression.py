"""Demonstration of regression detection.

NOTE: This file must be run IN ISOLATION from test_app.py.
It intentionally simulates a behavioral regression by returning a different
output than what the baseline snapshot would contain.

Run as a standalone demo:
    pytest tests/examples/test_regression.py --behaviorci -v

Do NOT run alongside test_app.py in the same pytest session — the
behavior IDs here are distinct, but the intent is to show a regression
scenario, not to co-exist with a passing baseline in the same suite.
"""

import pytest
from behaviorci import behavior
from fake_llm import get_llm, reset_llm


@pytest.fixture(autouse=True)
def fresh_llm():
    """Reset LLM before each test."""
    reset_llm()
    yield


@behavior("refund_classifier_regression_demo", threshold=0.85)
def test_refund_classification_regression():
    """Simulates a behavioral regression for demonstration purposes.

    A baseline snapshot for "refund_classifier_regression_demo" would store
    "REFUND_REQUEST". This test returns "BILLING_QUESTION" to trigger a
    regression failure when run with --behaviorci.

    Workflow:
        # Record baseline (first time only):
        pytest tests/examples/test_regression.py --behaviorci-record

        # Simulate regression — this should FAIL:
        pytest tests/examples/test_regression.py --behaviorci
    """
    # Simulate regression: classifier returns wrong category
    result = "BILLING_QUESTION"  # Was originally REFUND_REQUEST
    return result
