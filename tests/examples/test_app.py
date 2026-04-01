"""Example BehaviorCI tests using fake LLM.

These tests demonstrate how to use BehaviorCI for behavioral regression testing.
"""

import pytest
from behaviorci import behavior
from fake_llm import FakeLLM, get_llm, reset_llm


# Create a fresh LLM instance for tests
@pytest.fixture(autouse=True)
def fresh_llm():
    """Reset LLM before each test."""
    reset_llm()
    yield


# ============================================================================
# Basic Behavior Tests
# ============================================================================

@behavior("refund_classifier", threshold=0.85)
def test_refund_classification():
    """Test that refund requests are classified correctly."""
    llm = get_llm()
    result = llm.classify("I want a refund for my order")
    return result  # RETURN VALUE CAPTURED


@behavior("billing_classifier", threshold=0.85)
def test_billing_classification():
    """Test that billing questions are classified correctly."""
    llm = get_llm()
    result = llm.classify("There's a charge on my bill I don't recognize")
    return result


@behavior("tech_support_classifier", threshold=0.85)
def test_technical_classification():
    """Test that technical issues are classified correctly."""
    llm = get_llm()
    result = llm.classify("I'm getting an error when I try to log in")
    return result


# ============================================================================
# Tests with Lexical Constraints
# ============================================================================

@behavior(
    "refund_response",
    threshold=0.85,
    must_contain=["refund", "help"]
)
def test_refund_response_content():
    """Test refund response contains required phrases."""
    llm = get_llm()
    result = llm.generate("I need a refund please")
    return result


@behavior(
    "support_response",
    threshold=0.85,
    must_contain=["help"],
    must_not_contain=["error", "fail"]
)
def test_support_response_safety():
    """Test support response is safe and helpful."""
    llm = get_llm()
    result = llm.generate("I need support with my account")
    return result


# ============================================================================
# Tests with Different Thresholds
# ============================================================================

@behavior("strict_greeting", threshold=0.95)
def test_strict_greeting():
    """Test greeting with very strict threshold."""
    llm = get_llm()
    result = llm.generate("Hello there", variation=0)
    return result


@behavior("lenient_response", threshold=0.70)
def test_lenient_response():
    """Test with more lenient threshold."""
    llm = get_llm()
    result = llm.generate("Can you help me?")
    return result


# ============================================================================
# Edge Case Tests
# ============================================================================

@behavior("unknown_input", threshold=0.85)
def test_unknown_classification():
    """Test classification of unknown input."""
    llm = get_llm()
    result = llm.classify("xyz123 nonsense input")
    return result


@behavior("greeting_detection", threshold=0.85)
def test_greeting_classification():
    """Test greeting detection."""
    llm = get_llm()
    result = llm.classify("Hello, how are you today?")
    return result


# ============================================================================
# Tests for Validation
# ============================================================================

# This test should fail in check mode if output changes
def test_without_behavior():
    """Regular pytest test (not a behavior test)."""
    llm = get_llm()
    result = llm.classify("refund")
    assert result == "REFUND_REQUEST"


# ============================================================================
# Demonstration Tests
# ============================================================================

@behavior("demo_refund", threshold=0.85, must_contain=["refund"])
def test_demo_refund_scenario():
    """Demo test showing full BehaviorCI workflow.
    
    1. First run: pytest --behaviorci-record (creates snapshot)
    2. Normal runs: pytest --behaviorci (checks against snapshot)
    3. If behavior changes: pytest --behaviorci-update (updates snapshot)
    """
    llm = get_llm()
    
    # Simulate a customer service interaction
    classification = llm.classify("I want my money back")
    response = llm.generate("refund request")
    
    # Return the combined output
    return f"[{classification}] {response}"