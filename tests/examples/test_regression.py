"""Example showing how a changed output is captured as a distinct behavior."""

import pytest
from fake_llm import reset_llm

from behaviorci import behavior


@pytest.fixture(autouse=True)
def fresh_llm():
    reset_llm()
    yield


@behavior("refund_classifier_v2", threshold=0.85)
def test_refund_classification_v2():
    """A second classifier behavior, tracked independently from the first.

    Recording this and later changing the returned label is the quickest way to
    see BehaviorCI flag a regression in a demo.
    """
    return "BILLING_QUESTION"
