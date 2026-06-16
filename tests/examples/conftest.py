"""Fixtures for the example behavior tests.

The examples double as dogfooding for BehaviorCI: they run real ``@behavior``
tests, but with a deterministic embedder injected so they need no model
download and behave identically across processes (record now, check later).
"""

import os
import sys

import pytest

# Make ``fake_llm`` importable the way a user's own test module would import a
# local helper.
sys.path.insert(0, os.path.dirname(__file__))

from behaviorci.embedder import reset_embedder, set_embedder  # noqa: E402
from tests.support import MockEmbedder  # noqa: E402


@pytest.fixture(autouse=True)
def use_mock_embedder():
    set_embedder(MockEmbedder())
    yield
    reset_embedder()
