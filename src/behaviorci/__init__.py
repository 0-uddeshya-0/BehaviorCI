"""BehaviorCI - Pytest-native behavioral regression testing for LLM applications."""

from .api import behavior, get_behavior_config, is_behavior_test, get_captured_behavior
from .exceptions import (
    BehaviorCIError,
    SerializationError,
    SnapshotNotFoundError,
    StorageError,
    EmbeddingError,
    ComparisonError,
    ConfigurationError,
    ReplayError,
    ModelMismatchWarning,
)
from .models import Snapshot, BehaviorConfig, ComparisonResult, CapturedBehavior

__version__ = "0.1.0"
__all__ = [
    # API
    "behavior",
    "get_behavior_config",
    "is_behavior_test",
    "get_captured_behavior",
    # Exceptions
    "BehaviorCIError",
    "SerializationError",
    "SnapshotNotFoundError",
    "StorageError",
    "EmbeddingError",
    "ComparisonError",
    "ConfigurationError",
    "ReplayError",
    "ModelMismatchWarning",
    # Models
    "Snapshot",
    "BehaviorConfig",
    "ComparisonResult",
    "CapturedBehavior",
]