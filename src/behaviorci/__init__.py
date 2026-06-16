"""BehaviorCI - Pytest-native behavioral regression testing for LLM applications."""

from .api import behavior, get_behavior_config, get_captured_behavior, is_behavior_test
from .exceptions import (
    BehaviorCIError,
    ComparisonError,
    ConfigurationError,
    EmbeddingError,
    ModelMismatchError,
    ModelMismatchWarning,
    ReplayError,
    SerializationError,
    SnapshotNotFoundError,
    StorageError,
)
from .models import BehaviorConfig, CapturedBehavior, ComparisonResult, Snapshot
from .storage import get_storage, reset_all_storage, reset_storage

__version__ = "0.2.0"
__all__ = [
    # Version
    "__version__",
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
    "ModelMismatchError",
    "ModelMismatchWarning",
    # Models
    "Snapshot",
    "BehaviorConfig",
    "ComparisonResult",
    "CapturedBehavior",
    # Storage utilities
    "get_storage",
    "reset_storage",
    "reset_all_storage",
]
