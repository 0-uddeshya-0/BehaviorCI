"""Custom exceptions for BehaviorCI."""

from typing import Any, Dict, Optional


class BehaviorCIError(Exception):
    """Base exception for all BehaviorCI errors."""

    def __init__(self, message: str, details: Optional[Dict[Any, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class SerializationError(BehaviorCIError):
    """Raised when function inputs cannot be serialized to JSON."""

    def __init__(self, type_name: str, original_error: Optional[Exception] = None):
        msg = (
            f"Cannot serialize test inputs to JSON. Found non-serializable type: '{type_name}'.\n"
            f"BehaviorCI requires all test function arguments to be JSON-serializable "
            f"so they can be uniquely hashed for baseline tracking.\n"
            f"Hint: Use simpler types (str, int, dict) or mock external state objects."
        )
        super().__init__(msg, details={"type_name": type_name, "error": str(original_error)})


class ConfigurationError(BehaviorCIError):
    """Raised when BehaviorCI is configured incorrectly."""

    pass


class SnapshotNotFoundError(BehaviorCIError):
    """Raised when no baseline snapshot exists."""

    def __init__(self, behavior_id: str, snapshot_id: str):
        super().__init__(
            f"No snapshot found for behavior '{behavior_id}' (id: {snapshot_id[:8]}...).\n"
            f"Run with --behaviorci-record to create baseline."
        )


class ModelMismatchError(BehaviorCIError):
    """Raised when attempting to compare embeddings from different models."""

    def __init__(self, stored_model: str, current_model: str):
        self.stored_model = stored_model
        self.current_model = current_model
        msg = (
            f"Model mismatch detected.\n"
            f"The stored snapshot was created using '{stored_model}', "
            f"but the current test is running with '{current_model}'.\n"
            f"Embeddings from different models exist in different vector spaces "
            f"and cannot be accurately compared.\n"
            f"Resolution: Run pytest with --behaviorci-update to re-record the baseline "
            f"using the new model, or specify the original model using --behaviorci-model='{stored_model}'."
        )
        super().__init__(msg, details={"stored": stored_model, "current": current_model})


class EmbeddingError(BehaviorCIError):
    """Raised when embedding computation fails."""

    pass


class ComparisonError(BehaviorCIError):
    """Raised when comparison logic fails."""

    pass


class StorageError(BehaviorCIError):
    """Raised when database or storage operations fail."""

    pass


class ReplayError(BehaviorCIError):
    """Raised when replaying a behavior fails."""

    pass


class ModelMismatchWarning(Warning):
    """Warning issued when comparing snapshots from different embedding models."""

    def __init__(self, message: str, suggestion: Optional[str] = None):
        super().__init__(message)
        self.suggestion = suggestion
