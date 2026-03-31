"""Exception classes for BehaviorCI."""


class BehaviorCIError(Exception):
    """Base exception for all BehaviorCI errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class SerializationError(BehaviorCIError):
    """Raised when inputs cannot be JSON-serialized."""
    
    def __init__(self, obj_type: str, original_error: Exception = None):
        message = (
            f"Inputs must be JSON-serializable. "
            f"Non-serializable type: {obj_type}. "
            f"Use str/int/dict/list only."
        )
        super().__init__(message, {"obj_type": obj_type, "original_error": str(original_error)})


class SnapshotNotFoundError(BehaviorCIError):
    """Raised when no snapshot exists for a given input."""
    
    def __init__(self, snapshot_id: str, behavior_id: str):
        message = (
            f"No snapshot found for behavior '{behavior_id}' (id: {snapshot_id[:16]}...). "
            f"Run with --behaviorci-record to create initial snapshot."
        )
        super().__init__(message, {"snapshot_id": snapshot_id, "behavior_id": behavior_id})


class StorageError(BehaviorCIError):
    """Raised when database operations fail."""
    pass


class EmbeddingError(BehaviorCIError):
    """Raised when embedding computation fails."""
    pass


class ComparisonError(BehaviorCIError):
    """Raised when comparison logic encounters an error."""
    pass


class ConfigurationError(BehaviorCIError):
    """Raised when behavior configuration is invalid."""
    pass


class ReplayError(BehaviorCIError):
    """Raised when input replay fails."""
    
    def __init__(self, reason: str, suggestion: str = None):
        message = f"Replay failed: {reason}"
        if suggestion:
            message += f". {suggestion}"
        super().__init__(message, {"reason": reason})


class ModelMismatchWarning(UserWarning):
    """Warning when comparing embeddings from different models."""
    
    def __init__(self, stored_model: str, current_model: str):
        self.stored_model = stored_model
        self.current_model = current_model
        self.message = (
            f"Model mismatch: stored with '{stored_model}', "
            f"comparing with '{current_model}'. Similarity may be unreliable."
        )
        super().__init__(self.message)
