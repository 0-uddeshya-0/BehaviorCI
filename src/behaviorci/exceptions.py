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


class ModelMismatchError(BehaviorCIError):
    """Raised when comparing embeddings from different models.
    
    TASK 1 (v0.2): Model mismatch is now a hard error instead of a warning.
    Comparing embeddings from different models is mathematically invalid
    because they exist in different vector spaces.
    """
    
    def __init__(self, stored_model: str, current_model: str):
        self.stored_model = stored_model
        self.current_model = current_model
        message = (
            f"Model mismatch: snapshot recorded with '{stored_model}', "
            f"but current model is '{current_model}'. "
            f"Embeddings are incomparable (different vector spaces). "
            f"Use --behaviorci-update to re-record snapshots with the current model, "
            f"or specify the original model with --behaviorci-model."
        )
        super().__init__(message, {
            "stored_model": stored_model,
            "current_model": current_model
        })


class ModelMismatchWarning(UserWarning):
    """Warning when comparing embeddings from different models.
    
    DEPRECATED (v0.2): This is now a hard error (ModelMismatchError).
    Kept for backward compatibility with code that catches this warning.
    """
    
    def __init__(self, stored_model: str, current_model: str):
        self.stored_model = stored_model
        self.current_model = current_model
        self.message = (
            f"Model mismatch: stored with '{stored_model}', "
            f"comparing with '{current_model}'. Similarity may be unreliable."
        )
        super().__init__(self.message)
