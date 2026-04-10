"""Pydantic models for BehaviorCI."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Snapshot(BaseModel):
    """A behavioral snapshot of LLM output for a specific input."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(..., description="SHA256 hash of behavior_id + input_json")
    behavior_id: str = Field(..., description="Logical behavior identifier")
    input_json: str = Field(..., description="JSON-serialized function arguments")
    output_text: str = Field(..., description="Captured LLM output")
    embedding: bytes = Field(..., description="Float32 numpy array as BLOB")
    model_name: str = Field(..., description="Embedding model used")
    created_at: int = Field(..., description="Unix timestamp")
    git_commit: Optional[str] = Field(None, description="Git commit hash if available")

    def get_embedding_array(self) -> np.ndarray:
        """Convert BLOB back to numpy array."""
        return np.frombuffer(self.embedding, dtype=np.float32)


class SimilarityRecord(BaseModel):
    """A single similarity comparison result."""

    snapshot_id: str = Field(..., description="Reference snapshot ID")
    similarity: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score")
    timestamp: int = Field(..., description="Unix timestamp")


class BehaviorConfig(BaseModel):
    """Configuration for a behavior-decorated test."""

    behavior_id: str = Field(..., description="Logical behavior identifier")
    threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )
    must_contain: Optional[List[str]] = Field(
        default=None, description="Required substrings in output"
    )
    must_not_contain: Optional[List[str]] = Field(
        default=None, description="Forbidden substrings in output"
    )
    func: Optional[Any] = Field(default=None, exclude=True, description="Original test function")
    samples: int = Field(default=1, ge=1, description="Number of samples for centroid baseline")


class ComparisonResult(BaseModel):
    """Result of comparing current output against a snapshot."""

    passed: bool = Field(..., description="Whether comparison passed all checks")
    snapshot_id: str = Field(..., description="Reference snapshot ID")
    behavior_id: str = Field(..., description="Logical behavior identifier")
    similarity: float = Field(..., description="Computed cosine similarity")
    effective_threshold: float = Field(..., description="Threshold used (may be variance-adjusted)")
    base_threshold: float = Field(..., description="Original threshold from decorator")
    lexical_passed: bool = Field(..., description="Whether lexical checks passed")
    missing_must_contain: List[str] = Field(
        default_factory=list, description="Missing required substrings"
    )
    found_must_not_contain: List[str] = Field(
        default_factory=list, description="Found forbidden substrings"
    )
    model_mismatch: bool = Field(default=False, description="Whether model names differed")
    stored_model: Optional[str] = Field(None, description="Model used for stored snapshot")
    current_model: Optional[str] = Field(None, description="Model used for current embedding")
    message: str = Field(..., description="Human-readable result message")


class CapturedBehavior(BaseModel):
    """Captured behavior from test execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output_text: Union[str, List[str]] = Field(
        ..., description="Captured LLM output or list of samples"
    )
    args: Tuple = Field(default_factory=tuple, description="Positional arguments")
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments")
    behavior_id: str = Field(..., description="Logical behavior identifier")
    threshold: float = Field(default=0.85, description="Similarity threshold")
    must_contain: Optional[List[str]] = Field(default=None, description="Required substrings")
    must_not_contain: Optional[List[str]] = Field(default=None, description="Forbidden substrings")
