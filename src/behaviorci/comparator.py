"""Comparison logic for behavioral regression testing.

Layer 0: Lexical checks (must_contain, must_not_contain)
Layer 1: Semantic similarity using embeddings
"""

import numpy as np
import warnings
from typing import List, Optional

from .models import ComparisonResult, Snapshot
from .storage import Storage
from .embedder import Embedder, get_embedder
from .exceptions import ComparisonError, ModelMismatchWarning


class Comparator:
    """Compares current LLM output against stored snapshots."""
    
    def __init__(self, storage: Storage, embedder: Embedder = None):
        """Initialize comparator.
        
        Args:
            storage: Storage instance for snapshots
            embedder: Embedder instance (uses global if None)
        """
        self.storage = storage
        self.embedder = embedder or get_embedder()
    
    def check_lexical(
        self,
        output_text: str,
        must_contain: Optional[List[str]] = None,
        must_not_contain: Optional[List[str]] = None
    ) -> tuple:
        """Layer 0: Fast lexical checks.
        
        Args:
            output_text: Current LLM output
            must_contain: Required substrings (case-insensitive)
            must_not_contain: Forbidden substrings (case-insensitive)
            
        Returns:
            Tuple of (passed, missing_must_contain, found_must_not_contain)
        """
        output_lower = output_text.lower()
        
        missing_must_contain = []
        found_must_not_contain = []
        
        # Check must_contain
        if must_contain:
            for substr in must_contain:
                if substr.lower() not in output_lower:
                    missing_must_contain.append(substr)
        
        # Check must_not_contain
        if must_not_contain:
            for substr in must_not_contain:
                if substr.lower() in output_lower:
                    found_must_not_contain.append(substr)
        
        passed = len(missing_must_contain) == 0 and len(found_must_not_contain) == 0
        return passed, missing_must_contain, found_must_not_contain
    
    def compute_effective_threshold(
        self,
        snapshot_id: str,
        base_threshold: float
    ) -> float:
        """Compute variance-aware effective threshold.
        
        Uses per-snapshot similarity history to adjust threshold.
        If we have enough history (>=3 points), use mean - 2*std.
        Otherwise, use base threshold.
        
        Args:
            snapshot_id: The snapshot ID
            base_threshold: User-specified threshold
            
        Returns:
            Effective threshold to use
        """
        history = self.storage.get_similarity_history(snapshot_id, limit=5)
        
        if len(history) >= 3:
            mean = np.mean(history)
            std = np.std(history)
            # Use mean - 2*std, but never go below base_threshold
            effective = max(base_threshold, mean - 2 * std)
            return effective
        
        return base_threshold
    
    def compare(
        self,
        behavior_id: str,
        input_json: str,
        output_text: str,
        base_threshold: float = 0.85,
        must_contain: Optional[List[str]] = None,
        must_not_contain: Optional[List[str]] = None,
        record_mode: bool = False
    ) -> ComparisonResult:
        """Compare current output against stored snapshot.
        
        Args:
            behavior_id: Logical behavior identifier
            input_json: JSON-serialized input arguments
            output_text: Current LLM output
            base_threshold: Minimum similarity threshold
            must_contain: Required substrings
            must_not_contain: Forbidden substrings
            record_mode: If True, don't fail on missing snapshot
            
        Returns:
            ComparisonResult with all details
        """
        from .storage import compute_snapshot_id
        
        snapshot_id = compute_snapshot_id(behavior_id, input_json)
        
        # Load existing snapshot
        snapshot = self.storage.find_snapshot(behavior_id, input_json)
        
        if snapshot is None:
            if record_mode:
                # In record mode, create new snapshot
                return ComparisonResult(
                    passed=True,
                    snapshot_id=snapshot_id,
                    behavior_id=behavior_id,
                    similarity=1.0,
                    effective_threshold=base_threshold,
                    base_threshold=base_threshold,
                    lexical_passed=True,
                    message="Record mode: creating new snapshot"
                )
            else:
                return ComparisonResult(
                    passed=False,
                    snapshot_id=snapshot_id,
                    behavior_id=behavior_id,
                    similarity=0.0,
                    effective_threshold=base_threshold,
                    base_threshold=base_threshold,
                    lexical_passed=False,
                    message=f"No snapshot found. Run with --behaviorci-record to create."
                )
        
        # Layer 0: Lexical checks
        lexical_passed, missing_must, found_must_not = self.check_lexical(
            output_text, must_contain, must_not_contain
        )
        
        # If lexical checks fail, fail immediately (skip embedding)
        if not lexical_passed:
            message_parts = []
            if missing_must:
                message_parts.append(f"Missing required: {missing_must}")
            if found_must_not:
                message_parts.append(f"Found forbidden: {found_must_not}")
            
            return ComparisonResult(
                passed=False,
                snapshot_id=snapshot_id,
                behavior_id=behavior_id,
                similarity=0.0,
                effective_threshold=base_threshold,
                base_threshold=base_threshold,
                lexical_passed=False,
                missing_must_contain=missing_must,
                found_must_not_contain=found_must_not,
                message="; ".join(message_parts)
            )
        
        # Layer 1: Semantic comparison
        # Compute embedding for current output
        current_embedding = self.embedder.embed_single(output_text)
        stored_embedding = snapshot.get_embedding_array()
        
        # Check model mismatch
        model_mismatch = self.embedder.model_name != snapshot.model_name
        if model_mismatch:
            warnings.warn(ModelMismatchWarning(snapshot.model_name, self.embedder.model_name))
        
        # Compute similarity
        similarity = self.embedder.compute_similarity(current_embedding, stored_embedding)
        
        # Compute effective threshold (variance-aware)
        effective_threshold = self.compute_effective_threshold(snapshot_id, base_threshold)
        
        # Record this similarity for future variance tracking
        self.storage.record_similarity(snapshot_id, similarity)
        
        # Determine pass/fail
        passed = similarity >= effective_threshold
        
        # Build message
        if passed:
            message = f"Similarity {similarity:.4f} >= threshold {effective_threshold:.4f}"
            if effective_threshold != base_threshold:
                message += f" (variance-adjusted from {base_threshold:.4f})"
        else:
            message = f"Similarity {similarity:.4f} < threshold {effective_threshold:.4f}"
            if effective_threshold != base_threshold:
                message += f" (variance-adjusted from {base_threshold:.4f})"
        
        return ComparisonResult(
            passed=passed,
            snapshot_id=snapshot_id,
            behavior_id=behavior_id,
            similarity=similarity,
            effective_threshold=effective_threshold,
            base_threshold=base_threshold,
            lexical_passed=True,
            model_mismatch=model_mismatch,
            stored_model=snapshot.model_name,
            current_model=self.embedder.model_name,
            message=message
        )
    
    def record_snapshot(
        self,
        behavior_id: str,
        input_json: str,
        output_text: str,
        git_commit: Optional[str] = None
    ) -> str:
        """Record a new snapshot.
        
        Args:
            behavior_id: Logical behavior identifier
            input_json: JSON-serialized input arguments
            output_text: LLM output to store
            git_commit: Optional git commit hash
            
        Returns:
            snapshot_id: The computed snapshot ID
        """
        # Compute embedding
        embedding = self.embedder.embed_single(output_text)
        
        # Save to storage
        snapshot_id = self.storage.save_snapshot(
            behavior_id=behavior_id,
            input_json=input_json,
            output_text=output_text,
            embedding=embedding,
            model_name=self.embedder.model_name,
            git_commit=git_commit
        )
        
        return snapshot_id