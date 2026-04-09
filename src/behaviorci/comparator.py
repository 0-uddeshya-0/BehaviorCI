"""Comparison logic for behavioral regression testing.

Layer 0: Lexical checks (must_contain, must_not_contain)
Layer 1: Semantic similarity using embeddings
"""

import numpy as np
import json
import warnings
from typing import List, Optional, Union

from .models import ComparisonResult, Snapshot
from .storage import Storage
from .embedder import Embedder, get_embedder
from .exceptions import ComparisonError, ModelMismatchError, ModelMismatchWarning


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
    
    def _compute_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
        """Compute standard or centroid embedding.
        
        If a list of strings is provided, embeds all, averages them, 
        and re-normalizes the resulting centroid vector.
        """
        if isinstance(text, list):
            embeddings = [self.embedder.embed_single(t) for t in text]
            centroid = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            return centroid.astype(np.float32)
        return self.embedder.embed_single(text)
    
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
        
        CRITICAL-002 FIX: High variance outputs → lower threshold (more tolerant)
        Low variance outputs → base threshold (strict)
        
        WHY: Previous implementation used max() which could only raise threshold.
        This is backwards: high-variance outputs (creative writing) need LOWER
        thresholds to avoid false positives. Low-variance outputs (structured data)
        should stay strict.
        
        CORRECT LOGIC:
        - High variance (std=0.08, mean=0.85): floor=0.69, min(0.85, 0.69)=0.69 (tolerant)
        - Low variance (std=0.01, mean=0.92): floor=0.90, min(0.85, 0.90)=0.85 (strict)
        
        Args:
            snapshot_id: The snapshot ID
            base_threshold: User-specified threshold
            
        Returns:
            Effective threshold to use (never below 0.5)
            
        VERIFIED BY: tests/test_critical_002_variance_logic.py
        """
        history = self.storage.get_similarity_history(snapshot_id, limit=5)
        
        if len(history) < 3:
            return base_threshold
        
        mean = np.mean(history)
        std = np.std(history)
        
        # Lower bound based on variance (2 standard deviations below mean)
        variance_floor = mean - 2 * std
        
        # CRITICAL-002: Use minimum of base and floor, but never below 0.5
        # High variance → lower threshold (more tolerant)
        # Low variance → base threshold (strict)
        effective = max(0.5, min(base_threshold, variance_floor))
        
        return float(effective)
    
    def compare(
        self,
        behavior_id: str,
        input_json: str,
        output_text: Union[str, List[str]],
        base_threshold: float = 0.85,
        must_contain: Optional[List[str]] = None,
        must_not_contain: Optional[List[str]] = None,
        record_mode: bool = False
    ) -> ComparisonResult:
        """Compare current output against stored snapshot.
        
        TASK 1 (v0.2): Model mismatch is now a hard error. Comparing embeddings
        from different models is mathematically invalid (different vector spaces).
        
        Args:
            behavior_id: Logical behavior identifier
            input_json: JSON-serialized input arguments
            output_text: Current LLM output (or list of outputs for centroid)
            base_threshold: Minimum similarity threshold
            must_contain: Required substrings
            must_not_contain: Forbidden substrings
            record_mode: If True, don't fail on missing snapshot
            
        Returns:
            ComparisonResult with all details
            
        Raises:
            ModelMismatchError: If snapshot was recorded with a different model
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
        
        # TASK 1 (v0.2): Model mismatch is now a hard error.
        # Comparing embeddings from different models is mathematically invalid
        # because they exist in different vector spaces.
        if snapshot.model_name != self.embedder.model_name:
            raise ModelMismatchError(
                stored_model=snapshot.model_name,
                current_model=self.embedder.model_name
            )
        
        # Layer 0: Lexical checks (Checked against the primary/first generation)
        text_for_lexical = output_text[0] if isinstance(output_text, list) else output_text
        lexical_passed, missing_must, found_must_not = self.check_lexical(
            text_for_lexical, must_contain, must_not_contain
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
        # Compute embedding for current output (handles Centroid if list)
        current_embedding = self._compute_embedding(output_text)
        stored_embedding = snapshot.get_embedding_array()
        
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
            model_mismatch=False,
            stored_model=snapshot.model_name,
            current_model=self.embedder.model_name,
            message=message
        )
    
    def record_snapshot(
        self,
        behavior_id: str,
        input_json: str,
        output_text: Union[str, List[str]],
        git_commit: Optional[str] = None
    ) -> str:
        """Record a new snapshot.
        
        Args:
            behavior_id: Logical behavior identifier
            input_json: JSON-serialized input arguments
            output_text: LLM output to store (or list of outputs for centroid)
            git_commit: Optional git commit hash
            
        Returns:
            snapshot_id: The computed snapshot ID
        """
        # Compute embedding (handles centroid math if list)
        embedding = self._compute_embedding(output_text)
        
        # Serialize list if necessary for SQLite
        text_to_store = json.dumps(output_text) if isinstance(output_text, list) else output_text
        
        # Save to storage
        snapshot_id = self.storage.save_snapshot(
            behavior_id=behavior_id,
            input_json=input_json,
            output_text=text_to_store,
            embedding=embedding,
            model_name=self.embedder.model_name,
            git_commit=git_commit
        )
        
        return snapshot_id
