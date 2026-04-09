"""Comparison logic for behavioral regression testing."""

import numpy as np
import json
import warnings
from typing import List, Optional, Union, Tuple

from .models import ComparisonResult, Snapshot
from .storage import Storage
from .embedder import Embedder, get_embedder
from .exceptions import ComparisonError, ModelMismatchError, ModelMismatchWarning

class Comparator:
    def __init__(self, storage: Storage, embedder: Optional[Embedder] = None) -> None:
        self.storage = storage
        self.embedder = embedder or get_embedder()
    
    def _compute_embedding(self, text: Union[str, List[str]]) -> np.ndarray:
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
    ) -> Tuple[bool, List[str], List[str]]:
        output_lower = output_text.lower()
        missing_must_contain: List[str] = []
        found_must_not_contain: List[str] = []
        
        if must_contain:
            for substr in must_contain:
                if substr.lower() not in output_lower:
                    missing_must_contain.append(substr)
        
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
        history = self.storage.get_similarity_history(snapshot_id, limit=5)
        if len(history) < 3:
            return base_threshold
        
        mean = np.mean(history)
        std = np.std(history)
        variance_floor = float(mean - 2 * std)
        
        effective = float(max(0.5, min(base_threshold, variance_floor)))
        return effective
    
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
        from .storage import compute_snapshot_id
        snapshot_id = compute_snapshot_id(behavior_id, input_json)
        snapshot = self.storage.find_snapshot(behavior_id, input_json)
        
        if snapshot is None:
            if record_mode:
                return ComparisonResult(
                    passed=True,
                    snapshot_id=snapshot_id,
                    behavior_id=behavior_id,
                    similarity=1.0,
                    effective_threshold=base_threshold,
                    base_threshold=base_threshold,
                    lexical_passed=True,
                    message="Record mode: creating new snapshot",
                    stored_model=None,
                    current_model=None
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
                    message=f"No snapshot found. Run with --behaviorci-record to create.",
                    stored_model=None,
                    current_model=None
                )
        
        if snapshot.model_name != self.embedder.model_name:
            raise ModelMismatchError(
                stored_model=snapshot.model_name,
                current_model=self.embedder.model_name
            )
        
        text_for_lexical = output_text[0] if isinstance(output_text, list) else output_text
        lexical_passed, missing_must, found_must_not = self.check_lexical(
            text_for_lexical, must_contain, must_not_contain
        )
        
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
                message="; ".join(message_parts),
                stored_model=snapshot.model_name,
                current_model=self.embedder.model_name
            )
        
        current_embedding = self._compute_embedding(output_text)
        stored_embedding = snapshot.get_embedding_array()
        similarity = self.embedder.compute_similarity(current_embedding, stored_embedding)
        effective_threshold = self.compute_effective_threshold(snapshot_id, base_threshold)
        
        self.storage.record_similarity(snapshot_id, similarity)
        passed = similarity >= effective_threshold
        
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
        embedding = self._compute_embedding(output_text)
        text_to_store = json.dumps(output_text) if isinstance(output_text, list) else output_text
        
        snapshot_id = self.storage.save_snapshot(
            behavior_id=behavior_id,
            input_json=input_json,
            output_text=text_to_store,
            embedding=embedding,
            model_name=self.embedder.model_name,
            git_commit=git_commit
        )
        return snapshot_id
