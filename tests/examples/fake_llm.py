"""Fake LLM for testing BehaviorCI without API calls.

This deterministic "LLM" returns predictable outputs based on input,
making it perfect for testing the BehaviorCI framework.
"""

from typing import Dict, List, Optional


class FakeLLM:
    """Deterministic fake LLM for testing."""

    def __init__(self, seed_responses: Optional[Dict[str, str]] = None):
        """Initialize fake LLM with optional seed responses.

        Args:
            seed_responses: Dictionary mapping input patterns to responses
        """
        self.responses = seed_responses or {}
        self.call_history: List[str] = []
        self._default_responses = {
            "refund": "I understand you want a refund. I can help you with that process.",
            "support": "I'm here to help. What seems to be the issue?",
            "billing": "I can assist with billing questions. What would you like to know?",
            "technical": "Let me help troubleshoot this technical issue.",
            "greeting": "Hello! How can I assist you today?",
        }

    def add_response(self, pattern: str, response: str) -> None:
        """Add a response pattern.

        Args:
            pattern: Input pattern to match
            response: Response to return
        """
        self.responses[pattern.lower()] = response

    def generate(self, prompt: str, variation: int = 0) -> str:
        """Generate a deterministic response.

        Args:
            prompt: Input prompt
            variation: Variation index (0 = default, 1+ = variations)

        Returns:
            Deterministic response string
        """
        self.call_history.append(prompt)
        prompt_lower = prompt.lower()

        # Check custom responses first
        for pattern, response in self.responses.items():
            if pattern in prompt_lower:
                return self._apply_variation(response, variation)

        # Check default responses
        for pattern, response in self._default_responses.items():
            if pattern in prompt_lower:
                return self._apply_variation(response, variation)

        # Default fallback
        return self._apply_variation("I'm not sure I understand. Could you rephrase?", variation)

    def _apply_variation(self, response: str, variation: int) -> str:
        """Apply variation to response.

        Args:
            response: Base response
            variation: Variation index

        Returns:
            Modified response
        """
        if variation == 0:
            return response

        # Add variation markers for testing
        variations = [
            response,
            response + " (variation 1)",
            response.replace(".", "!"),
            response + " Please let me know if you need more help.",
        ]

        return variations[variation % len(variations)]

    def classify(self, text: str, categories: Optional[List[str]] = None) -> str:
        """Classify text into a category.

        Args:
            text: Text to classify
            categories: Optional list of categories

        Returns:
            Category label
        """
        text_lower = text.lower()

        classification_rules = [
            ("refund", "REFUND_REQUEST"),
            ("return", "RETURN_REQUEST"),
            ("billing", "BILLING_QUESTION"),
            ("charge", "BILLING_QUESTION"),
            ("technical", "TECH_SUPPORT"),
            ("error", "TECH_SUPPORT"),
            ("support", "GENERAL_SUPPORT"),
            ("help", "GENERAL_SUPPORT"),
            ("hello", "GREETING"),
            ("hi", "GREETING"),
        ]

        for pattern, category in classification_rules:
            if pattern in text_lower:
                return category

        return "UNKNOWN"

    def reset_history(self) -> None:
        """Clear call history."""
        self.call_history = []


# Global instance for convenience
_default_llm = FakeLLM()


def get_llm() -> FakeLLM:
    """Get default fake LLM instance."""
    return _default_llm


def reset_llm() -> None:
    """Reset default LLM."""
    global _default_llm
    _default_llm = FakeLLM()
