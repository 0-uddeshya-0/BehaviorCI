"""Public API for BehaviorCI - the @behavior decorator.

CRITICAL: Pytest ignores return values by default. We use function attributes
+ stash pattern to capture return values from test functions.
"""

import functools
import json
from typing import Callable, Optional, List, Any, Tuple

from .models import BehaviorConfig, CapturedBehavior
from .exceptions import SerializationError, ConfigurationError


class _StrictEncoder(json.JSONEncoder):
    """JSON encoder that raises SerializationError immediately on non-serializable types.

    WHY: The default json.dumps TypeError message is fragile to parse and varies
         across Python versions. A custom encoder raises a clean, typed error
         the moment it encounters a non-serializable object.
    """
    def default(self, obj: Any) -> Any:
        raise SerializationError(type(obj).__name__, TypeError(repr(obj)))


def serialize_inputs(args: Tuple, kwargs: dict) -> str:
    """Serialize function inputs to canonical JSON.

    CRITICAL: Uses STRICT serialization - NO default=str.
    Fails fast on non-serializable inputs.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Canonical JSON string

    Raises:
        SerializationError: If inputs are not JSON-serializable
    """
    data = {'args': args, 'kwargs': kwargs}
    try:
        return json.dumps(data, sort_keys=True, cls=_StrictEncoder)
    except SerializationError:
        raise
    except TypeError as e:
        raise SerializationError("unknown", e) from e


def behavior(
    behavior_id: str,
    threshold: float = 0.85,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None
) -> Callable:
    """Decorator to mark a test function for behavioral regression testing.

    The decorated function's RETURN VALUE is captured as the LLM output.
    The function MUST return a string (the LLM output to compare).

    Args:
        behavior_id: Logical behavior identifier (e.g., "refund_classifier")
        threshold: Minimum cosine similarity threshold (0-1)
        must_contain: List of substrings that MUST be in output (case-insensitive)
        must_not_contain: List of substrings that MUST NOT be in output

    Returns:
        Decorated function with behavior metadata

    Example:
        @behavior("classifier", threshold=0.85, must_contain=["refund"])
        def test_refund():
            return classify("I want a refund")  # RETURN VALUE CAPTURED
    """
    if not behavior_id or not isinstance(behavior_id, str):
        raise ConfigurationError("behavior_id must be a non-empty string")

    if not (0.0 <= threshold <= 1.0):
        raise ConfigurationError("threshold must be between 0.0 and 1.0")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)

            if result is None:
                raise ConfigurationError(
                    f"Test function '{func.__name__}' returned None. "
                    f"BehaviorCI tests must return the LLM output string."
                )

            if not isinstance(result, str):
                raise ConfigurationError(
                    f"Test function '{func.__name__}' returned {type(result).__name__}, "
                    f"expected str. BehaviorCI tests must return the LLM output string."
                )

            input_json = serialize_inputs(args, kwargs)

            wrapper._behaviorci_result = result
            wrapper._behaviorci_input = (args, kwargs)
            wrapper._behaviorci_input_json = input_json

            return result

        wrapper._behavior_config = BehaviorConfig(
            behavior_id=behavior_id,
            threshold=threshold,
            must_contain=must_contain,
            must_not_contain=must_not_contain,
            func=func
        )
        wrapper._is_behavior_test = True

        return wrapper

    return decorator


def get_behavior_config(func: Callable) -> Optional[BehaviorConfig]:
    """Get behavior configuration from a decorated function."""
    return getattr(func, '_behavior_config', None)


def is_behavior_test(func: Callable) -> bool:
    """Check if a function is a behavior test."""
    return getattr(func, '_is_behavior_test', False)


def get_captured_behavior(func: Callable) -> Optional[CapturedBehavior]:
    """Get captured behavior from a recently executed test function."""
    config = get_behavior_config(func)
    if config is None:
        return None

    result = getattr(func, '_behaviorci_result', None)
    if result is None:
        return None

    args, kwargs = getattr(func, '_behaviorci_input', ((), {}))

    return CapturedBehavior(
        output_text=result,
        args=args,
        kwargs=kwargs,
        behavior_id=config.behavior_id,
        threshold=config.threshold,
        must_contain=config.must_contain,
        must_not_contain=config.must_not_contain
    )


__all__ = [
    'behavior',
    'serialize_inputs',
    'get_behavior_config',
    'is_behavior_test',
    'get_captured_behavior',
]
