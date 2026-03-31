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
        raise  # Already a clean typed error from _StrictEncoder
    except TypeError as e:
        # Fallback: shouldn't normally be reached, but handle defensively
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
            # Execute the test function
            result = func(*args, **kwargs)

            # CRITICAL: Validate return value
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

            # Serialize inputs (FAIL FAST if not JSON-serializable)
            input_json = serialize_inputs(args, kwargs)

            # Store result for plugin to retrieve after normal pytest execution
            wrapper._behaviorci_result = result
            wrapper._behaviorci_input = (args, kwargs)
            wrapper._behaviorci_input_json = input_json

            return result

        # Attach configuration for pytest plugin discovery
        wrapper._behavior_config = BehaviorConfig(
            behavior_id=behavior_id,
            threshold=threshold,
            must_contain=must_contain,
            must_not_contain=must_not_contain,
            func=func
        )

        # Mark as behavior test for easy detection
        wrapper._is_behavior_test = True

        return wrapper

    return decorator


def get_behavior_config(func: Callable) -> Optional[BehaviorConfig]:
    """Get behavior configuration from a decorated function.

    Args:
        func: Potentially decorated function

    Returns:
        BehaviorConfig if decorated, None otherwise
    """
    return getattr(func, '_behavior_config', None)


def is_behavior_test(func: Callable) -> bool:
    """Check if a function is a behavior test.

    Args:
        func: Function to check

    Returns:
        True if decorated with @behavior
    """
    return getattr(func, '_is_behavior_test', False)


def get_captured_behavior(func: Callable) -> Optional[CapturedBehavior]:
    """Get captured behavior from a recently executed test function.

    Args:
        func: The decorated test function (after execution)

    Returns:
        CapturedBehavior with output and metadata, or None if not executed
    """
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


# Convenience exports
__all__ = [
    'behavior',
    'serialize_inputs',
    'get_behavior_config',
    'is_behavior_test',
    'get_captured_behavior',
]        obj = getattr(e, 'obj', None)
        if obj is not None:
            obj_type = type(obj).__name__
        else:
            # Try to extract from error message
            msg = str(e)
            if 'datetime' in msg.lower():
                obj_type = 'datetime'
            elif 'is not JSON serializable' in msg:
                # Extract type from message like "Object of type datetime is not JSON serializable"
                parts = msg.split('type ')
                if len(parts) > 1:
                    obj_type = parts[1].split()[0].rstrip('.')
                else:
                    obj_type = 'unknown'
            else:
                obj_type = 'unknown'
        raise SerializationError(obj_type, e) from e


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
            # Execute the test function
            result = func(*args, **kwargs)
            
            # CRITICAL: Validate return value
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
            
            # Serialize inputs (FAIL FAST if not JSON-serializable)
            try:
                input_json = serialize_inputs(args, kwargs)
            except SerializationError:
                raise  # Re-raise with context
            
            # Store result for pytest plugin to retrieve via item.obj._behaviorci_result
            wrapper._behaviorci_result = result
            wrapper._behaviorci_input = (args, kwargs)
            wrapper._behaviorci_input_json = input_json
            
            return result
        
        # Attach configuration for pytest plugin discovery
        wrapper._behavior_config = BehaviorConfig(
            behavior_id=behavior_id,
            threshold=threshold,
            must_contain=must_contain,
            must_not_contain=must_not_contain,
            func=func
        )
        
        # Mark as behavior test for easy detection
        wrapper._is_behavior_test = True
        
        return wrapper
    
    return decorator


def get_behavior_config(func: Callable) -> Optional[BehaviorConfig]:
    """Get behavior configuration from a decorated function.
    
    Args:
        func: Potentially decorated function
        
    Returns:
        BehaviorConfig if decorated, None otherwise
    """
    return getattr(func, '_behavior_config', None)


def is_behavior_test(func: Callable) -> bool:
    """Check if a function is a behavior test.
    
    Args:
        func: Function to check
        
    Returns:
        True if decorated with @behavior
    """
    return getattr(func, '_is_behavior_test', False)


def get_captured_behavior(func: Callable) -> Optional[CapturedBehavior]:
    """Get captured behavior from a recently executed test function.
    
    Args:
        func: The decorated test function (after execution)
        
    Returns:
        CapturedBehavior with output and metadata, or None if not executed
    """
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


# Convenience exports
__all__ = [
    'behavior',
    'serialize_inputs',
    'get_behavior_config',
    'is_behavior_test',
    'get_captured_behavior',
]
