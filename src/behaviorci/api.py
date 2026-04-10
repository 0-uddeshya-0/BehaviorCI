"""Public API for BehaviorCI - the @behavior decorator."""

import functools
import json
import inspect
from typing import Callable, Optional, List, Any, Tuple

from .models import BehaviorConfig, CapturedBehavior
from .exceptions import SerializationError, ConfigurationError


def serialize_inputs(args: Tuple, kwargs: dict) -> str:
    data = {"args": args, "kwargs": kwargs}
    try:
        return json.dumps(data, sort_keys=True)
    except TypeError as e:
        obj = getattr(e, "obj", None)
        if obj is not None:
            obj_type = type(obj).__name__
        else:
            msg = str(e)
            if "datetime" in msg.lower():
                obj_type = "datetime"
            elif "is not JSON serializable" in msg:
                parts = msg.split("type ")
                if len(parts) > 1:
                    obj_type = parts[1].split()[0].rstrip(".")
                else:
                    obj_type = "unknown"
            else:
                obj_type = "unknown"
        raise SerializationError(obj_type, e) from e


def behavior(
    behavior_id: str,
    threshold: float = 0.85,
    must_contain: Optional[List[str]] = None,
    must_not_contain: Optional[List[str]] = None,
    samples: int = 1,
) -> Callable:
    if not behavior_id or not isinstance(behavior_id, str):
        raise ConfigurationError("behavior_id must be a non-empty string")

    if not (0.0 <= threshold <= 1.0):
        raise ConfigurationError("threshold must be between 0.0 and 1.0")

    if samples < 1:
        raise ConfigurationError("samples must be at least 1")

    def decorator(func: Callable) -> Callable:

        def _validate_and_store(
            wrapper_func: Callable, result: Any, args: Tuple, kwargs: dict
        ) -> None:
            if result is None:
                raise ConfigurationError(
                    f"Test function '{func.__name__}' returned None. "
                    f"BehaviorCI tests must return the LLM output string."
                )

            if samples > 1:
                if not isinstance(result, list) or not all(isinstance(x, str) for x in result):
                    raise ConfigurationError(
                        f"Multi-sample test '{func.__name__}' must internally return a list of strings."
                    )
            else:
                if not isinstance(result, str):
                    raise ConfigurationError(
                        f"Test function '{func.__name__}' returned {type(result).__name__}, "
                        f"expected str. BehaviorCI tests must return the LLM output string."
                    )

            input_json = serialize_inputs(args, kwargs)

            wrapper_func._behaviorci_result = result  # type: ignore[attr-defined]
            wrapper_func._behaviorci_input = (args, kwargs)  # type: ignore[attr-defined]
            wrapper_func._behaviorci_input_json = input_json  # type: ignore[attr-defined]

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if samples > 1:
                    result = [await func(*args, **kwargs) for _ in range(samples)]
                else:
                    result = await func(*args, **kwargs)
                _validate_and_store(async_wrapper, result, args, kwargs)
                return result[0] if samples > 1 else result

            wrapper = async_wrapper

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                if samples > 1:
                    result = [func(*args, **kwargs) for _ in range(samples)]
                else:
                    result = func(*args, **kwargs)
                _validate_and_store(sync_wrapper, result, args, kwargs)
                return result[0] if samples > 1 else result

            wrapper = sync_wrapper

        wrapper._behavior_config = BehaviorConfig(  # type: ignore[attr-defined]
            behavior_id=behavior_id,
            threshold=threshold,
            must_contain=must_contain,
            must_not_contain=must_not_contain,
            func=func,
            samples=samples,
        )

        wrapper._is_behavior_test = True  # type: ignore[attr-defined]

        return wrapper

    return decorator


def get_behavior_config(func: Callable) -> Optional[BehaviorConfig]:
    return getattr(func, "_behavior_config", None)


def is_behavior_test(func: Callable) -> bool:
    return getattr(func, "_is_behavior_test", False)


def get_captured_behavior(func: Callable) -> Optional[CapturedBehavior]:
    config = get_behavior_config(func)
    if config is None:
        return None

    result = getattr(func, "_behaviorci_result", None)
    if result is None:
        return None

    args, kwargs = getattr(func, "_behaviorci_input", ((), {}))

    return CapturedBehavior(
        output_text=result,
        args=args,
        kwargs=kwargs,
        behavior_id=config.behavior_id,
        threshold=config.threshold,
        must_contain=config.must_contain,
        must_not_contain=config.must_not_contain,
    )


__all__ = [
    "behavior",
    "serialize_inputs",
    "get_behavior_config",
    "is_behavior_test",
    "get_captured_behavior",
]
