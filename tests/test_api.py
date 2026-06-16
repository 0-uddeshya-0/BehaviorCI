"""Tests for the public API: input serialization and the @behavior decorator."""

import json
from datetime import datetime

import pytest

from behaviorci import behavior, get_behavior_config, get_captured_behavior, is_behavior_test
from behaviorci.api import serialize_inputs
from behaviorci.exceptions import ConfigurationError, SerializationError


class TestSerialization:
    def test_serializes_simple_types(self):
        result = serialize_inputs(("hello", 42, True), {"key": "value"})
        data = json.loads(result)
        assert data["args"] == ["hello", 42, True]
        assert data["kwargs"] == {"key": "value"}

    def test_serializes_nested_structures(self):
        result = serialize_inputs(({"nested": [1, 2, 3]},), {"outer": {"inner": "value"}})
        data = json.loads(result)
        assert data["args"][0]["nested"] == [1, 2, 3]

    def test_input_order_is_canonical(self):
        # Reordered kwargs must serialize identically so the snapshot id is stable.
        result1 = serialize_inputs(("a", "b"), {"z": 1, "a": 2})
        result2 = serialize_inputs(("a", "b"), {"a": 2, "z": 1})
        assert result1 == result2

    def test_datetime_raises_serialization_error(self):
        # We never silently coerce non-JSON inputs -- that would make the snapshot
        # id depend on str() formatting rather than the real value.
        with pytest.raises(SerializationError) as exc_info:
            serialize_inputs((datetime.now(),), {})
        message = str(exc_info.value).lower()
        assert "datetime" in message or "json-serializable" in message

    def test_custom_object_raises_serialization_error(self):
        class Custom:
            pass

        with pytest.raises(SerializationError):
            serialize_inputs((Custom(),), {})


class TestDecorator:
    def test_attaches_config(self):
        @behavior("greeting", threshold=0.9, must_contain=["hi"])
        def my_test():
            return "hi there"

        config = get_behavior_config(my_test)
        assert config is not None
        assert config.behavior_id == "greeting"
        assert config.threshold == 0.9
        assert config.must_contain == ["hi"]

    def test_is_behavior_test(self):
        @behavior("flag")
        def decorated():
            return "x"

        def plain():
            return "x"

        assert is_behavior_test(decorated) is True
        assert is_behavior_test(plain) is False

    def test_captures_return_value(self):
        @behavior("capture")
        def my_test():
            return "captured output"

        my_test()
        assert my_test._behaviorci_result == "captured output"

    def test_captures_inputs(self):
        @behavior("inputs")
        def my_test(a, b, c=None):
            return f"{a} {b} {c}"

        my_test("hello", "world", c="!")
        args, kwargs = my_test._behaviorci_input
        assert args == ("hello", "world")
        assert kwargs == {"c": "!"}

    def test_none_return_raises(self):
        @behavior("none")
        def my_test():
            return None

        with pytest.raises(ConfigurationError):
            my_test()

    def test_non_string_return_raises(self):
        @behavior("not_str")
        def my_test():
            return 123

        with pytest.raises(ConfigurationError):
            my_test()

    def test_non_serializable_input_raises(self):
        @behavior("bad_input")
        def my_test(when):
            return "ok"

        with pytest.raises(SerializationError):
            my_test(datetime.now())

    def test_get_captured_behavior(self):
        @behavior("cap", threshold=0.8, must_contain=["a"], must_not_contain=["b"])
        def my_test(name):
            return f"hello {name}"

        my_test("alice")
        captured = get_captured_behavior(my_test)
        assert captured is not None
        assert captured.output_text == "hello alice"
        assert captured.behavior_id == "cap"
        assert captured.threshold == 0.8
        assert captured.args == ("alice",)


class TestDecoratorValidation:
    def test_empty_behavior_id_raises(self):
        with pytest.raises(ConfigurationError):
            behavior("")

    def test_threshold_out_of_range_raises(self):
        with pytest.raises(ConfigurationError):
            behavior("x", threshold=1.5)
        with pytest.raises(ConfigurationError):
            behavior("x", threshold=-0.1)

    def test_samples_below_one_raises(self):
        with pytest.raises(ConfigurationError):
            behavior("x", samples=0)


class TestSamples:
    def test_samples_returns_first_but_captures_list(self):
        calls = {"n": 0}

        @behavior("sampled", samples=3)
        def my_test():
            calls["n"] += 1
            return f"variation {calls['n']}"

        returned = my_test()
        # The caller sees the first sample; BehaviorCI keeps all of them.
        assert returned == "variation 1"
        assert my_test._behaviorci_result == ["variation 1", "variation 2", "variation 3"]
        assert calls["n"] == 3

    def test_samples_must_return_strings(self):
        @behavior("sampled_bad", samples=2)
        def my_test():
            return 123

        with pytest.raises(ConfigurationError):
            my_test()
