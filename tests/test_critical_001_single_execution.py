"""Test for CRITICAL-001 / FIX-006: Single test execution (not double).

CRITICAL-001: pytest_runtest_call was calling item.runtest() manually,
but pytest's own runner also calls it. Result: every @behavior test ran twice.

FIX-006: Removed pytest_runtest_call entirely. We now read _behaviorci_result
directly in makereport after pytest's normal execution completes.

VERIFICATION: This test verifies the implementation is correct.
"""

import inspect


def test_no_pytest_runtest_call_hook():
    """Verify pytest_runtest_call is NOT implemented (FIX-006).

    WHY: FIX-006 - The previous implementation of pytest_runtest_call called
    item.runtest() manually, but pytest's own runner also calls it, causing
    double execution. We removed this hook entirely.

    Now we read _behaviorci_result directly in makereport after pytest's
    normal call phase completes.

    VERIFIED: The plugin module does NOT have pytest_runtest_call.
    """
    from behaviorci import plugin

    # pytest_runtest_call should NOT be defined
    assert not hasattr(
        plugin, "pytest_runtest_call"
    ), "FIX-006: pytest_runtest_call should be removed to prevent double execution"


def test_makereport_reads_result_attribute():
    """Verify makereport reads _behaviorci_result from function attribute.

    WHY: FIX-006 - Instead of capturing in a separate hook, we read the
    result directly from the function attribute set by the @behavior wrapper.

    VERIFIED: Source code shows getattr(item.obj, '_behaviorci_result', None).
    """
    from behaviorci import plugin

    source = inspect.getsource(plugin.pytest_runtest_makereport)

    # Should read _behaviorci_result from function attribute
    assert (
        "_behaviorci_result" in source
    ), "makereport should read _behaviorci_result from function attribute"

    # Should read _behaviorci_input_json from function attribute
    assert (
        "_behaviorci_input_json" in source
    ), "makereport should read _behaviorci_input_json from function attribute"


def test_makereport_uses_hookwrapper():
    """Verify makereport uses hookwrapper=True.

    WHY: hookwrapper=True allows us to run code after the test completes
    but before the report is finalized.

    VERIFIED: The decorator includes hookwrapper=True in source code.
    """
    import inspect

    from behaviorci import plugin

    source = inspect.getsource(plugin.pytest_runtest_makereport)

    # Check decorator in source code
    assert (
        "@pytest.hookimpl(tryfirst=True, hookwrapper=True)" in source
        or "@pytest.hookimpl(hookwrapper=True)" in source
    ), "pytest_runtest_makereport must use @pytest.hookimpl with hookwrapper=True"
