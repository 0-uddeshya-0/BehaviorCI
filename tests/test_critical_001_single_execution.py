"""Test for CRITICAL-001: Single test execution.

CRITICAL-001: pytest_runtest_call was calling item.runtest() manually,
but pytest's own runner also calls it. Result: every @behavior test ran twice.

FIX: Removed pytest_runtest_call entirely (FIX-006). The @behavior decorator
in api.py stores the return value as a function attribute during normal pytest
execution. plugin.py reads it in pytest_runtest_makereport after pytest's own
runner completes the call phase — no manual runtest() needed.
"""


def test_pytest_runtest_call_is_absent():
    """Verify pytest_runtest_call does NOT exist in the plugin.

    WHY: CRITICAL-001 fix removed the function entirely to prevent
    double-execution. If it exists, every @behavior test runs twice.

    VERIFIED: plugin.py has no pytest_runtest_call function.
    """
    from behaviorci import plugin

    assert not hasattr(plugin, 'pytest_runtest_call'), (
        "CRITICAL-001 NOT FIXED: pytest_runtest_call still exists in plugin.py. "
        "This causes every @behavior test to execute twice (double LLM calls). "
        "Remove the function — result capture happens in pytest_runtest_makereport."
    )


def test_makereport_reads_result_attribute():
    """Verify pytest_runtest_makereport reads _behaviorci_result from function attribute.

    WHY: This is the correct pattern after FIX-006. Instead of manually running
    the test and capturing output, we read the attribute that the @behavior
    wrapper sets during normal pytest execution.

    VERIFIED: Source of pytest_runtest_makereport calls getattr(..., '_behaviorci_result').
    """
    import inspect
    from behaviorci import plugin

    assert hasattr(plugin, 'pytest_runtest_makereport'), (
        "pytest_runtest_makereport must exist — it is the sole result-capture hook."
    )

    source = inspect.getsource(plugin.pytest_runtest_makereport)

    assert "_behaviorci_result" in source, (
        "pytest_runtest_makereport must read _behaviorci_result from the function attribute."
    )

    assert "_behaviorci_input_json" in source, (
        "pytest_runtest_makereport must read _behaviorci_input_json from the function attribute."
    )


def test_makereport_is_hookwrapper():
    """Verify pytest_runtest_makereport uses hookwrapper=True.

    WHY: The hookwrapper pattern (yield) is required to intercept pytest's
    report generation and modify the outcome on regression failure.
    """
    import inspect
    from behaviorci import plugin

    source = inspect.getsource(plugin.pytest_runtest_makereport)

    assert "hookwrapper=True" in source, (
        "pytest_runtest_makereport must use @pytest.hookimpl(..., hookwrapper=True)."
    )

    assert "yield" in source, (
        "pytest_runtest_makereport must yield to let pytest generate the base report."
    )


def test_no_manual_runtest_anywhere():
    """Verify item.runtest() is never called manually in the plugin.

    WHY: Any manual call to item.runtest() in a non-firstresult hook causes
    double execution. pytest's own runner handles this automatically.
    """
    import inspect
    from behaviorci import plugin

    source = inspect.getsource(plugin)

    # Strip comments to avoid false positives from explanatory text
    code_lines = [
        line for line in source.split('\n')
        if line.strip() and not line.strip().startswith('#')
    ]
    code = '\n'.join(code_lines)

    assert 'item.runtest()' not in code, (
        "CRITICAL-001: Manual item.runtest() call found in plugin.py. "
        "This causes double execution. Remove it."
    )
