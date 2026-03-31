"""Test for CRITICAL-001: Single test execution (not double).

CRITICAL-001: pytest_runtest_call was calling item.runtest() manually,
but pytest's own runner also calls it. Result: every @behavior test ran twice.

FIX: Converted to hookwrapper=True with yield to let pytest run test once.

VERIFICATION: This test verifies the hook implementation is correct.
"""

import pytest


def test_hookwrapper_decorator_present():
    """Verify pytest_runtest_call uses hookwrapper=True.
    
    WHY: CRITICAL-001 fix - The hook must use hookwrapper=True to let
    pytest run the test normally, then capture the result afterwards.
    
    VERIFIED: The plugin module has hookwrapper decorator.
    """
    import inspect
    from behaviorci import plugin
    
    # Check that pytest_runtest_call is defined
    assert hasattr(plugin, 'pytest_runtest_call'), "pytest_runtest_call not found"
    
    # Check source code for @pytest.hookimpl(hookwrapper=True)
    source = inspect.getsource(plugin.pytest_runtest_call)
    assert "@pytest.hookimpl(hookwrapper=True)" in source or \
           "@pytest.hookimpl(tryfirst=True, hookwrapper=True)" in source, \
        "pytest_runtest_call must use @pytest.hookimpl(hookwrapper=True)"


def test_no_manual_runtest_call():
    """Verify the hook doesn't manually call item.runtest().
    
    WHY: CRITICAL-001 fix - Manual runtest() call caused double execution.
    The hook should use yield to let pytest run the test.
    
    VERIFIED: Source code inspection shows no item.runtest() call.
    """
    import inspect
    from behaviorci import plugin
    
    source = inspect.getsource(plugin.pytest_runtest_call)
    
    # Should NOT contain manual item.runtest() call (excluding docstring/comments)
    # Look for actual call pattern (not in quotes/comments)
    lines = source.split('\n')
    code_lines = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if '"""' in stripped:
            in_docstring = not in_docstring
            continue
        if not in_docstring and not stripped.startswith('#'):
            code_lines.append(line)
    
    code = '\n'.join(code_lines)
    assert "item.runtest()" not in code, \
        "CRITICAL-001 NOT FIXED: Manual item.runtest() call found"
    
    # SHOULD contain yield to let pytest run the test
    assert "yield" in source, \
        "CRITICAL-001 NOT FIXED: Missing yield for hookwrapper"
    
    # SHOULD check outcome.excinfo for exceptions
    assert "outcome.excinfo" in source, \
        "Should check outcome.excinfo for exception handling"
