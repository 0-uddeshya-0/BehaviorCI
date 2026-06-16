"""Root conftest.

Enables the ``pytester`` fixture so the plugin's own behavior can be tested by
running pytest sessions in a temporary directory.
"""

pytest_plugins = ["pytester"]
