"""System test collection tweaks.

Skips system tests unless `DOCMIND_RUN_SYSTEM=1` is set to keep CI fast.
"""

import os

import pytest


def pytest_collection_modifyitems(_config, items):
    """Skip system tests unless explicitly enabled via environment.

    Args:
        config: Pytest config object.
        items: Collected test items.
    """
    if os.getenv("DOCMIND_RUN_SYSTEM") in {"1", "true", "TRUE", "yes"}:
        return
    skip_sys = pytest.mark.skip(
        reason="System tests are skipped by default; set DOCMIND_RUN_SYSTEM=1 to run."
    )
    for item in items:
        # coarse path check to avoid marker dependency
        if str(item.fspath).endswith("tests/system") or "/tests/system/" in str(
            item.fspath
        ):
            item.add_marker(skip_sys)
