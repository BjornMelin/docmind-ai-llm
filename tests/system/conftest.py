import os
import pytest


def pytest_collection_modifyitems(config, items):
    if os.getenv("DOCMIND_RUN_SYSTEM") in {"1", "true", "TRUE", "yes"}:
        return
    skip_sys = pytest.mark.skip(reason="System tests are skipped by default; set DOCMIND_RUN_SYSTEM=1 to run.")
    for item in items:
        # coarse path check to avoid marker dependency
        if str(item.fspath).endswith("tests/system") or "/tests/system/" in str(item.fspath):
            item.add_marker(skip_sys)

