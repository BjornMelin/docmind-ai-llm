import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.getenv("DOCMIND_RUN_E2E") in {"1", "true", "TRUE", "yes"}:
        return
    skip_e2e = pytest.mark.skip(
        reason="E2E tests are skipped by default; set DOCMIND_RUN_E2E=1 to run."
    )
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)
