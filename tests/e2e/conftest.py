"""E2E test collection tweaks.

Skips E2E tests unless `DOCMIND_RUN_E2E=1` is set to keep CI fast.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Skip E2E tests unless explicitly enabled via environment."""
    del config  # unused; accepted for pytest hook signature compatibility
    if os.getenv("DOCMIND_RUN_E2E") in {"1", "true", "TRUE", "yes"}:
        return
    skip_e2e = pytest.mark.skip(
        reason="E2E tests are skipped by default; set DOCMIND_RUN_E2E=1 to run."
    )
    for item in items:
        # coarse path check to avoid marker dependency
        if str(item.fspath).endswith("tests/e2e") or "/tests/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
            item.add_marker(skip_e2e)
