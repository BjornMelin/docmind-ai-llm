"""UI integration test fixtures for consistent import state.

Ensures modules that are patched within AppTest-based tests are not left in an
unexpected state due to prior imports in the overall test session. This keeps
Streamlit AppTest runs order-independent.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _reset_router_and_graph_modules() -> Iterator[None]:
    """Clear cached modules that these tests patch to avoid order sensitivity."""
    targets = [
        "src.retrieval.router_factory",
        "src.retrieval.graph_config",
        "src.ui.ingest_adapter",
        "src.utils.storage",
        "src.pages.02_documents",
    ]
    saved = {name: sys.modules.get(name) for name in targets}
    for name in targets:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        # Do not restore to keep later tests from seeing a partially patched module
        for name in targets:
            if name in sys.modules:
                continue
            # leave cleared; next importer will load a fresh module
            if saved.get(name) is not None:
                # ensure previous reference can't leak
                del saved[name]
