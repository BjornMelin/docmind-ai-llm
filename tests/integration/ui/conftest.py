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
        # Restore the pre-test import state so UI tests can't contaminate other
        # test modules that imported these symbols during collection.
        for name in targets:
            original = saved.get(name)
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
