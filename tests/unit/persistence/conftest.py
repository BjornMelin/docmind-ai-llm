"""Persistence test fixtures to ensure order-independent module patching."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _clear_llamaindex_modules() -> Iterator[None]:
    """Ensure LlamaIndex modules are not cached before tests patch them."""
    targets = [
        "llama_index.core",
        "llama_index.core.graph_stores",
    ]
    for name in targets:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        # Leave cleared to avoid leaking state across tests
        for name in targets:
            sys.modules.pop(name, None)
