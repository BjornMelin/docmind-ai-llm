"""Persistence test fixtures to ensure order-independent module patching."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType

import pytest


@pytest.fixture(autouse=True)
def llamaindex_module_registry() -> Iterator[pytest.MonkeyPatch]:
    """Isolate LlamaIndex registry entries and restore exact package identity."""
    registry = pytest.MonkeyPatch()
    llama_index = sys.modules.get("llama_index")
    core = sys.modules.get("llama_index.core")

    if isinstance(core, ModuleType):
        registry.delattr(core, "graph_stores", raising=False)
    if isinstance(llama_index, ModuleType):
        registry.delattr(llama_index, "core", raising=False)
    registry.delitem(sys.modules, "llama_index.core.graph_stores", raising=False)
    registry.delitem(sys.modules, "llama_index.core", raising=False)
    try:
        yield registry
    finally:
        registry.undo()
