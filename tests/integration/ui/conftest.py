"""UI integration test fixtures for consistent import state.

Ensures modules that are patched within AppTest-based tests are not left in an
unexpected state due to prior imports in the overall test session. This keeps
Streamlit AppTest runs order-independent.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from types import ModuleType
from typing import cast

import pytest

_MISSING = object()


def _restore_parent_attribute(
    name: str,
    parent: object,
    original: object,
) -> None:
    """Restore and verify one target's parent attribute exactly."""
    parent_name, _, attribute = name.rpartition(".")
    current_parent = sys.modules.get(parent_name, _MISSING)
    if (
        current_parent is not _MISSING
        and current_parent is not parent
        and hasattr(current_parent, attribute)
    ):
        delattr(current_parent, attribute)
    if parent is not _MISSING:
        if original is _MISSING:
            if hasattr(parent, attribute):
                delattr(parent, attribute)
        else:
            setattr(parent, attribute, original)
    current_parent = sys.modules.get(parent_name, _MISSING)
    if current_parent is not _MISSING and current_parent is not parent:
        assert getattr(current_parent, attribute, _MISSING) is _MISSING
    if parent is not _MISSING:
        assert getattr(parent, attribute, _MISSING) is original


@pytest.fixture(autouse=True)
def _stub_chat_embedding_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep UI tests offline; embedding integration has dedicated tests."""
    import src.config as config

    monkeypatch.setattr(config, "setup_llamaindex", lambda **_kwargs: None)


@pytest.fixture(autouse=True)
def _reset_router_and_graph_modules() -> Iterator[pytest.MonkeyPatch]:
    """Clear cached modules that these tests patch to avoid order sensitivity."""
    targets = [
        "llama_index.core",
        "src.agents.coordinator",
        "src.pages.01_chat",
        "src.retrieval.multimodal_fusion",
        "src.retrieval.router_factory",
        "src.retrieval.graph_config",
        "src.ui.ingest_adapter",
        "src.ui.chat_runtime",
        "src.ui.chat_sessions",
        "src.utils.storage",
        "src.pages.02_documents",
    ]
    saved = {name: sys.modules.get(name, _MISSING) for name in targets}
    parent_attributes: dict[str, tuple[object, object]] = {}
    for name in targets:
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name, _MISSING)
        original = (
            _MISSING if parent is _MISSING else getattr(parent, attribute, _MISSING)
        )
        parent_attributes[name] = (parent, original)
        if parent is not _MISSING and original is not _MISSING:
            delattr(parent, attribute)
        sys.modules.pop(name, None)
    registry_patch = pytest.MonkeyPatch()
    try:
        yield registry_patch
    finally:
        registry_patch.undo()
        # Restore the pre-test import state so UI tests can't contaminate other
        # test modules that imported these symbols during collection.
        for name in targets:
            original = saved[name]
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = cast(ModuleType, original)
        for name, (parent, original) in parent_attributes.items():
            _restore_parent_attribute(name, parent, original)
        for name, original in saved.items():
            assert sys.modules.get(name, _MISSING) is original


@pytest.fixture(autouse=True)
def _stub_graphrag_health_badge(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid GraphRAG imports during AppTest UI runs.

    Dedicated unit tests cover the required LlamaIndex API health check.
    """
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.get_graphrag_health",
        lambda *, force_refresh=False: (
            False,
            "unavailable",
            "GraphRAG disabled for AppTest",
        ),
        raising=False,
    )
