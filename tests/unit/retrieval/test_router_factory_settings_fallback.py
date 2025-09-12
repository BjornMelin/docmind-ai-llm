"""Tests for router_factory hybrid settings fallback behavior.

When enable_hybrid is None, the router_factory should fall back to
settings.retrieval.enable_server_hybrid.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.config.settings import settings
from src.retrieval.router_factory import build_router_engine


class _VecIndex:
    """Tiny stub exposing as_query_engine for vector index."""

    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):  # pragma: no cover - not used in this test
        yield {"id": "n1"}


class _PgIndex:
    """Tiny stub for a property graph index with query engine method."""

    def __init__(self) -> None:
        self.property_graph_store = _HealthyStore()

    def as_query_engine(self, include_text=True):  # type: ignore[no-untyped-def]
        del include_text
        return MagicMock(name="graph_qe")


def _tool_count(router) -> int:  # type: ignore[no-untyped-def]
    """Return the tool count from a RouterQueryEngine instance.

    Supports both public and private attributes used by LlamaIndex versions.
    """
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


@pytest.mark.parametrize("flag", [True, False])
def test_build_router_engine_settings_fallback(monkeypatch, flag: bool) -> None:  # type: ignore[no-untyped-def]
    """When enable_hybrid is None, consult settings.retrieval.enable_server_hybrid."""
    # Patch ServerHybridRetriever to avoid real Qdrant dependency

    class _DummyHybrid:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)

    # Toggle settings flag
    settings.retrieval.enable_server_hybrid = flag

    vec = _VecIndex()
    pg = _PgIndex()

    router = build_router_engine(vec, pg, settings=None, enable_hybrid=None)
    count = _tool_count(router)
    # Base tools: vector + kg
    expected = 3 if flag else 2
    assert count == expected, f"expected {expected} tools, got {count}"
