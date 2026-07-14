"""Unit tests for router_factory depth policy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("llama_index.core", reason="requires llama_index.core")

from src.retrieval.router_factory import build_router_engine

from .conftest import get_router_tool_names

pytestmark = pytest.mark.requires_llama


class _VecIndex:
    def as_query_engine(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MagicMock(name="vector_qe")


class _PgIndex:
    def __init__(self) -> None:
        self.property_graph_store = object()


def test_graph_depth_uses_graphrag_cfg(
    monkeypatch: pytest.MonkeyPatch, router_settings
) -> None:  # type: ignore[no-untyped-def]
    """Graph tool respects settings.graphrag_cfg.default_path_depth."""
    captured: dict[str, object] = {}

    def _fake_build_graph_query_engine(pg_index, **kwargs):  # type: ignore[no-untyped-def]
        del pg_index
        captured.update(kwargs)
        return MagicMock(query_engine=MagicMock(name="graph_qe"))

    monkeypatch.setattr(
        "src.retrieval.router_factory.build_graph_query_engine",
        _fake_build_graph_query_engine,
        raising=True,
    )
    monkeypatch.setattr(
        "src.retrieval.router_factory.get_postprocessors",
        lambda *_a, **_k: [],
        raising=True,
    )

    router_settings.retrieval.top_k = 8
    router_settings.graphrag_cfg.default_path_depth = 3

    build_router_engine(_VecIndex(), pg_index=_PgIndex(), settings=router_settings)
    assert captured.get("path_depth") == 3


def test_graph_disabled_when_store_missing(router_settings) -> None:  # type: ignore[no-untyped-def]
    """No graph tool added when property_graph_store is absent."""

    class _PgMissing:
        property_graph_store = None

    router = build_router_engine(
        _VecIndex(),
        pg_index=_PgMissing(),
        settings=router_settings,
    )
    assert get_router_tool_names(router) == ["semantic_search"]
