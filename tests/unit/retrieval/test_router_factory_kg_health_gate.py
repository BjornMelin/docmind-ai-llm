"""Test KG tool gating when GraphRAG construction fails."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval import router_factory as rf

from .conftest import get_router_tool_names


@pytest.mark.unit
def test_kg_tool_absent_when_builder_errors(
    monkeypatch: pytest.MonkeyPatch, router_settings
) -> None:  # type: ignore[no-untyped-def]
    """Graph tool is omitted when build_graph_query_engine raises."""

    def _broken_graph_builder(*_a: object, **_k: object) -> None:
        raise ValueError("broken graph index")

    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        _broken_graph_builder,
        raising=True,
    )

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return MagicMock(name="vector_qe")

    class _Pg:
        def __init__(self) -> None:
            self.property_graph_store = object()

    router = rf.build_router_engine(_Vec(), pg_index=_Pg(), settings=router_settings)
    assert get_router_tool_names(router) == ["semantic_search"]


@pytest.mark.unit
def test_kg_tool_present_when_builder_ok(
    monkeypatch: pytest.MonkeyPatch, router_settings
) -> None:  # type: ignore[no-untyped-def]
    """Graph tool is present when build_graph_query_engine succeeds."""
    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        lambda *_a, **_k: MagicMock(query_engine=MagicMock(name="graph_qe")),
        raising=True,
    )

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return MagicMock(name="vector_qe")

    class _Pg:
        def __init__(self) -> None:
            self.property_graph_store = object()

    router = rf.build_router_engine(_Vec(), pg_index=_Pg(), settings=router_settings)
    assert get_router_tool_names(router) == ["semantic_search", "knowledge_graph"]
