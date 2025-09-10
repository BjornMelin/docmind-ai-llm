"""Unit tests for router_factory.build_router_engine."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.retrieval.router_factory import build_router_engine


class _VecIndex:
    def as_query_engine(self):
        return MagicMock(name="vector_qe")


class _HealthyStore:
    def get_nodes(self):
        yield {"id": "n1"}


class _PgIndex:
    def __init__(self, healthy=True) -> None:
        self.property_graph_store = _HealthyStore() if healthy else None

    def as_query_engine(self, include_text=True):
        del include_text
        return MagicMock(name="graph_qe")


def _tool_count(router) -> int:
    # Try public then private attributes; fallback to 0 if not found
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


@pytest.mark.unit
def test_build_router_engine_with_graph() -> None:
    vec = _VecIndex()
    pg = _PgIndex(healthy=True)
    router = build_router_engine(vec, pg, settings=None)
    assert _tool_count(router) == 2


@pytest.mark.unit
def test_build_router_engine_vector_only_fallback() -> None:
    vec = _VecIndex()
    pg = _PgIndex(healthy=False)
    router = build_router_engine(vec, pg, settings=None)
    assert _tool_count(router) == 1
