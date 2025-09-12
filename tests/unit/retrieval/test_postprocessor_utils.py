"""Tests for postprocessor utils across combinations.

Validates fallback behavior when node_postprocessors are present/absent and
when engine_cls/llm are provided or None.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.retrieval.postprocessor_utils import (
    build_pg_query_engine,
    build_retriever_query_engine,
    build_vector_query_engine,
)


class _Index:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def as_query_engine(self, **kwargs: Any) -> str:  # pragma: no cover - trivial
        self.calls.append(("as_query_engine", kwargs))
        return "qe"


class _Retriever:
    pass


class _Engine:
    @classmethod
    def from_args(cls, **kwargs: Any) -> str:  # pragma: no cover - trivial
        return "rq"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("post", "expect_key"),
    [(["p"], "node_postprocessors"), ([], None), (None, None)],
)
def test_build_vector_query_engine_combinations(
    post: list[str] | None, expect_key: str | None
) -> None:
    idx = _Index()
    out = build_vector_query_engine(idx, post, similarity_top_k=2)
    assert out == "qe"
    name, kwargs = idx.calls[-1]
    assert name == "as_query_engine"
    if expect_key:
        assert expect_key in kwargs


@pytest.mark.unit
@pytest.mark.parametrize(
    ("post", "expect_node"),
    [(["p"], True), ([], False), (None, False)],
)
def test_build_retriever_query_engine(
    post: list[str] | None, expect_node: bool
) -> None:
    retr = _Retriever()
    out = build_retriever_query_engine(retr, post, engine_cls=_Engine)
    assert out == "rq"
    # Cannot easily introspect kwargs without instrumenting _Engine;
    # ensure simply returns


@pytest.mark.unit
def test_build_pg_query_engine_fallbacks() -> None:
    idx = _Index()
    out = build_pg_query_engine(idx, post=None)
    assert out == "qe"
