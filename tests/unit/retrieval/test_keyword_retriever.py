"""Unit tests for sparse-only keyword retriever (SPEC-025 / ADR-044)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from qdrant_client import models as qmodels

from src.retrieval.keyword import KeywordParams, KeywordSparseRetriever


class _Point:
    def __init__(self, pid: str, score: float, payload: dict):
        self.id = pid
        self.score = score
        self.payload = payload


class _Resp:
    def __init__(self, points):
        self.points = points


@pytest.mark.unit
def test_keyword_sparse_unavailable_fails_open_and_emits_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    params = KeywordParams(collection="col", top_k=5)
    retr = KeywordSparseRetriever(params, client=client)

    events: list[dict] = []
    monkeypatch.setattr("src.retrieval.keyword.log_jsonl", lambda ev: events.append(ev))
    monkeypatch.setattr("src.retrieval.keyword.encode_to_qdrant", lambda _t: None)

    out = retr.retrieve("error-code-123")
    assert out == []
    client.query_points.assert_not_called()

    assert events
    ev = events[-1]
    assert ev["retrieval.tool"] == "keyword_search"
    assert ev["retrieval.vector"] == "text-sparse"
    assert ev["retrieval.sparse_fallback"] is True
    # Do not leak raw query text into telemetry
    assert "error-code-123" not in str(ev)


@pytest.mark.unit
def test_keyword_retriever_queries_qdrant_sparse_only_and_orders_deterministically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pts = [
        _Point("2", 0.5, {"text": "two"}),
        _Point("1", 0.5, {"text": "one"}),
        _Point("3", 0.9, {"text": "three"}),
    ]
    client = Mock()
    client.query_points.return_value = _Resp(pts)

    params = KeywordParams(collection="col", top_k=3)
    retr = KeywordSparseRetriever(params, client=client)

    events: list[dict] = []
    monkeypatch.setattr("src.retrieval.keyword.log_jsonl", lambda ev: events.append(ev))
    monkeypatch.setattr(
        "src.retrieval.keyword.encode_to_qdrant",
        lambda _t: qmodels.SparseVector(indices=[1], values=[1.0]),
    )

    out = retr.retrieve("q")
    assert [n.node.text for n in out] == ["three", "one", "two"]

    client.query_points.assert_called_once()
    _args, kwargs = client.query_points.call_args
    assert kwargs["collection_name"] == "col"
    assert kwargs["using"] == "text-sparse"
    assert kwargs["limit"] == 3
    assert "text" in kwargs["with_payload"]

    assert events
    ev = events[-1]
    assert ev["retrieval.sparse_fallback"] is False
    assert ev["retrieval.return_count"] == 3
