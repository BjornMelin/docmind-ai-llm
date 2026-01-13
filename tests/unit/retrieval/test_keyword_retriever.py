"""Unit tests for sparse-only keyword retriever (SPEC-025 / ADR-044)."""

from __future__ import annotations

import random
from unittest.mock import Mock

import pytest
from qdrant_client import models as qmodels
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse

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


@pytest.mark.unit
def test_keyword_retriever_retries_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pts = [_Point("1", 0.9, {"text": "one"})]

    class _Client:
        def __init__(self) -> None:
            self.calls = 0

        def query_points(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.calls += 1
            if self.calls == 1:
                raise ResourceExhaustedResponse("rate limit", 0)
            return _Resp(pts)

        def close(self) -> None:
            return None

    sleep_calls: list[float] = []
    monkeypatch.setattr(
        "src.retrieval.keyword.time.sleep",
        lambda s: sleep_calls.append(float(s)),
    )
    monkeypatch.setattr(
        "src.retrieval.keyword.encode_to_qdrant",
        lambda _t: qmodels.SparseVector(indices=[1], values=[1.0]),
    )
    monkeypatch.setattr("src.retrieval.keyword.log_jsonl", lambda _ev: None)

    params = KeywordParams(collection="col", top_k=1, rate_limit_retries=1)
    retr = KeywordSparseRetriever(params, client_factory=_Client)  # type: ignore[arg-type]

    out = retr.retrieve("q")
    assert [n.node.text for n in out] == ["one"]
    assert sleep_calls == [0.0]


@pytest.mark.unit
def test_keyword_close_is_noop_when_client_missing() -> None:
    params = KeywordParams(collection="col")
    created: list[object] = []

    def _factory():  # pragma: no cover - trivial
        created.append(object())
        return Mock()

    retr = KeywordSparseRetriever(params, client_factory=_factory)
    retr.close()
    assert created == []


@pytest.mark.unit
def test_keyword_retrieve_handles_qdrant_query_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Mock()
    params = KeywordParams(collection="col", top_k=2)
    retr = KeywordSparseRetriever(params, client=client)

    class _Boom(ResourceExhaustedResponse):
        pass

    monkeypatch.setattr(
        "src.retrieval.keyword.encode_to_qdrant",
        lambda _t: qmodels.SparseVector(indices=[1], values=[1.0]),
    )
    monkeypatch.setattr(
        retr,
        "_query_points_with_retry",
        lambda *_a, **_k: (_ for _ in ()).throw(_Boom("rate", 0)),
    )

    events: list[dict] = []
    monkeypatch.setattr("src.retrieval.keyword.log_jsonl", lambda ev: events.append(ev))

    out = retr.retrieve("q")
    assert out == []
    assert any(ev.get("retrieval.error_type") == "_Boom" for ev in events)


@pytest.mark.unit
def test_query_points_with_retry_raises_after_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Client:
        def query_points(self, **_kwargs):  # type: ignore[no-untyped-def]
            raise ResourceExhaustedResponse("rate limit", 0)

        def close(self) -> None:
            return None

    monkeypatch.setattr("src.retrieval.keyword.time.sleep", lambda *_a, **_k: None)
    retr = KeywordSparseRetriever(
        KeywordParams(collection="col", rate_limit_retries=0),
        client_factory=_Client,  # type: ignore[arg-type]
    )

    with pytest.raises(ResourceExhaustedResponse):
        retr._query_points_with_retry(qmodels.SparseVector(indices=[1], values=[1.0]))


@pytest.mark.unit
def test_rate_limit_delay_falls_back_to_exponential_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    params = KeywordParams(
        collection="col",
        rate_limit_backoff_base_s=1.0,
        rate_limit_backoff_max_s=8.0,
    )
    retr = KeywordSparseRetriever(params, client=Mock())

    class _TestError(Exception):
        pass

    monkeypatch.setattr(random, "random", lambda: 0.0)

    exc = _TestError("rate limit", "bad")
    delay = retr._rate_limit_delay(exc, attempt=1)
    assert delay == 1.0


@pytest.mark.unit
def test_emit_telemetry_swallows_io_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    params = KeywordParams(collection="col", top_k=1)
    retr = KeywordSparseRetriever(params, client=Mock())

    monkeypatch.setattr(
        "src.retrieval.keyword.log_jsonl",
        lambda _ev: (_ for _ in ()).throw(OSError("nope")),
    )
    retr._emit_telemetry(t0=0.0, return_count=0, sparse_fallback=False, error_type="X")
