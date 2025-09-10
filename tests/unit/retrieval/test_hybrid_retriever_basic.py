"""Basic unit tests for ServerHybridRetriever (determinism & dedup)."""

from __future__ import annotations

import pytest

from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams


class _Point:
    def __init__(self, pid: str, score: float, payload: dict):
        self.id = pid
        self.score = score
        self.payload = payload


class _Resp:
    def __init__(self, points):
        self.points = points


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub Settings.embed_model.get_query_embedding to return deterministic vector
    from llama_index.core import Settings  # type: ignore

    class _Embed:
        def get_query_embedding(self, text: str):  # type: ignore[no-untyped-def]
            del text
            return [0.1, 0.2, 0.3]

    Settings.embed_model = _Embed()  # type: ignore[attr-defined]


def test_hybrid_retriever_dedup_and_order(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare points with duplicate dedup_key (page_id) and ensure we pick highest score
    pts = [
        _Point("a", 0.8, {"page_id": "X", "text": "one"}),
        _Point("b", 0.9, {"page_id": "X", "text": "two"}),  # higher score for X
        _Point("c", 0.7, {"page_id": "Y", "text": "three"}),
    ]
    resp = _Resp(pts)

    # Patch QdrantClient.query_points on instance
    def _fake_query_points(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return resp

    params = _HybridParams(collection="col", fused_top_k=10, dedup_key="page_id")
    retr = ServerHybridRetriever(params)
    monkeypatch.setattr(retr._client, "query_points", _fake_query_points)  # type: ignore[attr-defined]

    out = retr.retrieve("q")
    # Expect two nodes (X and Y) with deterministic ordering:
    # score descending, id ascending (by key mapping)
    assert len(out) == 2
    texts = [n.node.text for n in out]
    # Highest for X is score 0.9 ("two") should appear first
    assert texts[0] == "two"


def test_hybrid_sparse_unavailable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # If sparse encoding unavailable, retriever still returns dense-prefetch results
    pts = [
        _Point("a", 0.5, {"page_id": "A", "text": "a"}),
    ]
    resp = _Resp(pts)

    def _fake_query_points(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return resp

    params = _HybridParams(collection="c")
    retr = ServerHybridRetriever(params)
    monkeypatch.setattr(retr, "_encode_sparse", lambda _t: None)
    monkeypatch.setattr(retr._client, "query_points", _fake_query_points)  # type: ignore[attr-defined]
    out = retr.retrieve("q")
    assert len(out) == 1
    assert out[0].node.text == "a"
