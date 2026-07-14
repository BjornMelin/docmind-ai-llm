"""Basic unit tests for ServerHybridRetriever (determinism & dedup)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.base.base_retriever import BaseRetriever

from src.retrieval.hybrid import HybridParams, ServerHybridRetriever
from src.retrieval.sparse_query import SparseEncodingError


class _Point:
    def __init__(self, pid: str, score: float, payload: dict):
        self.id = pid
        self.score = score
        self.payload = payload


class _Resp:
    def __init__(self, points):
        self.groups = [type("_Group", (), {"hits": [point]})() for point in points]


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a minimal BaseEmbedding-compatible stub on Settings.embed_model."""
    monkeypatch.setattr(
        "src.retrieval.hybrid.check_hybrid_collection",
        lambda *_args, **_kwargs: SimpleNamespace(compatible=True),
    )
    monkeypatch.setattr(
        "src.retrieval.hybrid._encode_sparse_query",
        lambda _text: None,
    )
    from llama_index.core import Settings  # type: ignore

    try:
        # Prefer real base class when available to satisfy strict setters
        from llama_index.core.base.embeddings.base import (  # type: ignore
            BaseEmbedding,
        )

        class _Embed(BaseEmbedding):  # type: ignore[misc]
            def _get_text_embedding(self, text: str):  # type: ignore[no-untyped-def]
                del text
                return [0.1, 0.2, 0.3]

            def _get_query_embedding(self, text: str):  # type: ignore[no-untyped-def]
                del text
                return [0.1, 0.2, 0.3]

            async def _aget_text_embedding(self, text: str):  # pragma: no cover
                return self._get_text_embedding(text)

            async def _aget_query_embedding(self, query: str):  # pragma: no cover
                return self._get_query_embedding(query)

        Settings.embed_model = _Embed()  # type: ignore[attr-defined]
    except Exception:
        # Fallback: non-strict environments accept plain stubs
        class _Embed:  # type: ignore[too-many-instance-attributes]
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
    # Qdrant group_size=1 returns only the best hit for each page_id.
    resp = _Resp([pts[1], pts[2]])

    def _fake_query_points_groups(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        return resp

    params = HybridParams(collection="col", fused_top_k=10, dedup_key="page_id")
    retr = ServerHybridRetriever(params)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        retr._client, "query_points_groups", _fake_query_points_groups, raising=False
    )

    out = retr.retrieve("q")
    # Expect two nodes (X and Y) with deterministic ordering:
    # score descending, id ascending (by key mapping)
    assert len(out) == 2
    texts = [n.node.get_content() for n in out]
    # Highest for X is score 0.9 ("two") should appear first
    assert texts[0] == "two"


def test_hybrid_sparse_unavailable_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    # If sparse encoding is unavailable, use only the canonical dense prefetch.
    pts = [
        _Point("a", 0.5, {"page_id": "A", "text": "a"}),
    ]
    resp = _Resp(pts)

    calls: dict[str, object] = {}

    def _fake_query_points_groups(*_args, **kwargs):  # type: ignore[no-untyped-def]
        calls.update(kwargs)
        return resp

    params = HybridParams(collection="c")
    retr = ServerHybridRetriever(params)
    monkeypatch.setattr(retr, "_encode_sparse", lambda _t: None)
    monkeypatch.setattr(  # type: ignore[attr-defined]
        retr._client, "query_points_groups", _fake_query_points_groups, raising=False
    )
    out = retr.retrieve("q")
    assert len(out) == 1
    assert out[0].node.get_content() == "a"
    assert [prefetch.using for prefetch in calls["prefetch"]] == ["text-dense"]  # type: ignore[index,union-attr]


def test_hybrid_sparse_encoder_failure_falls_back_to_dense(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use dense retrieval when the optional BM42 query model cannot load."""
    calls: dict[str, object] = {}

    def _fail_sparse(_text: str) -> None:
        raise SparseEncodingError("query encoding failed")

    def _query(*_args: object, **kwargs: object) -> _Resp:
        calls.update(kwargs)
        return _Resp([_Point("a", 0.5, {"page_id": "A", "text": "dense"})])

    monkeypatch.setattr("src.retrieval.hybrid._encode_sparse_query", _fail_sparse)
    retriever = ServerHybridRetriever(HybridParams(collection="c"))
    monkeypatch.setattr(
        retriever._client,
        "query_points_groups",
        _query,
        raising=False,
    )

    assert [node.node.get_content() for node in retriever.retrieve("q")] == ["dense"]
    assert [prefetch.using for prefetch in calls["prefetch"]] == ["text-dense"]  # type: ignore[index,union-attr]


def test_hybrid_retriever_checks_schema_without_mutating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checks: list[tuple[str, int, bool]] = []

    class _Client:
        def close(self) -> None:
            return None

        def create_collection(self, **_kwargs: object) -> None:
            raise AssertionError("retrieval must not create collections")

        def update_collection(self, **_kwargs: object) -> None:
            raise AssertionError("retrieval must not mutate collections")

    def _check(
        _client: object,
        collection: str,
        *,
        dense_dim: int,
        sparse_enabled: bool,
    ) -> SimpleNamespace:
        checks.append((collection, dense_dim, sparse_enabled))
        return SimpleNamespace(compatible=True)

    monkeypatch.setattr("src.retrieval.hybrid.check_hybrid_collection", _check)

    retriever = ServerHybridRetriever(
        HybridParams(collection="canonical"),
        client=_Client(),  # type: ignore[arg-type]
    )

    assert checks == [("canonical", 1024, True)]
    retriever.close()


async def test_hybrid_retriever_owns_sync_embedding_and_uses_async_qdrant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SyncClient:
        def close(self) -> None:
            return None

    class _AsyncClient:
        def __init__(self) -> None:
            self.queries = 0
            self.closed = False

        async def query_points_groups(self, **_kwargs):  # type: ignore[no-untyped-def]
            self.queries += 1
            return _Resp([_Point("a", 0.8, {"page_id": "A", "text": "async"})])

        async def close(self) -> None:
            self.closed = True

    class _Embed:
        def get_query_embedding(self, _text: str) -> list[float]:
            return [0.1, 0.2, 0.3]

        async def aget_query_embedding(self, _text: str) -> list[float]:
            raise AssertionError("global async embedding path must not be used")

    async_client = _AsyncClient()
    monkeypatch.setattr(
        "src.retrieval.hybrid.check_hybrid_collection",
        lambda *_a, **_k: SimpleNamespace(compatible=True),
    )
    monkeypatch.setattr("src.retrieval.hybrid.get_settings_embed_model", _Embed)

    def _fail_sparse(_text: str) -> None:
        raise SparseEncodingError("query encoding failed")

    monkeypatch.setattr("src.retrieval.hybrid._encode_sparse_query", _fail_sparse)

    retriever = ServerHybridRetriever(
        HybridParams(collection="col"),
        client=_SyncClient(),  # type: ignore[arg-type]
        async_client=async_client,  # type: ignore[arg-type]
    )
    assert isinstance(retriever, BaseRetriever)
    assert [node.node.get_content() for node in await retriever.aretrieve("query")] == [
        "async"
    ]
    assert async_client.queries == 1

    await retriever.aclose()
    assert async_client.closed is True
