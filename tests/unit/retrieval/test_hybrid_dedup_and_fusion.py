"""Unit tests for hybrid retriever dedup behavior and fusion selection."""

from __future__ import annotations

import importlib

import pytest


def test_hybrid_dedup_keeps_highest_score(monkeypatch):  # type: ignore[no-untyped-def]
    hmod = importlib.import_module("src.retrieval.hybrid")
    monkeypatch.setattr(
        hmod,
        "ensure_hybrid_collection",
        lambda *_args, **_kwargs: type("_Compatibility", (), {"compatible": True})(),
    )

    class _Point:
        def __init__(self, pid: str, score: float):
            self.id = pid
            self.score = score
            self.payload = {"page_id": pid, "text": pid}

    class _Res:
        def __init__(self):
            self.groups = [
                type("_Group", (), {"hits": [_Point("p1", 0.9)]})(),
                type("_Group", (), {"hits": [_Point("p2", 0.5)]})(),
            ]

    class _Client:
        def query_points_groups(self, **_kwargs):  # type: ignore[no-untyped-def]
            return _Res()

        def close(self):  # type: ignore[no-untyped-def]
            return None

    params = hmod.HybridParams(
        collection="c",
        fused_top_k=5,
        prefetch_sparse=2,
        prefetch_dense=2,
        fusion_mode="rrf",
        dedup_key="page_id",
    )
    retr = hmod.ServerHybridRetriever(params, client=_Client())
    out = retr.retrieve("q")
    # Ensure only one of p1 remains and it's the higher score one
    ids = [n.node.node_id for n in out]
    assert len([i for i in ids if i == "p1"]) == 1


@pytest.mark.parametrize(
    ("mode", "expected"),
    [("rrf", "RRF"), ("dbsf", "DBSF")],
)
def test_fusion_selection(monkeypatch, mode: str, expected: str):  # type: ignore[no-untyped-def]
    hmod = importlib.import_module("src.retrieval.hybrid")
    monkeypatch.setattr(
        hmod,
        "ensure_hybrid_collection",
        lambda *_args, **_kwargs: type("_Compatibility", (), {"compatible": True})(),
    )
    params = hmod.HybridParams(collection="c", fusion_mode=mode)
    retriever = hmod.ServerHybridRetriever(params, client=lambda: None)  # type: ignore[arg-type]
    query = retriever._fusion()
    if mode == "rrf":
        assert query.rrf.k == 60
    else:
        assert query.fusion.name == expected
