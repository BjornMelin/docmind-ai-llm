"""Unit tests for helper functions in src.retrieval.reranking.

Targets fast, deterministic paths: timeouts, parsing, and modality split.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.config.embedding_defaults import (
    DEFAULT_BGE_RERANKER_MODEL_ID,
    DEFAULT_BGE_RERANKER_MODEL_REVISION,
)
from src.retrieval import reranking as rr


def test_parse_top_k_invalid_raises():
    """_parse_top_k raises ValueError on non-int strings; None uses settings."""
    with pytest.raises(ValueError, match="Invalid top_n"):
        rr._parse_top_k("nope")


def test_parse_top_k_none_uses_settings(monkeypatch):
    monkeypatch.setattr(rr.settings.retrieval, "reranking_top_k", 7, raising=False)
    assert rr._parse_top_k(None) == 7


def test_split_by_modality_separates_text_and_visual():
    a = NodeWithScore(node=TextNode(text="a", id_="a"), score=0.1)
    b = NodeWithScore(node=TextNode(text="b", id_="b"), score=0.2)
    b.node.metadata["modality"] = "pdf_page_image"
    t, v = rr.MultimodalReranker._split_by_modality([a, b])
    assert len(t) == 1
    assert len(v) == 1


def test_get_siglip_prune_m_falls_back_on_bad_setting(monkeypatch):
    monkeypatch.setattr(rr.settings.retrieval, "siglip_prune_m", "nope", raising=False)
    assert rr._get_siglip_prune_m() == rr.SIGLIP_PRUNE_M


def test_build_text_reranker_falls_back_to_noop(monkeypatch):
    monkeypatch.setattr(
        rr,
        "_build_text_reranker_cached",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("offline")),
    )
    rer = rr.build_text_reranker(2)
    nodes = [
        NodeWithScore(node=TextNode(text=str(i), id_=str(i)), score=0.0)
        for i in range(5)
    ]
    out = rer.postprocess_nodes(nodes, query_str="q")
    assert len(out) == 2


def test_build_text_reranker_uses_pinned_local_snapshot(monkeypatch, tmp_path):
    """Resolve the canonical CrossEncoder from the configured cache only."""
    snapshot_path = tmp_path / "snapshot"
    snapshot_calls: list[dict[str, object]] = []
    constructor_calls: list[dict[str, object]] = []

    def _snapshot_download(**kwargs):  # type: ignore[no-untyped-def]
        snapshot_calls.append(kwargs)
        return str(snapshot_path)

    def _constructor(**kwargs):  # type: ignore[no-untyped-def]
        constructor_calls.append(kwargs)
        return SimpleNamespace()

    rr._build_text_reranker_cached.cache_clear()
    monkeypatch.setattr(rr, "snapshot_download", _snapshot_download)
    monkeypatch.setattr(rr, "SentenceTransformerRerank", _constructor)

    rr._build_text_reranker_cached(
        3,
        DEFAULT_BGE_RERANKER_MODEL_ID,
        str(tmp_path),
    )

    assert snapshot_calls == [
        {
            "repo_id": DEFAULT_BGE_RERANKER_MODEL_ID,
            "revision": DEFAULT_BGE_RERANKER_MODEL_REVISION,
            "cache_dir": str(tmp_path),
            "local_files_only": True,
        }
    ]
    assert constructor_calls == [
        {
            "model": str(snapshot_path),
            "top_n": 3,
            "trust_remote_code": False,
        }
    ]


def test_postprocess_nodes_returns_input_when_query_missing():
    nodes = [NodeWithScore(node=TextNode(text="a", id_="a"), score=0.1)]
    assert rr.MultimodalReranker()._postprocess_nodes(nodes, None) == nodes


def test_fuse_and_dedup_skips_duplicate_node_ids(monkeypatch):
    a1 = NodeWithScore(node=TextNode(text="a", id_="a"), score=1.0)
    a2 = NodeWithScore(node=TextNode(text="a2", id_="a"), score=0.5)
    b = NodeWithScore(node=TextNode(text="b", id_="b"), score=0.2)

    monkeypatch.setattr(rr, "rrf_merge", lambda *_a, **_k: [a1, a2, b], raising=True)
    out = rr.MultimodalReranker()._fuse_and_dedup([[a1], [a2], [b]])
    assert [n.node.node_id for n in out] == ["a", "b"]
