"""Unit tests for helper functions in src.retrieval.reranking.

Targets fast, deterministic paths: timeouts, parsing, modality split,
and ColPali enable heuristic with explicit overrides.
"""

from __future__ import annotations

import importlib
import time
from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import reranking as rr


def test_run_with_timeout_returns_value_and_none():
    """_run_with_timeout returns value when fast, None when exceeding budget."""

    def _fast():
        return 42

    def _slow():
        start = time.perf_counter()
        while (time.perf_counter() - start) < 0.05:
            pass
        return 1

    assert rr._run_with_timeout(_fast, timeout_ms=10_000) == 42
    assert rr._run_with_timeout(_slow, timeout_ms=1) is None


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


def test_should_enable_colpali_policy_and_override(monkeypatch):
    """Enable when visual fraction high and VRAM ok; ops override forces True."""
    # Monkeypatch module-local settings to a lightweight stub
    rr.settings = SimpleNamespace(
        retrieval=SimpleNamespace(reranking_top_k=10, rrf_k=60, enable_colpali=False)
    )

    # Ensure VRAM check returns True
    monkeypatch.setattr(rr, "has_cuda_vram", lambda *_, **__: True)

    # Build lists with visual fraction >= 0.4
    text_nodes = [
        NodeWithScore(node=TextNode(text="t", id_="t"), score=0.1) for _ in range(3)
    ]
    visual_nodes = [
        NodeWithScore(node=TextNode(text="i", id_="i"), score=0.1) for _ in range(2)
    ]
    # Mark visual modality
    for n in visual_nodes:
        n.node.metadata["modality"] = "pdf_page_image"

    lists = [text_nodes + visual_nodes]
    assert rr.MultimodalReranker._should_enable_colpali(visual_nodes, lists) is True

    # Ops override forces True even if VRAM returns False and fraction is low
    rr.settings.retrieval.enable_colpali = True
    monkeypatch.setattr(rr, "has_cuda_vram", lambda *_, **__: False)
    assert rr.MultimodalReranker._should_enable_colpali([], [text_nodes]) is True


def test_get_siglip_prune_m_falls_back_on_bad_setting(monkeypatch):
    monkeypatch.setattr(rr.settings.retrieval, "siglip_prune_m", "nope", raising=False)
    assert rr._get_siglip_prune_m() == rr.SIGLIP_PRUNE_M


def test_run_with_timeout_process_executor_path(monkeypatch):
    monkeypatch.setattr(
        rr.settings.retrieval, "rerank_executor", "process", raising=False
    )

    created: list[str] = []

    class _Future:
        def __init__(self, value: int) -> None:
            self._value = value

        def result(self, timeout: float | None = None):  # type: ignore[no-untyped-def]
            del timeout
            return self._value

        def cancel(self) -> bool:
            return True

    class _Exec:
        def __init__(self, max_workers: int = 1):  # type: ignore[no-untyped-def]
            created.append("process")

        def submit(self, fn):  # type: ignore[no-untyped-def]
            return _Future(int(fn()))

        def shutdown(self, **_k):  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr(rr, "ProcessPoolExecutor", _Exec, raising=True)

    assert rr._run_with_timeout(lambda: 41, timeout_ms=10_000) == 41
    assert created == ["process"]


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


def test_build_visual_reranker_cached_raises_value_error_when_missing(monkeypatch):
    rr._build_visual_reranker_cached.cache_clear()
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("nope")),
    )
    with pytest.raises(ValueError, match="ColPaliRerank not available"):
        rr._build_visual_reranker_cached(1)


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
