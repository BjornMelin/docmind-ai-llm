"""ColPali enablement heuristic tests for visual fraction gating.

Covers non-ops_force path: enable only when visual fraction is high,
top-k is within threshold, and sufficient VRAM is reported.
"""

from __future__ import annotations

import importlib

from llama_index.core.schema import NodeWithScore, TextNode


def _mk_nodes(n: int, *, score: float = 0.1) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=f"t{i}"), score=score) for i in range(n)]


def test_colpali_visual_fraction_true_when_high_and_vram_ok(monkeypatch):
    """Enables when visual fraction >= 0.4, topk <= limit, and VRAM OK."""
    rr = importlib.import_module("src.retrieval.reranking")

    # Make VRAM check pass
    monkeypatch.setattr(rr, "_has_cuda_vram", lambda *_a, **_k: True)
    # Ensure top-k <= MAX
    monkeypatch.setattr(rr.settings.retrieval, "reranking_top_k", rr.COLPALI_TOPK_MAX)
    # No ops_force override path
    monkeypatch.setattr(rr.settings.retrieval, "enable_colpali", False, raising=False)

    visual_nodes = _mk_nodes(5, score=0.2)
    text_nodes = _mk_nodes(5, score=0.1)
    lists = [visual_nodes, text_nodes]

    assert (
        rr.MultimodalReranker._should_enable_colpali(visual_nodes, lists) is True
    )


def test_colpali_visual_fraction_false_when_low_even_with_vram(monkeypatch):
    """Does not enable when visual fraction < 0.4, even with VRAM OK and topk OK."""
    rr = importlib.import_module("src.retrieval.reranking")

    monkeypatch.setattr(rr, "_has_cuda_vram", lambda *_a, **_k: True)
    monkeypatch.setattr(rr.settings.retrieval, "reranking_top_k", rr.COLPALI_TOPK_MAX)
    monkeypatch.setattr(rr.settings.retrieval, "enable_colpali", False, raising=False)

    visual_nodes = _mk_nodes(1, score=0.2)
    text_nodes = _mk_nodes(9, score=0.1)
    lists = [visual_nodes, text_nodes]

    assert (
        rr.MultimodalReranker._should_enable_colpali(visual_nodes, lists) is False
    )

