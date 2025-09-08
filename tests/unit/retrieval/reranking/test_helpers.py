"""Unit tests for helper functions in src.retrieval.reranking.

Targets fast, deterministic paths: timeouts, parsing, modality split,
and ColPali enable heuristic with explicit overrides.
"""

from __future__ import annotations

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
        time.sleep(0.05)
        return 1

    assert rr._run_with_timeout(_fast, timeout_ms=10_000) == 42
    assert rr._run_with_timeout(_slow, timeout_ms=1) is None


def test_parse_top_k_invalid_raises():
    """_parse_top_k raises ValueError on non-int strings; None uses settings."""
    with pytest.raises(ValueError, match="Invalid top_n"):
        rr._parse_top_k("nope")


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
    monkeypatch.setattr(rr, "_has_cuda_vram", lambda *_: True)

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
    monkeypatch.setattr(rr, "_has_cuda_vram", lambda *_: False)
    assert rr.MultimodalReranker._should_enable_colpali([], [text_nodes]) is True
