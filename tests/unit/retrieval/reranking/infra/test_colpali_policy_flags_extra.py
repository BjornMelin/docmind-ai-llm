"""Extra tests for ColPali policy heuristic flags."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

from llama_index.core.schema import NodeWithScore, TextNode


def test_colpali_policy_ops_force_and_topk(monkeypatch):
    rr = importlib.import_module("src.retrieval.reranking")

    # Provide a fake settings object to supply optional flag
    class _RetrCfg:
        reranking_top_k = rr.COLPALI_TOPK_MAX
        enable_colpali = True

    class _S:
        retrieval = _RetrCfg()

    monkeypatch.setattr(rr, "settings", _S(), raising=False)
    text_nodes = [NodeWithScore(node=TextNode(text="t"), score=0.1)]
    assert rr.MultimodalReranker._should_enable_colpali([], [text_nodes]) is True

    # Disable via topk gate
    class _RetrCfg2:
        reranking_top_k = rr.COLPALI_TOPK_MAX + 5
        enable_colpali = False

    monkeypatch.setattr(
        rr, "settings", SimpleNamespace(retrieval=_RetrCfg2()), raising=False
    )
    monkeypatch.setattr(rr, "has_cuda_vram", lambda *_a, **_k: True)
    assert (
        rr.MultimodalReranker._should_enable_colpali(text_nodes, [text_nodes]) is False
    )
