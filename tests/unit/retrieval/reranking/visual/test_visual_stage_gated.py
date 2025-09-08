"""Visual stage gating: ensure no SigLIP load when no visual nodes.

We patch _load_siglip to crash if called, and assert the reranker proceeds
with text-only path without touching SigLIP.
"""

from __future__ import annotations

from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.retrieval import reranking as rr


def test_visual_stage_not_invoked_without_visual_nodes(monkeypatch):
    calls = {"text": 0}

    class _TextOnly:
        def postprocess_nodes(self, nodes, query_str):
            calls["text"] += 1
            # Reverse to signal text path executed
            return list(reversed(nodes))

    # If SigLIP is attempted, raise
    monkeypatch.setattr(
        rr,
        "_load_siglip",
        lambda: (_ for _ in ()).throw(RuntimeError("should not load siglip")),
    )
    monkeypatch.setattr(rr, "build_text_reranker", lambda *a, **k: _TextOnly())

    nodes = [
        NodeWithScore(node=TextNode(text="A"), score=0.1),
        NodeWithScore(node=TextNode(text="B"), score=0.2),
    ]
    out = rr.MultimodalReranker()._postprocess_nodes(nodes, QueryBundle(query_str="q"))
    assert [n.node.text for n in out] == ["B", "A"]
    assert calls["text"] == 1
