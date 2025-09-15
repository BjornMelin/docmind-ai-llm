"""Ensure settings.retrieval.siglip_prune_m is honored for ColPali input.

This test patches the SigLIP stage to return the full visual list unchanged,
so the ColPali preparation path (base slicing) determines the input length.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode


@pytest.mark.unit
def test_colpali_prune_respects_setting(monkeypatch):
    import src.retrieval.reranking as rr

    # Stub settings with desired prune value
    prune_m = 5
    rr.settings = SimpleNamespace(
        retrieval=SimpleNamespace(
            reranking_top_k=10,
            rrf_k=60,
            enable_colpali=False,
            siglip_prune_m=prune_m,
        )
    )

    # Visual nodes > prune size
    visual_nodes = [
        NodeWithScore(node=TextNode(text=f"v{i}", id_=f"v{i}"), score=0.0)
        for i in range(12)
    ]
    for n in visual_nodes:
        n.node.metadata["modality"] = "pdf_page_image"

    # Query bundle required by API
    qb = QueryBundle(query_str="q")

    # Make SigLIP stage return the full visual list (no pruning here)
    monkeypatch.setattr(
        rr,
        "_siglip_rescore",
        lambda _q, nodes, _b: nodes,
        raising=True,
    )

    # Force ColPali enablement
    monkeypatch.setattr(
        rr.MultimodalReranker,
        "_should_enable_colpali",
        staticmethod(lambda _v, _l: True),
        raising=True,
    )

    captured = {"got": None}

    class _DummyColPali:
        def __init__(self, top_n: int) -> None:
            self.top_n = top_n

        def postprocess_nodes(self, nodes, **_):
            captured["got"] = len(nodes)
            return nodes[: self.top_n]

    # Replace factory to return our dummy
    monkeypatch.setattr(
        rr,
        "build_visual_reranker",
        lambda top_n=None: _DummyColPali(top_n or 10),
        raising=True,
    )

    # Build a mixed set: only visual should matter
    nodes = visual_nodes.copy()
    out = rr.MultimodalReranker()._postprocess_nodes(nodes, qb)

    # Verify ColPali saw exactly prune_m items
    assert captured["got"] == prune_m
    # Out should not be empty and length <= top_k
    assert out
    assert len(out) <= rr.settings.retrieval.reranking_top_k
