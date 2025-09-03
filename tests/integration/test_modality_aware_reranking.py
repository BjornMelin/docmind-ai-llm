"""Integration test: modality-aware reranking gating and fusion.

This test patches the reranker builders to avoid loading real models and
verifies that visual nodes are considered when mode is auto/multimodal.
"""

from __future__ import annotations

from typing import Any

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from src.config import settings
from src.retrieval import reranking as rr


class _FakePostprocessor(BaseNodePostprocessor):
    def __init__(self, kind: str) -> None:
        self.kind = kind

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        # Assign deterministic scores based on kind to validate gating
        scored: list[NodeWithScore] = []
        for n in nodes:
            base = 0.6 if self.kind == "text" else 0.9
            scored.append(NodeWithScore(node=n.node, score=base))
        return scored


def test_multimodal_gating_prefers_visual_nodes_when_present(monkeypatch: Any) -> None:
    """Visual node should rank higher when visual reranker is applied."""
    # Patch builders to avoid model loading and control scores
    monkeypatch.setattr(
        rr, "build_text_reranker", lambda top_n=None: _FakePostprocessor("text")
    )
    monkeypatch.setattr(
        rr, "build_visual_reranker", lambda top_n=None: _FakePostprocessor("visual")
    )

    # Build mixed nodes: 1 text, 1 image
    tnode = NodeWithScore(node=TextNode(text="Report content", id_="t1"), score=0.1)
    inode = NodeWithScore(node=TextNode(text="Image placeholder", id_="i1"), score=0.1)
    inode.node.metadata["modality"] = "pdf_page_image"
    nodes = [tnode, inode]

    settings.retrieval.reranking_top_k = 2
    settings.retrieval.reranker_mode = "auto"

    ranked = rr.MultimodalReranker().postprocess_nodes(
        nodes, QueryBundle(query_str="What does the figure show?")
    )
    assert len(ranked) == 2
    # Visual node should appear with higher score due to our fake visual reranker
    assert ranked[0].node.node_id in {"i1"}
