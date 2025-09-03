"""Modality-aware reranking utilities per ADR-037.

This module provides minimal, library-first reranking builders using
LlamaIndex postprocessors and a small multimodal postprocessor wrapper
to gate visual vs text nodes and fuse results.
"""

from __future__ import annotations

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.colpali_rerank import ColPaliRerank

from src.config import settings


def build_text_reranker(top_n: int | None = None) -> SentenceTransformerRerank:
    """Create text CrossEncoder reranker (BGE v2-m3)."""
    k = int(top_n if top_n is not None else settings.retrieval.reranking_top_k)
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=k,
        use_fp16=True,
        normalize=settings.retrieval.reranker_normalize_scores,
    )


def build_visual_reranker(top_n: int | None = None) -> ColPaliRerank:
    """Create visual reranker (ColPali)."""
    k = int(top_n if top_n is not None else settings.retrieval.reranking_top_k)
    return ColPaliRerank(model="vidore/colpali-v1.2", top_n=k)


class MultimodalReranker(BaseNodePostprocessor):
    """Postprocessor that applies text and visual rerankers per node modality."""

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        # Split nodes by modality
        text_nodes = [
            n for n in nodes if n.node.metadata.get("modality", "text") == "text"
        ]
        visual_nodes = [
            n
            for n in nodes
            if n.node.metadata.get("modality") in {"image", "pdf_page_image"}
        ]

        out: list[NodeWithScore] = []
        mode = getattr(settings.retrieval, "reranker_mode", "auto")

        if mode in {"auto", "text"} and text_nodes:
            out += build_text_reranker().postprocess_nodes(
                text_nodes, query_str=query_bundle.query_str
            )

        if mode in {"auto", "multimodal"} and visual_nodes:
            out += build_visual_reranker().postprocess_nodes(
                visual_nodes, query_str=query_bundle.query_str
            )

        if not out:
            return nodes[: settings.retrieval.reranking_top_k]

        # Fuse and deduplicate by node id, keeping best score
        best: dict[str, NodeWithScore] = {}
        for n in out:
            nid = n.node.node_id
            cur = best.get(nid)
            if cur is None or (n.score or 0.0) > (cur.score or 0.0):
                best[nid] = n

        ranked = sorted(best.values(), key=lambda x: x.score or 0.0, reverse=True)
        return ranked[: settings.retrieval.reranking_top_k]


__all__ = [
    "MultimodalReranker",
    "build_text_reranker",
    "build_visual_reranker",
]
