"""Modality-aware reranking utilities per ADR-037.

This module provides minimal, library-first reranking builders using
LlamaIndex postprocessors and a small multimodal postprocessor wrapper
to gate visual vs text nodes and fuse results.
"""

from __future__ import annotations

from functools import cache

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.postprocessor.colpali_rerank import ColPaliRerank

from src.config import settings


def _parse_top_k(value: int | str | None) -> int:
    """Validate and convert ``top_n`` values to ``int``."""
    if value is None:
        return settings.retrieval.reranking_top_k
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid top_n: {value}") from exc


@cache
def _build_text_reranker_cached(top_n: int) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(
        model="BAAI/bge-reranker-v2-m3",
        top_n=top_n,
        use_fp16=True,
        normalize=settings.retrieval.reranker_normalize_scores,
    )


def build_text_reranker(top_n: int | str | None = None) -> SentenceTransformerRerank:
    """Create text CrossEncoder reranker (BGE v2-m3)."""
    k = _parse_top_k(top_n)
    return _build_text_reranker_cached(k)


@cache
def _build_visual_reranker_cached(top_n: int) -> ColPaliRerank:
    return ColPaliRerank(model="vidore/colpali-v1.2", top_n=top_n)


def build_visual_reranker(top_n: int | str | None = None) -> ColPaliRerank:
    """Create visual reranker (ColPali)."""
    k = _parse_top_k(top_n)
    try:
        return _build_visual_reranker_cached(k)
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - library quirk
        raise ValueError(f"ColPaliRerank initialization failed: {exc}") from exc


class MultimodalReranker(BaseNodePostprocessor):
    """Postprocessor that applies text and visual rerankers per node modality."""

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if not nodes or not query_bundle:
            return nodes

        text_nodes, visual_nodes = self._split_by_modality(nodes)
        reranked = self._invoke_rerankers(
            text_nodes, visual_nodes, query_bundle.query_str
        )
        if not reranked:
            return nodes[: settings.retrieval.reranking_top_k]
        return self._fuse_and_dedupe(reranked)

    @staticmethod
    def _split_by_modality(
        nodes: list[NodeWithScore],
    ) -> tuple[list[NodeWithScore], list[NodeWithScore]]:
        text = [n for n in nodes if n.node.metadata.get("modality", "text") == "text"]
        visual = [
            n
            for n in nodes
            if n.node.metadata.get("modality") in {"image", "pdf_page_image"}
        ]
        return text, visual

    @staticmethod
    def _invoke_rerankers(
        text_nodes: list[NodeWithScore],
        visual_nodes: list[NodeWithScore],
        query_str: str,
    ) -> list[NodeWithScore]:
        mode = getattr(settings.retrieval, "reranker_mode", "auto")
        out: list[NodeWithScore] = []
        if mode in {"auto", "text", "multimodal"} and text_nodes:
            out += build_text_reranker().postprocess_nodes(
                text_nodes, query_str=query_str
            )
        if mode in {"auto", "multimodal"} and visual_nodes:
            out += build_visual_reranker().postprocess_nodes(
                visual_nodes, query_str=query_str
            )
        return out

    @staticmethod
    def _fuse_and_dedupe(reranked: list[NodeWithScore]) -> list[NodeWithScore]:
        best: dict[str, NodeWithScore] = {}
        for n in reranked:
            nid = n.node.node_id
            cur = best.get(nid)
            n_score = n.score if n.score is not None else 0.0
            cur_score = cur.score if cur and cur.score is not None else 0.0
            if cur is None or n_score > cur_score:
                best[nid] = n
        ranked = sorted(best.values(), key=lambda x: x.score or 0.0, reverse=True)
        return ranked[: settings.retrieval.reranking_top_k]


__all__ = [
    "MultimodalReranker",
    "build_text_reranker",
    "build_visual_reranker",
]
