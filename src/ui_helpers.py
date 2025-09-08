"""UI helper utilities."""

from __future__ import annotations

import streamlit as st

from src.config.settings import DocMindSettings


def build_reranker_controls(settings: DocMindSettings) -> None:
    """Render a read-only retrieval info panel (no UI toggles)."""
    sb = st.sidebar
    sb.markdown("### Retrieval & Reranking (locked)")
    sb.write(
        f"Fusion: {settings.retrieval.fusion_mode.upper()} · "
        f"Prefetch TopK: {settings.retrieval.fused_top_k} · "
        f"RRF K: {settings.retrieval.rrf_k}"
    )
    sb.write(
        f"Reranking: on · TopK: {settings.retrieval.reranking_top_k} · "
        f"Normalize: {settings.retrieval.reranker_normalize_scores}"
    )
    st.info("Hybrid & Rerank tuning guide: docs/perf/hybrid_tuning.md")


__all__ = ["build_reranker_controls"]
