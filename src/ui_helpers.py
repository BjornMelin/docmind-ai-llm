"""UI helper utilities."""

from __future__ import annotations

import streamlit as st

from src.config.settings import DocMindSettings


def build_reranker_controls(settings: DocMindSettings) -> None:
    """Render the reranker sidebar and sync user inputs back to settings."""
    sb = st.sidebar
    sb.markdown("### Retrieval & Reranking")
    mode = sb.radio(
        "Reranker Mode",
        options=["auto", "text", "multimodal"],
        index=["auto", "text", "multimodal"].index(settings.retrieval.reranker_mode),
        key="reranker_mode",
    )
    normalize = sb.checkbox(
        "Normalize scores",
        value=bool(settings.retrieval.reranker_normalize_scores),
        key="reranker_normalize_scores",
    )
    top_n = sb.number_input(
        "Top N",
        min_value=1,
        max_value=20,
        value=int(settings.retrieval.reranking_top_k),
        key="reranking_top_k",
    )
    settings.retrieval.reranker_mode = mode
    settings.retrieval.reranker_normalize_scores = normalize
    settings.retrieval.reranking_top_k = int(top_n)


__all__ = ["build_reranker_controls"]
