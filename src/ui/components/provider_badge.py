"""Provider badge Streamlit component.

Shows the active LLM provider and model in a compact badge. Use on Chat and
Settings pages per SPEC-001 and SPEC-008.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from src.config.settings import DocMindSettings

if TYPE_CHECKING:
    from src.retrieval.llama_index_adapter import GraphRAGHealth


def provider_badge(
    cfg: DocMindSettings,
    *,
    graphrag_health: GraphRAGHealth | None = None,
) -> None:
    """Render a small badge indicating active provider and model.

    Args:
        cfg: Current unified settings.
        graphrag_health: Optional typed GraphRAG health state to reuse.
    """
    from src.retrieval.llama_index_adapter import get_graphrag_health

    provider = cfg.llm_backend
    model = cfg.effective_model
    base_url: str | None = None
    if provider == "ollama":
        base_url = str(cfg.ollama_base_url).rstrip("/")
    elif provider == "lmstudio":
        base_url = str(cfg.lmstudio_base_url).rstrip("/")
    elif provider == "vllm":
        base_url = str(cfg.vllm_base_url).rstrip("/")
    elif provider == "llamacpp":
        base_url = (
            str(cfg.llamacpp_base_url).rstrip("/") if cfg.llamacpp_base_url else None
        )

    if graphrag_health is None:
        graphrag_health = get_graphrag_health()
    status_label = {
        "ready": "enabled",
        "installed": "installed (validation deferred)",
        "unavailable": "disabled",
    }[graphrag_health.status]

    st.badge(f"Provider: {provider}")
    if model:
        st.caption(f"Model: {model}")
    st.badge(
        f"GraphRAG: {status_label} ({graphrag_health.adapter_name})",
        icon={"ready": "✅", "installed": ":material/info:", "unavailable": "⚠️"}[
            graphrag_health.status
        ],
        help=graphrag_health.hint,
    )
    if base_url:
        st.caption(f"Base URL: {base_url}")


__all__ = ["provider_badge"]
