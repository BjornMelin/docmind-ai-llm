"""Provider badge Streamlit component.

Shows the active LLM provider and model in a compact badge. Use on Chat and
Settings pages per SPEC-001 and SPEC-008.
"""

from __future__ import annotations

import html

import streamlit as st

from src.config.settings import DocMindSettings
from src.retrieval.adapter_registry import get_default_adapter_health


def provider_badge(cfg: DocMindSettings) -> None:
    """Render a small badge indicating active provider and model.

    Args:
        cfg: Current unified settings.
    """
    provider = cfg.llm_backend
    vllm_cfg = getattr(cfg, "vllm", None)
    model = cfg.model or (getattr(vllm_cfg, "model", None) if vllm_cfg else None)
    base_url: str | None = None
    if provider == "ollama":
        base_url = cfg.ollama_base_url
    elif provider == "lmstudio":
        base_url = cfg.lmstudio_base_url
    elif provider == "vllm":
        base_url = cfg.vllm_base_url or (
            getattr(vllm_cfg, "vllm_base_url", None) if vllm_cfg else None
        )
    elif provider == "llamacpp":
        if cfg.llamacpp_base_url:
            base_url = cfg.llamacpp_base_url
        else:
            base_url = (
                str(getattr(vllm_cfg, "llamacpp_model_path", "")) if vllm_cfg else None
            )

    supports_graphrag, adapter_name, hint = get_default_adapter_health()
    status_label = "enabled" if supports_graphrag else "disabled"
    tooltip = html.escape(hint, quote=True)
    graphrag_html = (
        f" · GraphRAG: <span title='{tooltip}'><b>{status_label}</b>"
        f" ({adapter_name})</span>"
    )

    badge = (
        "<span style='padding:4px 8px;border-radius:12px;"
        "background:#eef;border:1px solid #aac'>"
        f"Provider: <b>{provider}</b> · Model: <code>{model}</code>"
        + (f" · <small>{base_url}</small>" if base_url else "")
        + graphrag_html
        + "</span>"
    )
    st.markdown(badge, unsafe_allow_html=True)


__all__ = ["provider_badge"]
