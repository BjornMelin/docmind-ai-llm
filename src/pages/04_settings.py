# pylint: disable=invalid-name, C0103, too-many-statements
"""Settings page for LLM runtime (SPEC-001).

Provides provider selection, URLs, model, context window, timeout, and GPU
toggle. Supports applying runtime immediately and saving to .env.
"""

from __future__ import annotations

import os
from contextlib import suppress
from pathlib import Path

import streamlit as st

from src.config.integrations import initialize_integrations
from src.config.settings import settings
from src.retrieval.adapter_registry import get_default_adapter_health
from src.ui.components.provider_badge import provider_badge


def _is_localhost(url: str) -> bool:
    return (
        url.startswith("http://localhost")
        or url.startswith("https://localhost")
        or url.startswith("http://127.0.0.1")
        or url.startswith("https://127.0.0.1")
    )


def _persist_env(vars_to_set: dict[str, str]) -> None:
    """Persist key=value pairs into the project's .env file.

    Minimal .env updater: overwrites existing keys; preserves others.
    """
    env_path = Path(".env")
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    existing |= vars_to_set
    content = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    env_path.write_text(content, encoding="utf-8")


def _apply_runtime() -> None:
    """Apply current runtime by rebinding Settings.llm and context caps."""
    initialize_integrations(force_llm=True, force_embed=False)
    st.success("Runtime applied. Settings.llm rebound.")


def main() -> None:
    """Render the LLM Runtime settings page and actions."""
    st.title("Settings · LLM Runtime")

    # Show badge
    provider_badge(settings)

    st.subheader("Provider")
    provider = st.selectbox(
        "LLM Provider",
        options=["ollama", "vllm", "lmstudio", "llamacpp"],
        index=(["ollama", "vllm", "lmstudio", "llamacpp"].index(settings.llm_backend)),
        help="Select the active LLM backend",
    )

    st.subheader("Model & Context")
    model = st.text_input(
        "Model (id or GGUF path)",
        value=(settings.model or settings.vllm.model),
        help=("Model identifier (Ollama/vLLM/LM Studio) or GGUF path (LlamaCPP)"),
    )
    context_window = st.number_input(
        "Context window",
        min_value=1024,
        max_value=200_000,
        value=int(settings.context_window or settings.vllm.context_window),
        step=1024,
    )
    timeout_s = st.number_input(
        "Request timeout (seconds)",
        min_value=5,
        max_value=600,
        value=int(settings.llm_request_timeout_seconds),
    )
    use_gpu = st.checkbox(
        "Enable GPU acceleration",
        value=bool(settings.enable_gpu_acceleration),
    )

    st.subheader("Provider URLs")
    col1, col2 = st.columns(2)
    with col1:
        ollama_url = st.text_input(
            "Ollama base URL",
            value=settings.ollama_base_url,
        )
        vllm_url = st.text_input(
            "vLLM base URL",
            value=(settings.vllm_base_url or settings.vllm.vllm_base_url),
            help="OpenAI-compatible server or native. /v1 optional",
        )
    with col2:
        lmstudio_url = st.text_input(
            "LM Studio base URL",
            value=settings.lmstudio_base_url,
            help="Must end with /v1",
        )
        llamacpp_url = st.text_input(
            "llama.cpp server URL (optional)",
            value=(settings.llamacpp_base_url or ""),
            placeholder="http://localhost:8080/v1",
        )

    # Show resolved normalized backend base URL (read-only)
    st.caption("Resolved backend base URL (normalized)")
    st.text_input(
        "Resolved base URL",
        value=str(getattr(settings, "backend_base_url_normalized", "")),
        disabled=True,
    )

    gguf_path = st.text_input(
        "GGUF model path (LlamaCPP local)",
        value=str(settings.vllm.llamacpp_model_path),
    )
    gguf_valid = False
    if gguf_path:
        if os.path.exists(gguf_path) and gguf_path.lower().endswith(".gguf"):
            gguf_valid = True
        else:
            st.error(
                "Invalid GGUF model path. File must exist and have a .gguf extension."
            )

    st.subheader("Security")
    allow_remote = st.checkbox(
        "Allow remote endpoints",
        value=bool(settings.security.allow_remote_endpoints),
        help="When off, only localhost URLs are accepted",
    )
    st.caption("Effective policy (read-only)")
    st.text_input(
        "Remote endpoints allowed",
        value="true" if settings.security.allow_remote_endpoints else "false",
        disabled=True,
    )
    st.text_input(
        "Endpoint allowlist size",
        value=str(len(settings.security.endpoint_allowlist)),
        disabled=True,
    )

    # Retrieval settings (expose minimal toggles; no IO on import)
    st.subheader("Retrieval (Policy)")
    st.caption("Server-side hybrid and fusion are managed by environment policy")
    st.text_input(
        "Server-side hybrid enabled",
        value=str(bool(getattr(settings.retrieval, "enable_server_hybrid", False))),
        disabled=True,
    )
    st.text_input(
        "Fusion mode",
        value=str(getattr(settings.retrieval, "fusion_mode", "rrf")),
        disabled=True,
    )
    # Reranking remains displayed as read-only policy
    use_rerank = bool(getattr(settings.retrieval, "use_reranking", True))
    st.text_input("Reranking enabled", value=str(use_rerank), disabled=True)
    rrf_k = st.number_input(
        "RRF k-constant",
        min_value=1,
        max_value=256,
        value=int(getattr(settings.retrieval, "rrf_k", 60)),
    )
    col1t, col2t, col3t, col4t = st.columns(4)
    with col1t:
        t_text = st.number_input(
            "Text rerank timeout (ms)",
            min_value=50,
            max_value=5000,
            value=int(getattr(settings.retrieval, "text_rerank_timeout_ms", 250)),
        )
    with col2t:
        t_siglip = st.number_input(
            "SigLIP timeout (ms)",
            min_value=25,
            max_value=5000,
            value=int(getattr(settings.retrieval, "siglip_timeout_ms", 150)),
        )
    with col3t:
        t_colpali = st.number_input(
            "ColPali timeout (ms)",
            min_value=25,
            max_value=10000,
            value=int(getattr(settings.retrieval, "colpali_timeout_ms", 400)),
        )

    st.subheader("GraphRAG")
    supports, adapter_name, hint = get_default_adapter_health()
    st.text_input(
        "Adapter",
        value=adapter_name,
        disabled=True,
    )
    st.text_input(
        "GraphRAG status",
        value="enabled" if supports else "disabled",
        disabled=True,
    )
    if not supports:
        st.info(hint)
    with col4t:
        t_total = st.number_input(
            "Total rerank budget (ms)",
            min_value=100,
            max_value=20000,
            value=int(getattr(settings.retrieval, "total_rerank_budget_ms", 800)),
        )

    # Basic validation rules
    if lmstudio_url and not lmstudio_url.rstrip("/").endswith("/v1"):
        st.error("LM Studio URL must end with /v1")
    if not allow_remote:
        for name, url in (
            ("Ollama", ollama_url),
            ("vLLM", vllm_url),
            ("LM Studio", lmstudio_url),
            ("llama.cpp", llamacpp_url or ""),
        ):
            if url and not _is_localhost(url):
                st.error(
                    f"{name} URL must be localhost when remote endpoints are disabled"
                )

    # Actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Apply runtime", use_container_width=True):
            # Update in-memory settings first
            settings.llm_backend = provider  # type: ignore[assignment]
            settings.model = model  # type: ignore[assignment]
            settings.context_window = int(context_window)
            settings.llm_request_timeout_seconds = int(timeout_s)
            settings.enable_gpu_acceleration = bool(use_gpu)
            settings.ollama_base_url = ollama_url  # type: ignore[assignment]
            settings.vllm_base_url = vllm_url  # type: ignore[assignment]
            settings.lmstudio_base_url = lmstudio_url  # type: ignore[assignment]
            settings.llamacpp_base_url = llamacpp_url or None
            # nested path
            # Update GGUF path only when valid
            if gguf_valid and gguf_path:
                with suppress(Exception):  # pragma: no cover - UI guard
                    settings.vllm.llamacpp_model_path = Path(gguf_path)
            settings.security.allow_remote_endpoints = bool(allow_remote)
            # Apply retrieval timeouts to in-memory settings; policy is read-only
            with suppress(Exception):
                settings.retrieval.rrf_k = int(rrf_k)
                settings.retrieval.text_rerank_timeout_ms = int(t_text)
                settings.retrieval.siglip_timeout_ms = int(t_siglip)
                settings.retrieval.colpali_timeout_ms = int(t_colpali)
                settings.retrieval.total_rerank_budget_ms = int(t_total)

            _apply_runtime()

    with col_b:
        if st.button("Save", use_container_width=True):
            env_map = {
                "DOCMIND_LLM_BACKEND": provider,
                "DOCMIND_MODEL": model,
                "DOCMIND_CONTEXT_WINDOW": str(int(context_window)),
                "DOCMIND_LLM_REQUEST_TIMEOUT_SECONDS": str(int(timeout_s)),
                "DOCMIND_ENABLE_GPU_ACCELERATION": ("true" if use_gpu else "false"),
                "DOCMIND_OLLAMA_BASE_URL": ollama_url,
                "DOCMIND_VLLM_BASE_URL": vllm_url,
                "DOCMIND_LMSTUDIO_BASE_URL": lmstudio_url,
                "DOCMIND_LLAMACPP_BASE_URL": llamacpp_url or "",
                # nested path override
                "DOCMIND_VLLM__LLAMACPP_MODEL_PATH": gguf_path,
                "DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS": (
                    "true" if allow_remote else "false"
                ),
                # Retrieval policy is configured via env; read-only here
                "DOCMIND_RETRIEVAL__RRF_K": str(int(rrf_k)),
                "DOCMIND_RETRIEVAL__TEXT_RERANK_TIMEOUT_MS": str(int(t_text)),
                "DOCMIND_RETRIEVAL__SIGLIP_TIMEOUT_MS": str(int(t_siglip)),
                "DOCMIND_RETRIEVAL__COLPALI_TIMEOUT_MS": str(int(t_colpali)),
                "DOCMIND_RETRIEVAL__TOTAL_RERANK_BUDGET_MS": str(int(t_total)),
            }
            _persist_env(env_map)
            st.success("Saved to .env")

    st.subheader("Cache Utilities")
    st.caption(
        "Bump the global cache version and clear Streamlit caches. "
        "Use this if results seem stale after changing settings or content."
    )
    if st.button("Clear caches", use_container_width=True):
        try:
            from src.ui.cache import clear_caches

            new_v = clear_caches(settings)
            st.success(f"Caches cleared. Cache version bumped to {new_v}.")
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as e:  # pragma: no cover - defensive UI feedback
            st.error(f"Failed to clear caches: {e}")


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
