"""Settings page for LLM runtime (SPEC-001).

Provides provider selection, URLs, model, context window, timeout, and GPU
toggle. Supports applying runtime immediately and saving to .env.
"""

from __future__ import annotations

from contextlib import suppress
from pathlib import Path

import streamlit as st

from src.config.integrations import initialize_integrations
from src.config.settings import DocMindSettings, settings
from src.ui.components.provider_badge import provider_badge


def _persist_env(vars_to_set: dict[str, str]) -> None:
    """Persist key=value pairs into the project's .env file.

    Minimal .env updater: overwrites existing keys; preserves others.
    """
    env_path = Path(".env")
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if not line.strip() or line.strip().startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    existing.update(vars_to_set)
    content = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    env_path.write_text(content)


def _apply_runtime(cfg: DocMindSettings) -> None:
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
        index=["ollama", "vllm", "lmstudio", "llamacpp"].index(settings.llm_backend),
        help="Select the active LLM backend",
    )

    st.subheader("Model & Context")
    model = st.text_input(
        "Model (id or GGUF path)",
        value=(settings.model or settings.vllm.model),
        help="Model identifier (Ollama/vLLM/LM Studio) or GGUF path (LlamaCPP)",
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

    gguf_path = st.text_input(
        "GGUF model path (LlamaCPP local)",
        value=str(settings.vllm.llamacpp_model_path),
    )

    st.subheader("Security")
    allow_remote = st.checkbox(
        "Allow remote endpoints",
        value=bool(settings.allow_remote_endpoints),
        help="When off, only localhost URLs are accepted",
    )

    # Actions
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Apply runtime", use_container_width=True):
            # Update in-memory settings first
            settings.llm_backend = provider  # type: ignore[assignment]
            settings.model = model  # type: ignore[assignment]
            settings.context_window = int(context_window)  # type: ignore[assignment]
            settings.llm_request_timeout_seconds = int(timeout_s)  # type: ignore[assignment]
            settings.enable_gpu_acceleration = bool(use_gpu)  # type: ignore[assignment]
            settings.ollama_base_url = ollama_url  # type: ignore[assignment]
            settings.vllm_base_url = vllm_url  # type: ignore[assignment]
            settings.lmstudio_base_url = lmstudio_url  # type: ignore[assignment]
            settings.llamacpp_base_url = llamacpp_url or None  # type: ignore[assignment]
            # nested path
            with suppress(Exception):  # pragma: no cover - UI guard
                settings.vllm.llamacpp_model_path = Path(gguf_path)
            settings.allow_remote_endpoints = bool(allow_remote)  # type: ignore[assignment]

            _apply_runtime(settings)

    with col_b:
        if st.button("Save", use_container_width=True):
            env_map = {
                "DOCMIND_LLM_BACKEND": provider,
                "DOCMIND_MODEL": model,
                "DOCMIND_CONTEXT_WINDOW": str(int(context_window)),
                "DOCMIND_LLM_REQUEST_TIMEOUT_SECONDS": str(int(timeout_s)),
                "DOCMIND_ENABLE_GPU_ACCELERATION": "true" if use_gpu else "false",
                "DOCMIND_OLLAMA_BASE_URL": ollama_url,
                "DOCMIND_VLLM_BASE_URL": vllm_url,
                "DOCMIND_LMSTUDIO_BASE_URL": lmstudio_url,
                "DOCMIND_LLAMACPP_BASE_URL": llamacpp_url or "",
                # nested path override
                "DOCMIND_VLLM__LLAMACPP_MODEL_PATH": gguf_path,
                "DOCMIND_ALLOW_REMOTE_ENDPOINTS": "true" if allow_remote else "false",
            }
            _persist_env(env_map)
            st.success("Saved to .env")


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
