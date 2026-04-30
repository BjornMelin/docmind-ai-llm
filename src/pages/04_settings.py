"""Settings page for LLM runtime (SPEC-001, SPEC-022).

Provides provider selection, URLs, model, context window, timeout, and GPU
toggle. Supports applying runtime immediately and saving to .env with
pre-validation to avoid persisting invalid configuration.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from typing import Any, TypedDict

import streamlit as st
from pydantic import ValidationError

from src.config.env_persistence import persist_env
from src.config.llm_runtime_probe import probe_openai_compatible_runtime
from src.config.settings import DocMindSettings, settings
from src.config.settings_utils import DEFAULT_OPENAI_BASE_URL, ensure_v1
from src.retrieval import adapter_registry
from src.ui.components.provider_badge import provider_badge
from src.utils.telemetry import log_jsonl

_LLAMACPP_DISALLOWED_SHARED_URLS = frozenset(
    filter(
        None,
        {
            str(DEFAULT_OPENAI_BASE_URL).rstrip("/"),
            ensure_v1(DEFAULT_OPENAI_BASE_URL),
            "https://api.openai.com",
            "https://api.openai.com/v1",
        },
    )
)
_LLAMACPP_DISALLOWED_EXPLICIT_URLS = frozenset(
    {"https://api.openai.com", "https://api.openai.com/v1"}
)

_OPENAI_BASE_URL_KEY = "docmind_openai_base_url"
_OPENAI_API_KEY_KEY = "docmind_openai_api_key"
_OPENAI_REQUIRE_V1_KEY = "docmind_openai_require_v1"
_OPENAI_API_MODE_KEY = "docmind_openai_api_mode"
_OPENAI_HEADERS_KEY = "docmind_openai_headers_json"


def _ensure_openai_compatible_form_state() -> None:
    """Seed persistent OpenAI-compatible form values."""
    if _OPENAI_BASE_URL_KEY not in st.session_state:
        st.session_state[_OPENAI_BASE_URL_KEY] = str(settings.openai.base_url)
    if _OPENAI_API_KEY_KEY not in st.session_state:
        st.session_state[_OPENAI_API_KEY_KEY] = (
            settings.openai.api_key.get_secret_value()
            if settings.openai.api_key is not None
            else ""
        )
    if _OPENAI_REQUIRE_V1_KEY not in st.session_state:
        st.session_state[_OPENAI_REQUIRE_V1_KEY] = bool(
            getattr(settings.openai, "require_v1", True)
        )
    if _OPENAI_API_MODE_KEY not in st.session_state:
        st.session_state[_OPENAI_API_MODE_KEY] = str(
            getattr(settings.openai, "api_mode", "chat_completions")
        )
    if _OPENAI_HEADERS_KEY not in st.session_state:
        st.session_state[_OPENAI_HEADERS_KEY] = _safe_json_dumps(
            getattr(settings.openai, "default_headers", None) or {}
        )


def _validate_candidate(
    candidate: dict[str, object],
) -> tuple[DocMindSettings | None, list[str]]:
    """Validate a candidate settings payload before Apply/Save.

    Args:
        candidate: A dictionary whose keys correspond to DocMindSettings fields.

    Returns:
        tuple[DocMindSettings | None, list[str]]: A tuple of (settings, errors).
            The settings object is None if validation fails. The errors list
            contains human-readable validation messages.
    """
    try:
        validated = DocMindSettings.model_validate(candidate)
    except ValidationError as exc:
        messages: list[str] = []
        for err in exc.errors():
            loc_parts = err.get("loc", ())
            if not isinstance(loc_parts, (list, tuple)):
                loc_parts = (loc_parts,)
            loc = ".".join(str(p) for p in loc_parts if p is not None)
            msg = str(err.get("msg", "Invalid value"))
            messages.append(f"{loc}: {msg}" if loc else msg)
        return None, messages
    except (TypeError, ValueError) as exc:
        return None, [str(exc)]
    return validated, []


def _apply_validated_runtime(validated: DocMindSettings) -> None:
    """Apply runtime and rebind LlamaIndex settings.

    Args:
        validated: The validated settings object to apply.
    """
    updated = settings.model_copy(
        update={
            "llm_backend": validated.llm_backend,
            "model": validated.model,
            "context_window": validated.context_window,
            "llm_request_timeout_seconds": (validated.llm_request_timeout_seconds),
            "enable_gpu_acceleration": validated.enable_gpu_acceleration,
            "openai": validated.openai,
            "ollama_base_url": validated.ollama_base_url,
            "ollama_api_key": validated.ollama_api_key,
            "ollama_enable_web_search": validated.ollama_enable_web_search,
            "ollama_embed_dimensions": validated.ollama_embed_dimensions,
            "ollama_enable_logprobs": validated.ollama_enable_logprobs,
            "ollama_top_logprobs": validated.ollama_top_logprobs,
            "vllm_base_url": validated.vllm_base_url,
            "lmstudio_base_url": validated.lmstudio_base_url,
            "llamacpp_base_url": validated.llamacpp_base_url,
            "security": settings.security.model_copy(
                update={
                    "allow_remote_endpoints": (
                        validated.security.allow_remote_endpoints
                    ),
                }
            ),
            "retrieval": settings.retrieval.model_copy(
                update={
                    "rrf_k": validated.retrieval.rrf_k,
                    "text_rerank_timeout_ms": (
                        validated.retrieval.text_rerank_timeout_ms
                    ),
                    "siglip_timeout_ms": (validated.retrieval.siglip_timeout_ms),
                    "colpali_timeout_ms": (validated.retrieval.colpali_timeout_ms),
                    "total_rerank_budget_ms": (
                        validated.retrieval.total_rerank_budget_ms
                    ),
                }
            ),
        }
    )
    # Apply updated settings in-place so existing imports keep the same
    # instance.
    # NOTE: For our singleton Pydantic settings object, re-calling
    # __init__ with new data is the canonical pattern for in-place
    # reload after removing apply_settings_in_place(), and is
    # intentional here.
    settings.__init__(**updated.model_dump(mode="python"))

    model_label = validated.model or validated.vllm.model
    try:
        from src.config.integrations import initialize_integrations

        initialize_integrations(force_llm=True, force_embed=False)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover - defensive UI feedback
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="settings.apply")
        st.error(f"Runtime apply failed: {exc.__class__.__name__}")
        log_jsonl(
            {
                "settings.apply": True,
                "success": False,
                "backend": validated.llm_backend,
                "model": model_label,
                "reason": exc.__class__.__name__,
                "error_type": exc.__class__.__name__,
                "error": redaction.redacted,
            }
        )
        return

    from llama_index.core import (
        Settings as LISettings,
    )  # local import (tests patch)

    if getattr(LISettings, "llm", None) is None:
        st.error("Runtime apply failed: Settings.llm is not bound.")
        log_jsonl(
            {
                "settings.apply": True,
                "success": False,
                "backend": validated.llm_backend,
                "model": model_label,
                "reason": "llm_unbound",
            }
        )
        return

    st.success("Runtime applied. Settings.llm bound.")
    log_jsonl(
        {
            "settings.apply": True,
            "success": True,
            "backend": validated.llm_backend,
            "model": model_label,
        }
    )


def main() -> None:
    """Render the LLM Runtime settings page and actions."""
    st.title("Settings · LLM Runtime")

    # Show badge
    graphrag_health = adapter_registry.get_default_adapter_health()
    provider_badge(settings, graphrag_health=graphrag_health)
    provider = _render_provider_section()
    model, context_window, timeout_s, use_gpu = _render_model_section()
    if provider == "openai_compatible":
        (
            openai_base_url,
            openai_api_key,
            openai_require_v1,
            openai_api_mode,
            openai_headers_json,
            openai_ui_errors,
        ) = _render_openai_compatible_section()
    else:
        openai_base_url = str(settings.openai.base_url)
        openai_api_key = (
            settings.openai.api_key.get_secret_value()
            if settings.openai.api_key is not None
            else ""
        )
        openai_require_v1 = bool(getattr(settings.openai, "require_v1", True))
        openai_api_mode = str(getattr(settings.openai, "api_mode", "chat_completions"))
        openai_headers_json = _safe_json_dumps(
            getattr(settings.openai, "default_headers", None) or {}
        )
        openai_ui_errors = []

    ollama_url, vllm_url, lmstudio_url, llamacpp_url = _render_provider_urls(provider)
    _render_llamacpp_server_guide(provider)
    (
        ollama_api_key,
        ollama_enable_web_search,
        ollama_embed_dimensions,
        ollama_enable_logprobs,
        ollama_top_logprobs,
    ) = _render_ollama_advanced_section(provider)
    allow_remote = _render_security_section()
    rrf_k, t_text, t_siglip, t_colpali, t_total = _render_retrieval_section()
    _render_graphrag_section(graphrag_health)
    if provider == "ollama":
        _render_ollama_web_search_warning(
            enabled=ollama_enable_web_search,
            allow_remote=allow_remote,
            allowlist=settings.security.endpoint_allowlist,
        )

    ui_errors = _validate_llamacpp_inputs(provider, llamacpp_url, openai_base_url)
    ui_errors.extend(openai_ui_errors)
    values: SettingsFormValues = {
        "provider": provider,
        "model": model,
        "context_window": context_window,
        "openai_base_url": openai_base_url,
        "openai_api_key": openai_api_key,
        "openai_require_v1": openai_require_v1,
        "openai_api_mode": openai_api_mode,
        "openai_headers_json": openai_headers_json,
        "ollama_url": ollama_url,
        "ollama_api_key": ollama_api_key,
        "ollama_enable_web_search": ollama_enable_web_search,
        "ollama_embed_dimensions": ollama_embed_dimensions,
        "ollama_enable_logprobs": ollama_enable_logprobs,
        "ollama_top_logprobs": ollama_top_logprobs,
        "vllm_url": vllm_url,
        "lmstudio_url": lmstudio_url,
        "llamacpp_url": llamacpp_url,
        "timeout_s": timeout_s,
        "use_gpu": use_gpu,
        "allow_remote": allow_remote,
        "rrf_k": rrf_k,
        "t_text": t_text,
        "t_siglip": t_siglip,
        "t_colpali": t_colpali,
        "t_total": t_total,
    }
    candidate = _build_candidate_settings(values)
    validated, validation_errors = _validate_candidate(candidate)
    _render_resolved_base_url(validated)
    _render_validation(ui_errors, validation_errors)
    _render_endpoint_test(validated)
    _render_actions(validated, ui_errors)
    _render_cache_controls()


def _render_provider_section() -> str:
    """Render provider selector and return selection.

    Returns:
        str: Selected provider name.
    """
    st.subheader("Provider")
    options = ["ollama", "openai_compatible", "vllm", "lmstudio", "llamacpp"]
    try:
        default_index = options.index(settings.llm_backend)
    except ValueError:
        default_index = 0  # Fall back to first option
    return st.selectbox(
        "LLM Provider",
        options=options,
        index=default_index,
        help="Select the active LLM backend",
    )


def _render_model_section() -> tuple[str, int, int, bool]:
    """Render model/context inputs and return values.

    Returns:
        tuple[str, int, int, bool]: Selected (model_name, context_window,
            timeout_sec, use_gpu).
    """
    st.subheader("Model & Context")
    model = st.text_input(
        "Model ID",
        value=(settings.model or settings.vllm.model),
        help="Model identifier exposed by the selected provider.",
    )
    context_window = int(
        st.number_input(
            "Context window",
            min_value=1024,
            max_value=200_000,
            value=int(settings.context_window or settings.vllm.context_window),
            step=1024,
        )
    )
    timeout_s = int(
        st.number_input(
            "Request timeout (seconds)",
            min_value=5,
            max_value=600,
            value=int(settings.llm_request_timeout_seconds),
        )
    )
    use_gpu = st.checkbox(
        "Enable GPU acceleration",
        value=bool(settings.enable_gpu_acceleration),
    )
    return model, context_window, timeout_s, use_gpu


_OPENAI_COMPAT_PRESETS: dict[str, dict[str, object]] = {
    "Custom": {},
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "require_v1": True,
        "headers": {},
        "api_mode": "responses",
    },
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "require_v1": True,
        "headers": {"HTTP-Referer": "", "X-Title": ""},
        "api_mode": "chat_completions",
    },
    "xAI": {
        "base_url": "https://api.x.ai/v1",
        "require_v1": True,
        "headers": {},
        "api_mode": "responses",
    },
    "Vercel AI Gateway": {
        "base_url": "https://ai-gateway.vercel.sh/v1",
        "require_v1": True,
        "headers": {},
        "api_mode": "responses",
    },
    "LiteLLM Proxy": {
        "base_url": "http://localhost:4000",
        "require_v1": False,
        "headers": {},
        "api_mode": "responses",
    },
}


def _safe_json_dumps(value: object) -> str:
    """Serialize value to pretty-printed JSON with fallback.

    Args:
        value: Object to serialize.

    Returns:
        str: JSON string or "{}" on serialization failure.
    """
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


def _safe_json_dumps_compact(value: object) -> str:
    """Serialize value to compact JSON with fallback.

    Args:
        value: Object to serialize.

    Returns:
        str: JSON string or "{}" on serialization failure.
    """
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return "{}"


_HEADER_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")


def _parse_headers_json(raw: str) -> tuple[dict[str, str] | None, list[str]]:
    """Parse a JSON object as headers mapping.

    Args:
        raw: Raw JSON string from UI input.

    Returns:
        tuple[dict[str, str] | None, list[str]]: A tuple of (headers, errors).
            The headers dictionary is None if the input is empty or invalid.
            The errors list contains human-readable validation messages.
    """
    text = (raw or "").strip()
    if not text:
        return None, []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"openai.default_headers: invalid JSON ({exc.msg})"]
    if not isinstance(parsed, dict):
        return None, ["openai.default_headers: must be a JSON object"]

    headers: dict[str, str] = {}
    errors: list[str] = []
    for raw_k, raw_v in parsed.items():
        k = str(raw_k).strip()
        if not k:
            continue
        v = str(raw_v).strip()
        if not v:
            continue
        if _HEADER_CONTROL_CHARS_RE.search(k) or _HEADER_CONTROL_CHARS_RE.search(v):
            errors.append(
                "openai.default_headers: control characters are not "
                "allowed in keys/values"
            )
            continue
        if "\n" in k or "\r" in k or "\n" in v or "\r" in v:
            errors.append(
                "openai.default_headers: newlines are not allowed in keys/values"
            )
            continue
        headers[k] = v
    return (headers or None), errors


def _render_openai_compatible_section() -> tuple[str, str, bool, str, str, list[str]]:
    """Render OpenAI-compatible provider settings.

    Returns:
        tuple[str, str, bool, str, str, list[str]]: Selected (base_url, api_key,
            require_v1, api_mode, headers_json, ui_errors).
    """
    st.subheader("OpenAI-Compatible Provider")
    st.caption(
        "Use this for OpenAI, OpenRouter, xAI, Vercel AI Gateway, "
        "LiteLLM Proxy, or any OpenAI-compatible server. Remote "
        "endpoints are opt-in."
    )

    preset_names = list(_OPENAI_COMPAT_PRESETS.keys())
    preset = st.selectbox(
        "Provider preset",
        options=preset_names,
        index=0,
    )
    preset_cfg = _OPENAI_COMPAT_PRESETS.get(preset) or {}
    _ensure_openai_compatible_form_state()

    if st.button("Use preset values", use_container_width=True):
        if "base_url" in preset_cfg:
            st.session_state[_OPENAI_BASE_URL_KEY] = str(preset_cfg["base_url"])
        if "require_v1" in preset_cfg:
            st.session_state[_OPENAI_REQUIRE_V1_KEY] = bool(preset_cfg["require_v1"])
        if "api_mode" in preset_cfg:
            st.session_state[_OPENAI_API_MODE_KEY] = str(preset_cfg["api_mode"])
        if "headers" in preset_cfg:
            st.session_state[_OPENAI_HEADERS_KEY] = _safe_json_dumps(
                preset_cfg["headers"]
            )
        st.rerun()

    base_url = st.text_input(
        "Base URL",
        help="Full base URL (with or without /v1 depending on provider).",
        key=_OPENAI_BASE_URL_KEY,
    )
    api_key = st.text_input(
        "API key (optional)",
        type="password",
        help=(
            "Bearer token for the provider. For local servers, a placeholder is fine."
        ),
        key=_OPENAI_API_KEY_KEY,
    )
    require_v1 = st.checkbox(
        "Normalize base URL to include /v1",
        help=(
            "Disable for providers rooted at '/', such as the LiteLLM Proxy default."
        ),
        key=_OPENAI_REQUIRE_V1_KEY,
    )
    api_mode = st.selectbox(
        "API mode",
        options=["chat_completions", "responses"],
        help=(
            "Use /responses only when the provider supports it "
            "(e.g., OpenAI, Vercel AI Gateway, xAI, vLLM, "
            "LiteLLM Proxy, Ollama; OpenRouter support is beta)."
        ),
        key=_OPENAI_API_MODE_KEY,
    )
    headers_json = st.text_area(
        "Default headers (JSON object)",
        height=140,
        help=(
            "Optional. Example: "
            '{"HTTP-Referer": "https://example.com", "X-Title": "DocMind"}'
        ),
        key=_OPENAI_HEADERS_KEY,
    )
    _, header_errors = _parse_headers_json(headers_json)
    return (
        base_url,
        api_key,
        require_v1,
        str(api_mode),
        headers_json,
        header_errors,
    )


def _render_provider_urls(provider: str) -> tuple[str, str, str, str]:
    """Render provider URL inputs.

    Args:
        provider: The currently selected LLM provider.

    Returns:
        tuple[str, str, str, str]: Provider URLs.
    """
    if provider == "openai_compatible":
        return (
            str(settings.ollama_base_url).rstrip("/"),
            str(settings.vllm_base_url or settings.vllm.vllm_base_url),
            str(settings.lmstudio_base_url),
            str(settings.llamacpp_base_url) if settings.llamacpp_base_url else "",
        )
    st.subheader("Provider URLs")
    col1, col2 = st.columns(2)
    with col1:
        ollama_url = st.text_input(
            "Ollama base URL",
            value=str(settings.ollama_base_url).rstrip("/"),
        )
        vllm_url = st.text_input(
            "vLLM base URL",
            value=str(settings.vllm_base_url or settings.vllm.vllm_base_url),
            help="OpenAI-compatible server or native. /v1 optional",
        )
    with col2:
        lmstudio_url = st.text_input(
            "LM Studio base URL",
            value=str(settings.lmstudio_base_url),
            help="OpenAI-compatible; normalized to end with /v1",
        )
        llamacpp_url = st.text_input(
            "llama.cpp server URL",
            value=str(settings.llamacpp_base_url) if settings.llamacpp_base_url else "",
            placeholder="http://localhost:8080/v1",
        )
    return ollama_url, vllm_url, lmstudio_url, llamacpp_url


def _render_ollama_advanced_section(
    provider: str,
) -> tuple[str, bool, int | None, bool, int]:
    """Render Ollama advanced settings and return values.

    Args:
        provider: The currently selected LLM provider.

    Returns:
        Tuple of (api_key, enable_web_search, embed_dimensions, enable_logprobs,
        top_logprobs).
    """
    if provider != "ollama":
        api_key_value = (
            settings.ollama_api_key.get_secret_value()
            if settings.ollama_api_key is not None
            else ""
        )
        embed_dimensions = (
            int(settings.ollama_embed_dimensions)
            if settings.ollama_embed_dimensions is not None
            else None
        )
        return (
            api_key_value,
            bool(settings.ollama_enable_web_search),
            embed_dimensions,
            bool(settings.ollama_enable_logprobs),
            int(settings.ollama_top_logprobs),
        )

    st.subheader("Ollama (Advanced)")
    st.caption("Optional Ollama-native features (cloud web tools are opt-in).")
    api_key_value = (
        settings.ollama_api_key.get_secret_value()
        if settings.ollama_api_key is not None
        else ""
    )
    api_key = st.text_input(
        "Ollama API key (optional)",
        type="password",
        value=api_key_value,
        help="Required for Ollama Cloud web_search/web_fetch tools.",
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        enable_web_search = st.checkbox(
            "Enable Ollama web search tools",
            value=bool(settings.ollama_enable_web_search),
        )
    with col2:
        embed_dim_raw = int(
            st.number_input(
                "Embed dimensions (0 = default)",
                min_value=0,
                max_value=8192,
                value=int(settings.ollama_embed_dimensions or 0),
            )
        )
    with col3:
        enable_logprobs = st.checkbox(
            "Enable Ollama logprobs",
            value=bool(settings.ollama_enable_logprobs),
        )
    top_logprobs = int(
        st.number_input(
            "Top logprobs (0-20)",
            min_value=0,
            max_value=20,
            value=int(settings.ollama_top_logprobs),
            disabled=not enable_logprobs,
        )
    )
    embed_dimensions = int(embed_dim_raw) if embed_dim_raw > 0 else None
    return (
        api_key,
        enable_web_search,
        embed_dimensions,
        enable_logprobs,
        top_logprobs,
    )


def _render_security_section() -> bool:
    """Render security controls and return allow-remote setting.

    Returns:
        bool: True if remote connections are permitted.
    """
    st.subheader("Security")
    allow_remote = st.checkbox(
        "Allow remote endpoints",
        value=bool(settings.security.allow_remote_endpoints),
        help=(
            "When off, non-loopback hosts must be allowlisted and resolve to "
            "public IPs. When on, strict endpoint validation is disabled "
            "(use for internal endpoints)."
        ),
    )
    if allow_remote:
        st.warning(
            "Remote endpoints are allowed. Strict endpoint validation and "
            "DNS hardening are disabled in this mode. Prefer keeping this "
            "off and using `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST` when "
            "possible."
        )
    else:
        st.info(
            "Remote endpoints are restricted. Non-loopback hosts must be "
            "allowlisted and resolve to public IPs."
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
    return allow_remote


def _render_retrieval_section() -> tuple[int, int, int, int, int]:
    """Render retrieval policy inputs and return values.

    Returns:
        tuple[int, int, int, int, int]: Selected (rrf_k, text_rerank_timeout_ms,
            siglip_timeout_ms, colpali_timeout_ms, total_rerank_budget_ms).
    """
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
    use_rerank = bool(getattr(settings.retrieval, "use_reranking", True))
    st.text_input("Reranking enabled", value=str(use_rerank), disabled=True)
    rrf_k = int(
        st.number_input(
            "RRF k-constant",
            min_value=1,
            max_value=256,
            value=int(getattr(settings.retrieval, "rrf_k", 60)),
        )
    )
    col1t, col2t, col3t, col4t = st.columns(4)
    with col1t:
        t_text = int(
            st.number_input(
                "Text rerank timeout (ms)",
                min_value=50,
                max_value=5000,
                value=int(getattr(settings.retrieval, "text_rerank_timeout_ms", 250)),
            )
        )
    with col2t:
        t_siglip = int(
            st.number_input(
                "SigLIP timeout (ms)",
                min_value=25,
                max_value=5000,
                value=int(getattr(settings.retrieval, "siglip_timeout_ms", 150)),
            )
        )
    with col3t:
        t_colpali = int(
            st.number_input(
                "ColPali timeout (ms)",
                min_value=25,
                max_value=10000,
                value=int(getattr(settings.retrieval, "colpali_timeout_ms", 400)),
            )
        )
    with col4t:
        t_total = int(
            st.number_input(
                "Total rerank budget (ms)",
                min_value=100,
                max_value=20000,
                value=int(getattr(settings.retrieval, "total_rerank_budget_ms", 800)),
            )
        )
    return rrf_k, t_text, t_siglip, t_colpali, t_total


def _render_graphrag_section(
    graphrag_health: tuple[bool, str, str] | None = None,
) -> None:
    """Render GraphRAG status section.

    Args:
        graphrag_health: Optional tuple containing `(supports, adapter_name,
            hint)` for GraphRAG health. If None, it will be fetched.
    """
    st.subheader("GraphRAG")
    if graphrag_health is None:
        graphrag_health = adapter_registry.get_default_adapter_health()
    supports, adapter_name, hint = graphrag_health
    st.text_input("Adapter", value=adapter_name, disabled=True)
    st.text_input(
        "GraphRAG status",
        value="enabled" if supports else "disabled",
        disabled=True,
    )
    if not supports:
        st.info(hint)


def _validate_llamacpp_inputs(
    provider: str,
    llamacpp_url: str,
    openai_base_url: str = "",
) -> list[str]:
    """Validate llama.cpp server settings.

    Args:
        provider: Current provider name.
        llamacpp_url: The llama.cpp server URL.
        openai_base_url: Shared OpenAI-compatible server URL fallback.

    Returns:
        list[str]: Human-readable validation messages.
    """
    ui_errors: list[str] = []
    clean_llamacpp_url = (llamacpp_url or "").strip()
    clean_openai_base_url = (openai_base_url or "").strip()
    is_llamacpp = provider == "llamacpp"
    if not is_llamacpp:
        return ui_errors

    normalized_openai_base_url = (
        ensure_v1(clean_openai_base_url) if clean_openai_base_url else None
    )
    normalized_llamacpp_url = (
        ensure_v1(clean_llamacpp_url) if clean_llamacpp_url else None
    )
    if normalized_llamacpp_url in _LLAMACPP_DISALLOWED_EXPLICIT_URLS:
        ui_errors.append("Provide a llama.cpp OpenAI-compatible server URL.")
        return ui_errors

    has_llamacpp_url = bool(normalized_llamacpp_url)
    has_custom_shared_url = (
        bool(normalized_openai_base_url)
        and normalized_openai_base_url not in _LLAMACPP_DISALLOWED_SHARED_URLS
    )
    if has_llamacpp_url or has_custom_shared_url:
        return ui_errors

    ui_errors.append("Provide a llama.cpp OpenAI-compatible server URL.")
    return ui_errors


def _render_llamacpp_server_guide(provider: str) -> None:
    """Render llama.cpp server launch guidance for GGUF users."""
    if provider != "llamacpp":
        return

    st.subheader("llama.cpp server")
    st.caption(
        "DocMind uses llama.cpp through `llama-server`'s OpenAI-compatible "
        "HTTP API. Keep the server bound to loopback unless you explicitly "
        "need remote access."
    )
    with st.expander("Launch examples", expanded=True):
        st.code(
            "\n".join(
                (
                    "# CPU / portable baseline",
                    "llama-server -m ./models/model.gguf --alias local-gguf "
                    "--ctx-size 8192 --host 127.0.0.1 --port 8080",
                    "",
                    "# CUDA or other GPU backends",
                    "llama-server -m ./models/model.gguf --alias local-gguf "
                    "--ctx-size 8192 -ngl 999 -fa --host 127.0.0.1 --port 8080",
                    "",
                    "# Add auth when exposing beyond local loopback",
                    "llama-server -m ./models/model.gguf --alias local-gguf "
                    "--api-key $DOCMIND_OPENAI__API_KEY --host 0.0.0.0 --port 8080",
                )
            ),
            language="bash",
        )
        st.markdown(
            "- Set **Model ID** to the `--alias` value, for example `local-gguf`.\n"
            "- Set **llama.cpp server URL** to `http://localhost:8080/v1`.\n"
            "- Match **Context window** to `--ctx-size`.\n"
            "- Use trusted GGUF files only; DocMind does not load local GGUF "
            "paths in-process."
        )


class SettingsFormValues(TypedDict):
    """Schema for values collected from the settings form."""

    provider: str
    model: str
    context_window: int
    openai_base_url: str
    openai_api_key: str
    openai_require_v1: bool
    openai_api_mode: str
    openai_headers_json: str
    ollama_url: str
    ollama_api_key: str
    ollama_enable_web_search: bool
    ollama_embed_dimensions: int | None
    ollama_enable_logprobs: bool
    ollama_top_logprobs: int
    vllm_url: str
    lmstudio_url: str
    llamacpp_url: str
    timeout_s: int
    use_gpu: bool
    allow_remote: bool
    rrf_k: int
    t_text: int
    t_siglip: int
    t_colpali: int
    t_total: int


def _build_candidate_settings(values: SettingsFormValues) -> dict[str, Any]:
    """Build a settings payload for validation."""
    provider = str(values["provider"])
    headers, _ = _parse_headers_json(values["openai_headers_json"])
    return {
        "llm_backend": provider,
        "model": str(values["model"]).strip() or None,
        "context_window": int(values["context_window"]),
        "openai": {
            "base_url": str(values["openai_base_url"]).strip(),
            "api_key": str(values["openai_api_key"]).strip() or None,
            "require_v1": bool(values["openai_require_v1"]),
            "api_mode": str(values["openai_api_mode"]).strip() or "chat_completions",
            "default_headers": headers,
        },
        "ollama_base_url": str(values["ollama_url"]).strip(),
        "ollama_api_key": str(values["ollama_api_key"]).strip() or None,
        "ollama_enable_web_search": bool(values["ollama_enable_web_search"]),
        "ollama_embed_dimensions": values["ollama_embed_dimensions"],
        "ollama_enable_logprobs": bool(values["ollama_enable_logprobs"]),
        "ollama_top_logprobs": int(values["ollama_top_logprobs"]),
        "vllm_base_url": str(values["vllm_url"]).strip() or None,
        "lmstudio_base_url": str(values["lmstudio_url"]).strip(),
        "llamacpp_base_url": str(values["llamacpp_url"]).strip() or None,
        "llm_request_timeout_seconds": int(values["timeout_s"]),
        "enable_gpu_acceleration": bool(values["use_gpu"]),
        "security": {
            "allow_remote_endpoints": bool(values["allow_remote"]),
            "endpoint_allowlist": list(settings.security.endpoint_allowlist),
            "trust_remote_code": bool(settings.security.trust_remote_code),
        },
        "retrieval": {
            "rrf_k": int(values["rrf_k"]),
            "text_rerank_timeout_ms": int(values["t_text"]),
            "siglip_timeout_ms": int(values["t_siglip"]),
            "colpali_timeout_ms": int(values["t_colpali"]),
            "total_rerank_budget_ms": int(values["t_total"]),
        },
    }


def _render_resolved_base_url(validated: DocMindSettings | None) -> None:
    """Render the normalized backend base URL."""
    resolved_base_url = (
        str(getattr(validated, "backend_base_url_normalized", ""))
        if validated is not None
        else str(getattr(settings, "backend_base_url_normalized", ""))
    )
    st.caption("Resolved backend base URL (normalized)")
    st.text_input("Resolved base URL", value=resolved_base_url, disabled=True)


def _render_validation(ui_errors: list[str], validation_errors: list[str]) -> None:
    """Render validation errors in the UI."""
    if not ui_errors and not validation_errors:
        return
    st.subheader("Validation")
    for msg in ui_errors + validation_errors:
        st.error(msg)


def _build_endpoint_test_headers(validated: DocMindSettings) -> dict[str, str]:
    """Build headers for the manual endpoint connectivity test.

    Args:
        validated: The validated runtime settings.

    Returns:
        Headers to send with the connectivity probe.
    """
    headers: dict[str, str] = {"Accept": "application/json"}
    if validated.openai.api_key is not None:
        headers["Authorization"] = (
            f"Bearer {validated.openai.api_key.get_secret_value()}"
        )
    if validated.openai.default_headers:
        headers.update(validated.openai.default_headers)
    return headers


def _render_endpoint_test(validated: DocMindSettings | None) -> None:
    """Render a manual endpoint connectivity test."""
    if validated is None:
        return
    base_url = getattr(validated, "backend_base_url_normalized", None)
    if not base_url:
        return
    backend_type = validated.llm_backend
    if backend_type not in (
        "openai_compatible",
        "vllm",
        "lmstudio",
        "llamacpp",
    ):
        st.info(
            "Connectivity test is only available for OpenAI-compatible "
            "backends (OpenAI-compatible, vLLM, LM Studio, llama.cpp)."
        )
        return

    st.subheader("Connectivity Test")
    st.caption(
        "Sends lightweight readiness requests to the configured endpoint. "
        "For llama.cpp this checks `/health` before `/v1/models`."
    )
    cooldown_key = "docmind_endpoint_test_last_ts"
    cooldown_s = 3.0
    now = time.monotonic()
    last_ts = float(st.session_state.get(cooldown_key, 0.0) or 0.0)
    remaining = cooldown_s - (now - last_ts)
    cooldown_active = remaining > 0
    if cooldown_active:
        st.caption(f"Cooldown: {remaining:.1f}s")
    if not st.button(
        "Test endpoint",
        use_container_width=True,
        disabled=cooldown_active,
    ):
        return
    st.session_state[cooldown_key] = now

    try:
        headers = _build_endpoint_test_headers(validated)
        timeout_s = min(30.0, float(validated.llm_request_timeout_seconds))
        result = probe_openai_compatible_runtime(
            base_url=str(base_url),
            backend=backend_type,
            headers=headers,
            timeout_s=timeout_s,
        )
        if result.ok:
            st.success(result.message)
        else:
            st.error(result.message)
        if result.health_status_code is not None:
            st.caption(f"Health status: HTTP {result.health_status_code}")
        if result.models_status_code is not None:
            st.caption(f"Models status: HTTP {result.models_status_code}")
    except Exception as exc:  # pragma: no cover - UI feedback
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="settings.endpoint_test")
        st.error(f"Endpoint test failed ({type(exc).__name__}).")
        st.caption(f"Error reference: {redaction.redacted}")
        log_jsonl(
            {
                "settings.endpoint_test": True,
                "success": False,
                "error_type": type(exc).__name__,
                "error": redaction.redacted,
            }
        )


def _render_ollama_web_search_warning(
    *, enabled: bool, allow_remote: bool, allowlist: Sequence[str]
) -> None:
    """Warn when Ollama Cloud web tools are enabled but not fully allowed.

    Args:
        enabled: Whether web tools are enabled.
        allow_remote: Whether remote endpoints are allowed.
        allowlist: Current endpoint allowlist.
    """
    from src.config.settings_utils import (
        normalize_endpoint_host,
        parse_any_http_url,
    )

    if not enabled:
        return
    if not allow_remote:
        st.warning(
            "Ollama web tools require remote endpoints. Enable "
            "`DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS`."
        )
        return

    def _is_ollama_host(entry: str) -> bool:
        """Check if entry references ollama.com or a subdomain."""
        raw = entry.strip()
        if not raw:
            return False
        candidate = raw if "://" in raw else f"https://{raw}"
        try:
            parsed = parse_any_http_url(candidate)
            host = normalize_endpoint_host(str(parsed.host))
        except Exception:
            host = normalize_endpoint_host(raw.split("/")[0].split(":")[0])
        return host == "ollama.com" or host.endswith(".ollama.com")

    has_ollama_host = any(_is_ollama_host(str(entry)) for entry in allowlist if entry)
    if not has_ollama_host:
        st.warning(
            "Ollama web tools require `https://ollama.com` in "
            "`DOCMIND_SECURITY__ENDPOINT_ALLOWLIST`."
        )


def _render_actions(validated: DocMindSettings | None, ui_errors: list[str]) -> None:
    """Render apply/save actions based on validation status."""
    actions_disabled = validated is None or bool(ui_errors)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button(
            "Apply runtime",
            use_container_width=True,
            disabled=actions_disabled,
        ):
            if validated is None:  # pragma: no cover - defensive
                st.error("Cannot apply: invalid settings.")
            else:
                _apply_validated_runtime(validated)

    with col_b:
        if st.button("Save", use_container_width=True, disabled=actions_disabled):
            if validated is None:  # pragma: no cover - defensive
                st.error("Cannot save: invalid settings.")
            else:
                _persist_env_from_validated(validated)


def _persist_env_from_validated(validated: DocMindSettings) -> None:
    """Persist validated settings to the `.env` file.

    Args:
        validated: Validated settings instance to serialize.
    """
    default_headers = getattr(validated.openai, "default_headers", None)
    context_window_value = (
        validated.context_window
        if validated.context_window is not None
        else validated.vllm.context_window
    )
    env_map = {
        "DOCMIND_LLM_BACKEND": validated.llm_backend,
        "DOCMIND_MODEL": (validated.model or ""),
        "DOCMIND_CONTEXT_WINDOW": str(int(context_window_value)),
        "DOCMIND_LLM_REQUEST_TIMEOUT_SECONDS": str(
            int(validated.llm_request_timeout_seconds)
        ),
        "DOCMIND_ENABLE_GPU_ACCELERATION": (
            "true" if validated.enable_gpu_acceleration else "false"
        ),
        "DOCMIND_OPENAI__BASE_URL": str(validated.openai.base_url).rstrip("/"),
        "DOCMIND_OPENAI__API_KEY": (
            validated.openai.api_key.get_secret_value()
            if validated.openai.api_key is not None
            else ""
        ),
        "DOCMIND_OPENAI__REQUIRE_V1": (
            "true" if getattr(validated.openai, "require_v1", True) else "false"
        ),
        "DOCMIND_OPENAI__API_MODE": str(
            getattr(validated.openai, "api_mode", "chat_completions")
        ),
        "DOCMIND_OPENAI__DEFAULT_HEADERS": (
            _safe_json_dumps_compact(default_headers) if default_headers else ""
        ),
        "DOCMIND_OLLAMA_BASE_URL": str(validated.ollama_base_url).rstrip("/"),
        "DOCMIND_OLLAMA_API_KEY": (
            validated.ollama_api_key.get_secret_value()
            if validated.ollama_api_key is not None
            else ""
        ),
        "DOCMIND_OLLAMA_ENABLE_WEB_SEARCH": (
            "true" if validated.ollama_enable_web_search else "false"
        ),
        "DOCMIND_OLLAMA_EMBED_DIMENSIONS": (
            str(validated.ollama_embed_dimensions)
            if validated.ollama_embed_dimensions is not None
            else ""
        ),
        "DOCMIND_OLLAMA_ENABLE_LOGPROBS": (
            "true" if validated.ollama_enable_logprobs else "false"
        ),
        "DOCMIND_OLLAMA_TOP_LOGPROBS": str(int(validated.ollama_top_logprobs)),
        "DOCMIND_VLLM_BASE_URL": (
            str(validated.vllm_base_url) if validated.vllm_base_url else ""
        ),
        "DOCMIND_LMSTUDIO_BASE_URL": str(validated.lmstudio_base_url),
        "DOCMIND_LLAMACPP_BASE_URL": (
            str(validated.llamacpp_base_url) if validated.llamacpp_base_url else ""
        ),
        "DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS": (
            "true" if validated.security.allow_remote_endpoints else "false"
        ),
        "DOCMIND_RETRIEVAL__RRF_K": str(int(validated.retrieval.rrf_k)),
        "DOCMIND_RETRIEVAL__TEXT_RERANK_TIMEOUT_MS": str(
            int(validated.retrieval.text_rerank_timeout_ms)
        ),
        "DOCMIND_RETRIEVAL__SIGLIP_TIMEOUT_MS": str(
            int(validated.retrieval.siglip_timeout_ms)
        ),
        "DOCMIND_RETRIEVAL__COLPALI_TIMEOUT_MS": str(
            int(validated.retrieval.colpali_timeout_ms)
        ),
        "DOCMIND_RETRIEVAL__TOTAL_RERANK_BUDGET_MS": str(
            int(validated.retrieval.total_rerank_budget_ms)
        ),
    }
    try:
        persist_env(env_map)
    except (ValueError, OSError, RuntimeError) as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="settings.save")
        key = getattr(exc, "key", "")
        suffix = f" (key={key})" if key else ""
        st.error(f"Failed to write .env{suffix}: {exc.__class__.__name__}")
        log_jsonl(
            {
                "settings.save": True,
                "success": False,
                "reason": exc.__class__.__name__,
                "error_type": exc.__class__.__name__,
                "error": redaction.redacted,
            }
        )
    else:
        st.success("Saved to .env")
        log_jsonl({"settings.save": True, "success": True})


def _render_cache_controls() -> None:
    """Render cache maintenance controls."""
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
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(e), key_id="settings.clear_caches")
            st.error(f"Failed to clear caches ({type(e).__name__}).")
            st.caption(f"Error reference: {redaction.redacted}")
            log_jsonl(
                {
                    "settings.clear_caches": True,
                    "success": False,
                    "error_type": type(e).__name__,
                    "error": redaction.redacted,
                }
            )


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
