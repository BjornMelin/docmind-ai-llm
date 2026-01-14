"""Settings page for LLM runtime (SPEC-001, SPEC-022).

Provides provider selection, URLs, model, context window, timeout, and GPU
toggle. Supports applying runtime immediately and saving to .env with
pre-validation to avoid persisting invalid configuration.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePath
from typing import Any

import streamlit as st
from pydantic import ValidationError

from src.config.env_persistence import persist_env
from src.config.settings import DocMindSettings, settings
from src.retrieval.adapter_registry import get_default_adapter_health
from src.ui.components.provider_badge import provider_badge
from src.utils.telemetry import log_jsonl


def _validate_candidate(
    candidate: dict[str, object],
) -> tuple[DocMindSettings | None, list[str]]:
    """Validate a candidate settings payload before Apply/Save.

    The ``candidate`` argument is expected to be a dictionary whose keys
    correspond to the fields of :class:`DocMindSettings`. Typical keys include
    top-level runtime options such as ``"llm_backend"``, ``"model"``,
    ``"context_window"``, ``"llm_request_timeout_seconds"``,
    ``"enable_gpu_acceleration"``, provider URLs like ``"ollama_base_url"``,
    ``"vllm_base_url"``, ``"lmstudio_base_url"``, ``"llamacpp_base_url"``,
    and nested sections such as ``"retrieval"`` (e.g. ``"rrf_k"``,
    ``"text_rerank_timeout_ms"``, ``"siglip_timeout_ms"``,
    ``"colpali_timeout_ms"``, ``"total_rerank_budget_ms"``) and ``"security"``
    (e.g. ``"allow_remote_endpoints"``).

    Args:
        candidate: A dict-like settings payload (for example, data collected
            from the settings form or from ``st.session_state``) that matches
            the schema of :class:`DocMindSettings`.

    Returns:
        A tuple ``(validated_settings, error_messages)`` where
        ``validated_settings`` is a :class:`DocMindSettings` instance on
        success and ``None`` on failure, and ``error_messages`` is a list of
        human-readable validation error messages (empty when validation
        succeeds).
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


_GGUF_PATH_PATTERN = re.compile(r"^[A-Za-z0-9._ \-~/\\\\:]+$")


def _get_resolved_home_dir() -> Path:
    """Resolve the home directory with fallback to original on errors.

    Returns:
        Expanded and resolved home directory path, or original Path.home() if
        resolution fails due to OS, runtime, or value errors.
    """
    home_dir = Path.home()
    try:
        return home_dir.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return home_dir


def _clean_gguf_path_text(path_text: str) -> str | None:
    """Validate and normalize raw GGUF path input."""
    clean = (path_text or "").strip()
    if not clean:
        return None
    if clean.startswith("~") and not clean.startswith("~/"):
        return None
    if (
        "\x00" in clean
        or any(ord(ch) < 32 for ch in clean)
        or _GGUF_PATH_PATTERN.fullmatch(clean) is None
    ):
        return None
    return clean


def _resolve_allowed_gguf_bases() -> list[Path]:
    """Resolve allowed base directories for GGUF paths."""
    home_dir_resolved = _get_resolved_home_dir()

    allowed_bases = [home_dir_resolved]
    try:
        extra_bases = st.session_state.get("docmind_allowed_gguf_base_dirs")
    except Exception:  # pragma: no cover - defensive
        extra_bases = None
    if isinstance(extra_bases, (list, tuple)):
        allowed_bases.extend(
            [Path(str(base)) for base in extra_bases if str(base).strip()]
        )

    base_dirs: list[Path] = []
    for base in allowed_bases:
        try:
            base_dir = Path(str(base)).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        base_dirs.append(base_dir)
    return base_dirs


def _build_gguf_candidates(
    clean: str, base_dirs: list[Path], home_dir_resolved: Path
) -> list[tuple[Path, Path]]:
    """Construct candidate GGUF paths under allowed bases."""
    raw_text = clean
    base_override: Path | None = None
    if clean.startswith("~/"):
        raw_text = clean[2:]
        if not raw_text:
            return []
        base_override = home_dir_resolved

    raw = PurePath(raw_text)
    if any(part == ".." for part in raw.parts):
        return []

    candidates: list[tuple[Path, Path]] = []
    if raw.is_absolute() and base_override is None:
        for base_dir in base_dirs:
            if raw.is_relative_to(base_dir):
                candidates.append((base_dir, Path(raw)))
                break
        return candidates

    for base_dir in base_dirs:
        if base_override is not None and base_dir != base_override:
            continue
        candidates.append((base_dir, base_dir / raw))
    return candidates


def _is_valid_gguf_candidate(base_dir: Path, candidate: Path) -> Path | None:
    """Return a resolved GGUF path if candidate passes safety checks."""
    stop_at = base_dir
    symlink_found = False
    reached_base = False
    for part in (candidate, *candidate.parents):
        if part.is_symlink():
            symlink_found = True
            break
        if part == stop_at:
            reached_base = True
            break
    if symlink_found or not reached_base:
        return None

    try:
        resolved = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None
    try:
        resolved.relative_to(base_dir)
    except ValueError:
        return None
    if resolved.is_file() and resolved.suffix.lower() == ".gguf":
        return resolved
    return None


def resolve_valid_gguf_path(path_text: str) -> Path | None:
    """Resolve and validate a GGUF model path.

    Security: resolve once and enforce that the resolved path lives within one
    of the allowed base directories (default: the current user home directory).

    WARNING: This function is best-effort pre-validation only. Callers MUST
    treat any subsequent file operations as untrusted and handle failures
    defensively (including TOCTOU races, permission errors, and IO errors).
    """
    clean = _clean_gguf_path_text(path_text)
    if clean is None:
        return None

    base_dirs = _resolve_allowed_gguf_bases()
    if not base_dirs:
        return None

    home_dir_resolved = _get_resolved_home_dir()

    candidates = _build_gguf_candidates(clean, base_dirs, home_dir_resolved)
    for base_dir, candidate in candidates:
        resolved = _is_valid_gguf_candidate(base_dir, candidate)
        if resolved is not None:
            return resolved
    return None


def _apply_validated_runtime(validated: DocMindSettings) -> None:
    """Apply runtime by updating settings then rebinding LlamaIndex Settings.llm."""
    updated = settings.model_copy(
        update={
            "llm_backend": validated.llm_backend,
            "model": validated.model,
            "context_window": validated.context_window,
            "llm_request_timeout_seconds": validated.llm_request_timeout_seconds,
            "enable_gpu_acceleration": validated.enable_gpu_acceleration,
            "ollama_base_url": validated.ollama_base_url,
            "vllm_base_url": validated.vllm_base_url,
            "lmstudio_base_url": validated.lmstudio_base_url,
            "llamacpp_base_url": validated.llamacpp_base_url,
            "vllm": settings.vllm.model_copy(
                update={
                    "llamacpp_model_path": validated.vllm.llamacpp_model_path,
                }
            ),
            "security": settings.security.model_copy(
                update={
                    "allow_remote_endpoints": validated.security.allow_remote_endpoints,
                }
            ),
            "retrieval": settings.retrieval.model_copy(
                update={
                    "rrf_k": validated.retrieval.rrf_k,
                    "text_rerank_timeout_ms": (
                        validated.retrieval.text_rerank_timeout_ms
                    ),
                    "siglip_timeout_ms": validated.retrieval.siglip_timeout_ms,
                    "colpali_timeout_ms": validated.retrieval.colpali_timeout_ms,
                    "total_rerank_budget_ms": (
                        validated.retrieval.total_rerank_budget_ms
                    ),
                }
            ),
        }
    )
    # Apply updated settings in-place so existing imports keep the same instance.
    for field in settings.model_fields:
        try:
            value = getattr(updated, field)
        except AttributeError:
            continue
        setattr(settings, field, value)

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
        st.error(f"Runtime apply failed: {exc.__class__.__name__}")
        log_jsonl(
            {
                "settings.apply": True,
                "success": False,
                "backend": validated.llm_backend,
                "model": model_label,
                "reason": exc.__class__.__name__,
                "error": str(exc),
            }
        )
        return

    from llama_index.core import Settings as LISettings  # local import (tests patch)

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
    graphrag_health = get_default_adapter_health()
    provider_badge(settings, graphrag_health=graphrag_health)
    provider = _render_provider_section()
    model, context_window, timeout_s, use_gpu = _render_model_section()
    ollama_url, vllm_url, lmstudio_url, llamacpp_url = _render_provider_urls()
    gguf_path = _render_gguf_path()
    allow_remote = _render_security_section()
    rrf_k, t_text, t_siglip, t_colpali, t_total = _render_retrieval_section()
    _render_graphrag_section(graphrag_health)

    ui_errors, resolved_gguf_path = _validate_gguf_inputs(
        provider, llamacpp_url, gguf_path
    )
    values = {
        "provider": provider,
        "model": model,
        "context_window": context_window,
        "ollama_url": ollama_url,
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
    candidate = _build_candidate_settings(values, resolved_gguf_path)
    validated, validation_errors = _validate_candidate(candidate)
    _render_resolved_base_url(validated)
    _render_validation(ui_errors, validation_errors)
    _render_actions(validated, ui_errors)
    _render_cache_controls()


def _render_provider_section() -> str:
    """Render provider selector and return selection."""
    st.subheader("Provider")
    return st.selectbox(
        "LLM Provider",
        options=["ollama", "vllm", "lmstudio", "llamacpp"],
        index=(["ollama", "vllm", "lmstudio", "llamacpp"].index(settings.llm_backend)),
        help="Select the active LLM backend",
    )


def _render_model_section() -> tuple[str, int, int, bool]:
    """Render model/context inputs and return values."""
    st.subheader("Model & Context")
    model = st.text_input(
        "Model (id or GGUF path)",
        value=(settings.model or settings.vllm.model),
        help=("Model identifier (Ollama/vLLM/LM Studio) or GGUF path (LlamaCPP)"),
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


def _render_provider_urls() -> tuple[str, str, str, str]:
    """Render provider URL inputs and return values."""
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
            help="OpenAI-compatible; normalized to end with /v1",
        )
        llamacpp_url = st.text_input(
            "llama.cpp server URL (optional)",
            value=(settings.llamacpp_base_url or ""),
            placeholder="http://localhost:8080/v1",
        )
    return ollama_url, vllm_url, lmstudio_url, llamacpp_url


def _render_gguf_path() -> str:
    """Render GGUF path input and return value."""
    return st.text_input(
        "GGUF model path (LlamaCPP local)",
        value=str(settings.vllm.llamacpp_model_path),
    )


def _render_security_section() -> bool:
    """Render security controls and return allow-remote setting."""
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
    return allow_remote


def _render_retrieval_section() -> tuple[int, int, int, int, int]:
    """Render retrieval policy inputs and return values."""
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
    """Render GraphRAG status section."""
    st.subheader("GraphRAG")
    if graphrag_health is None:
        graphrag_health = get_default_adapter_health()
    supports, adapter_name, hint = graphrag_health
    st.text_input("Adapter", value=adapter_name, disabled=True)
    st.text_input(
        "GraphRAG status",
        value="enabled" if supports else "disabled",
        disabled=True,
    )
    if not supports:
        st.info(hint)


def _validate_gguf_inputs(
    provider: str, llamacpp_url: str, gguf_path: str
) -> tuple[list[str], Path | None]:
    """Validate GGUF path inputs for llama.cpp settings."""
    ui_errors: list[str] = []
    clean_llamacpp_url = (llamacpp_url or "").strip()
    clean_gguf_path = (gguf_path or "").strip()
    is_llamacpp = provider == "llamacpp"
    if not is_llamacpp:
        return ui_errors, None

    # Server mode: base URL provided -> no local GGUF validation required.
    if clean_llamacpp_url:
        return ui_errors, None

    # Local mode: require a valid GGUF path.
    if not clean_gguf_path:
        ui_errors.append(
            "Provide either a llama.cpp base URL or a local GGUF model path."
        )
        return ui_errors, None

    resolved_gguf_path = resolve_valid_gguf_path(clean_gguf_path)
    if resolved_gguf_path is None:
        ui_errors.append(
            "Invalid GGUF model path. File must exist, have a .gguf extension, "
            "and be under the allowed base directories."
        )
        return ui_errors, None

    return ui_errors, resolved_gguf_path


def _build_candidate_settings(
    values: dict[str, Any], resolved_gguf_path: Path | None
) -> dict[str, Any]:
    """Build a settings payload for validation."""
    provider = str(values.get("provider", ""))
    is_llamacpp = provider == "llamacpp"
    llamacpp_model_path = (
        str(resolved_gguf_path)
        if is_llamacpp and resolved_gguf_path is not None
        else str(settings.vllm.llamacpp_model_path)
    )
    return {
        "llm_backend": provider,
        "model": str(values.get("model", "")).strip() or None,
        "context_window": int(values.get("context_window", 0)),
        "ollama_base_url": str(values.get("ollama_url", "")).strip(),
        "vllm_base_url": str(values.get("vllm_url", "")).strip() or None,
        "lmstudio_base_url": str(values.get("lmstudio_url", "")).strip(),
        "llamacpp_base_url": str(values.get("llamacpp_url", "")).strip() or None,
        "llm_request_timeout_seconds": int(values.get("timeout_s", 0)),
        "enable_gpu_acceleration": bool(values.get("use_gpu")),
        "vllm": {"llamacpp_model_path": llamacpp_model_path},
        "security": {
            "allow_remote_endpoints": bool(values.get("allow_remote")),
            "endpoint_allowlist": list(settings.security.endpoint_allowlist),
            "trust_remote_code": bool(settings.security.trust_remote_code),
        },
        "retrieval": {
            "rrf_k": int(values.get("rrf_k", 0)),
            "text_rerank_timeout_ms": int(values.get("t_text", 0)),
            "siglip_timeout_ms": int(values.get("t_siglip", 0)),
            "colpali_timeout_ms": int(values.get("t_colpali", 0)),
            "total_rerank_budget_ms": int(values.get("t_total", 0)),
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
    """Persist validated settings to the .env file."""
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
        "DOCMIND_OLLAMA_BASE_URL": validated.ollama_base_url,
        "DOCMIND_VLLM_BASE_URL": (validated.vllm_base_url or ""),
        "DOCMIND_LMSTUDIO_BASE_URL": validated.lmstudio_base_url,
        "DOCMIND_LLAMACPP_BASE_URL": (validated.llamacpp_base_url or ""),
        "DOCMIND_VLLM__LLAMACPP_MODEL_PATH": str(validated.vllm.llamacpp_model_path),
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
        key = getattr(exc, "key", "")
        suffix = f" (key={key})" if key else ""
        st.error(f"Failed to write .env{suffix}: {exc}")
    else:
        st.success("Saved to .env")


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
            st.error(f"Failed to clear caches: {e}")


if __name__ == "__main__":  # pragma: no cover - manual launch
    main()
