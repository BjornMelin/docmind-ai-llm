"""Settings page for LLM runtime (SPEC-001, SPEC-022).

Provides provider selection, URLs, model, context window, timeout, and GPU
toggle. Supports applying runtime immediately and saving to .env with
pre-validation to avoid persisting invalid configuration.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePath

import streamlit as st
from pydantic import ValidationError

from src.config.env_persistence import persist_env
from src.config.integrations import initialize_integrations
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


def resolve_valid_gguf_path(path_text: str) -> Path | None:
    """Resolve and validate a GGUF model path.

    Security: resolve once and enforce that the resolved path lives within one
    of the allowed base directories (default: the current user home directory).

    WARNING: This function is best-effort pre-validation only. Callers MUST
    treat any subsequent file operations as untrusted and handle failures
    defensively (including TOCTOU races, permission errors, and IO errors).
    """
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

    home_dir = Path.home()
    try:
        home_dir_resolved = home_dir.expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        home_dir_resolved = home_dir

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

    if not base_dirs:
        return None

    raw_text = clean
    base_override: Path | None = None
    if clean.startswith("~/"):
        raw_text = clean[2:]
        if not raw_text:
            return None
        base_override = home_dir_resolved

    raw = PurePath(raw_text)
    if any(part == ".." for part in raw.parts):
        return None

    candidates: list[tuple[Path, Path]] = []
    if raw.is_absolute() and base_override is None:
        for base_dir in base_dirs:
            if raw.is_relative_to(base_dir):
                candidates.append((base_dir, Path(raw)))
                break
        if not candidates:
            return None
    else:
        for base_dir in base_dirs:
            if base_override is not None and base_dir != base_override:
                continue
            candidates.append((base_dir, base_dir / raw))

    for base_dir, candidate in candidates:
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
            continue

        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            continue
        try:
            resolved.relative_to(base_dir)
        except ValueError:
            continue
        if resolved.is_file() and resolved.suffix.lower() == ".gguf":
            return resolved

    return None


def _apply_validated_runtime(validated: DocMindSettings) -> None:
    """Apply runtime by updating settings then rebinding LlamaIndex Settings.llm."""
    settings.llm_backend = validated.llm_backend  # type: ignore[assignment]
    settings.model = validated.model  # type: ignore[assignment]
    settings.context_window = validated.context_window
    settings.llm_request_timeout_seconds = validated.llm_request_timeout_seconds
    settings.enable_gpu_acceleration = validated.enable_gpu_acceleration
    settings.ollama_base_url = validated.ollama_base_url  # type: ignore[assignment]
    settings.vllm_base_url = validated.vllm_base_url  # type: ignore[assignment]
    settings.lmstudio_base_url = validated.lmstudio_base_url  # type: ignore[assignment]
    settings.llamacpp_base_url = validated.llamacpp_base_url
    settings.vllm.llamacpp_model_path = validated.vllm.llamacpp_model_path  # type: ignore[assignment]
    settings.security.allow_remote_endpoints = validated.security.allow_remote_endpoints

    # Apply retrieval timeouts to in-memory settings; policy remains env-driven.
    settings.retrieval.rrf_k = validated.retrieval.rrf_k
    settings.retrieval.text_rerank_timeout_ms = (
        validated.retrieval.text_rerank_timeout_ms
    )
    settings.retrieval.siglip_timeout_ms = validated.retrieval.siglip_timeout_ms
    settings.retrieval.colpali_timeout_ms = validated.retrieval.colpali_timeout_ms
    settings.retrieval.total_rerank_budget_ms = (
        validated.retrieval.total_rerank_budget_ms
    )

    model_label = validated.model or validated.vllm.model
    try:
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
            help="OpenAI-compatible; normalized to end with /v1",
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
    with col4t:
        t_total = st.number_input(
            "Total rerank budget (ms)",
            min_value=100,
            max_value=20000,
            value=int(getattr(settings.retrieval, "total_rerank_budget_ms", 800)),
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

    ui_errors: list[str] = []

    clean_llamacpp_url = (llamacpp_url or "").strip()
    clean_gguf_path = (gguf_path or "").strip()

    is_llamacpp = provider == "llamacpp"
    resolved_gguf_path = resolve_valid_gguf_path(clean_gguf_path)
    gguf_missing_for_local = not clean_llamacpp_url and not clean_gguf_path

    if is_llamacpp and (resolved_gguf_path is None) and (not gguf_missing_for_local):
        ui_errors.append(
            "Invalid GGUF model path. File must exist, have a .gguf extension, "
            "and be under the allowed base directories."
        )

    if is_llamacpp and gguf_missing_for_local:
        ui_errors.append(
            "Provide either a llama.cpp base URL or a local GGUF model path."
        )

    # NOTE: The settings UI intentionally validates only the values it edits.
    # Missing sections are populated from DocMindSettings defaults, so this form
    # does not act as a full-validator for unrelated config blocks.
    candidate = {
        "llm_backend": provider,
        "model": model.strip() or None,
        "context_window": int(context_window),
        "ollama_base_url": ollama_url.strip(),
        "vllm_base_url": vllm_url.strip() or None,
        "lmstudio_base_url": lmstudio_url.strip(),
        "llamacpp_base_url": clean_llamacpp_url or None,
        "llm_request_timeout_seconds": int(timeout_s),
        "enable_gpu_acceleration": bool(use_gpu),
        "vllm": {
            "llamacpp_model_path": (
                str(resolved_gguf_path)
                if is_llamacpp and resolved_gguf_path is not None
                else ""
            )
        },
        "security": {
            "allow_remote_endpoints": bool(allow_remote),
            "endpoint_allowlist": list(settings.security.endpoint_allowlist),
            "trust_remote_code": bool(settings.security.trust_remote_code),
        },
        "retrieval": {
            "rrf_k": int(rrf_k),
            "text_rerank_timeout_ms": int(t_text),
            "siglip_timeout_ms": int(t_siglip),
            "colpali_timeout_ms": int(t_colpali),
            "total_rerank_budget_ms": int(t_total),
        },
    }
    validated, validation_errors = _validate_candidate(candidate)

    resolved_base_url = (
        str(getattr(validated, "backend_base_url_normalized", ""))
        if validated is not None
        else str(getattr(settings, "backend_base_url_normalized", ""))
    )
    st.caption("Resolved backend base URL (normalized)")
    st.text_input(
        "Resolved base URL",
        value=resolved_base_url,
        disabled=True,
    )

    if ui_errors or validation_errors:
        st.subheader("Validation")
        for msg in ui_errors + validation_errors:
            st.error(msg)

    # Actions
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
                    # nested path override
                    "DOCMIND_VLLM__LLAMACPP_MODEL_PATH": str(
                        validated.vllm.llamacpp_model_path
                    ),
                    "DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS": (
                        "true" if validated.security.allow_remote_endpoints else "false"
                    ),
                    # Retrieval policy is configured via env; read-only here
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
