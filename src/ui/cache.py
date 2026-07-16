"""Cache utilities for Streamlit and application settings.

Provides helpers to clear Streamlit caches and bump a global cache version
salt stored in settings. These helpers are UI-agnostic to enable unit testing.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import streamlit as st
from loguru import logger

from src.utils.log_safety import build_pii_log_entry


def clear_caches(settings_obj: Any | None = None) -> int:
    """Clear Streamlit caches and bump a global cache version.

    Args:
        settings_obj: Optional settings object. When omitted, imports the
            global settings singleton.

    Returns:
        The new cache_version value after bumping.
    """
    if settings_obj is None:
        # Lazy import to avoid heavy dependencies at import-time
        from src.config.settings import settings as _settings  # type: ignore

        settings_obj = _settings

    from src.ui.background_jobs import get_job_manager

    with get_job_manager().admission_quiescence():
        return _clear_caches_quiesced(settings_obj)


def _clear_caches_quiesced(settings_obj: Any) -> int:
    """Clear caches after the caller has paused background job admission."""
    try:
        cur = int(getattr(settings_obj, "cache_version", 0))
        settings_obj.cache_version = cur + 1
    except (ValueError, TypeError, AttributeError):
        settings_obj.cache_version = 1

    with suppress(Exception):
        from src.ui.vector_session import clear_session_runtime

        clear_session_runtime(
            st.session_state,
            runtime_generation=int(settings_obj.cache_version),
        )

    with suppress(Exception):
        from src.ui.chat_runtime import invalidate_coordinator

        invalidate_coordinator()

    # Best-effort cache clearing; never raise
    with suppress(Exception):  # pragma: no cover - defensive
        try:
            st.cache_data.clear()
        except Exception as exc:  # pragma: no cover - defensive
            redaction = build_pii_log_entry(str(exc), key_id="ui.cache_data.clear")
            logger.debug(
                "failed to clear st.cache_data (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )
    with suppress(Exception):  # pragma: no cover - defensive
        try:
            st.cache_resource.clear()
        except Exception as exc:  # pragma: no cover - defensive
            redaction = build_pii_log_entry(str(exc), key_id="ui.cache_resource.clear")
            logger.debug(
                "failed to clear st.cache_resource (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )

    return int(getattr(settings_obj, "cache_version", 0))


__all__ = ["clear_caches"]
