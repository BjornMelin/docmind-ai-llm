"""Cache utilities for Streamlit and application settings.

Provides helpers to clear Streamlit caches and bump a global cache version
salt stored in settings. These helpers are UI-agnostic to enable unit testing.
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import streamlit as st
from loguru import logger


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

    try:
        cur = int(getattr(settings_obj, "cache_version", 0))
        settings_obj.cache_version = cur + 1
    except (ValueError, TypeError, AttributeError):
        settings_obj.cache_version = 1

    # Best-effort cache clearing; never raise
    with suppress(Exception):  # pragma: no cover - defensive
        try:
            st.cache_data.clear()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("failed to clear st.cache_data", exc_info=exc)
    with suppress(Exception):  # pragma: no cover - defensive
        try:
            st.cache_resource.clear()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("failed to clear st.cache_resource", exc_info=exc)

    return int(getattr(settings_obj, "cache_version", 0))


__all__ = ["clear_caches"]
