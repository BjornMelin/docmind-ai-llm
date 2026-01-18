"""Shared helpers for Streamlit AppTest stability in CI.

Centralizes default timeout policy so AppTest-based integration/UI tests are
robust under coverage and slower CI runners while remaining fast locally.
"""

from __future__ import annotations

import os


def is_ci() -> bool:
    """Return True when running under a CI environment.

    Args:
        None.

    Returns:
        True when the process appears to be running in CI.
    """

    def _is_truthy(key: str) -> bool:
        return os.getenv(key, "").strip().lower() in {"1", "true", "yes", "on"}

    return _is_truthy("CI") or _is_truthy("GITHUB_ACTIONS")


def apptest_timeout_sec(*, default_local: int = 8, default_ci: int = 20) -> int:
    """Return a default timeout for AppTest runs.

    Args:
        default_local: Default timeout in seconds for local runs.
        default_ci: Default timeout in seconds for CI runs.

    Returns:
        Timeout in seconds after applying overrides.

    Precedence:
    1) `TEST_TIMEOUT` env var (int seconds)
    2) CI default (`default_ci`)
    3) Local default (`default_local`)
    """
    raw = os.getenv("TEST_TIMEOUT", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return max(1, int(default_ci if is_ci() else default_local))
    return max(1, int(default_ci if is_ci() else default_local))
