"""Programmatic multipage app entrypoint (SPEC-008).

All UI pages live under `src/pages/`. This module is intentionally kept as a
thin multipage shell with no business logic or helper wrappers.

This module is launched via the repository root `app.py`. Running `src/app.py`
directly is not a supported entrypoint.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import bootstrap_settings, settings
from src.persistence.snapshot import recover_snapshot_transactions


@st.cache_resource(show_spinner=False)
def _recover_persistence_once(data_dir: str, cache_version: int) -> None:
    """Recover interrupted persistence transactions once per runtime generation."""
    del cache_version
    recover_snapshot_transactions(Path(data_dir) / "storage")


def main() -> None:  # pragma: no cover - Streamlit entrypoint
    """Initialize and run the DocMind AI Streamlit application.

    This function sets up the Streamlit page configuration and defines the
    navigation structure for the multipage application. It configures the
    main pages including chat, documents, analytics, and settings.

    The function does not take any parameters and does not return any values.
    It runs the Streamlit navigation loop to handle page routing.
    """
    bootstrap_settings()
    app_title = getattr(settings, "app_name", "DocMind AI")
    st.set_page_config(page_title=app_title, page_icon="🧠", layout="wide")
    try:
        _recover_persistence_once(
            str(settings.data_dir.resolve()), settings.cache_version
        )
    except (OSError, RuntimeError, ValueError):
        st.error(
            "DocMind could not safely recover local persistence. "
            "Stop other writers and restart the app."
        )
        st.stop()

    chat = st.Page(
        "src/pages/01_chat.py",
        title="Chat",
        icon=":material/chat:",
        default=True,
    )
    docs = st.Page(
        "src/pages/02_documents.py",
        title="Documents",
        icon=":material/description:",
    )
    analytics = st.Page(
        "src/pages/03_analytics.py",
        title="Analytics",
        icon=":material/insights:",
    )
    settings_page = st.Page(
        "src/pages/04_settings.py",
        title="Settings",
        icon=":material/settings:",
    )

    st.navigation([chat, docs, analytics, settings_page]).run()
