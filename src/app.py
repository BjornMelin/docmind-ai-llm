"""Programmatic multipage app entrypoint (SPEC-008).

All UI pages live under `src/pages/`. This module is intentionally kept as a
thin multipage shell with no business logic or helper wrappers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

project_root = Path(__file__).resolve().parents[1]
root_str = str(project_root)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from src.config import bootstrap_settings, settings  # noqa: E402


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


if __name__ == "__main__":  # pragma: no cover
    main()
