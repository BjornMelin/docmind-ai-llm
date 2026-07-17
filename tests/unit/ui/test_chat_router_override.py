"""Unit test for Chat page router override wiring.

Verifies that when a router engine is present in Streamlit session state,
the Chat page computes a settings_override dict for the coordinator.
"""

from __future__ import annotations

import importlib

import streamlit as st

from src.ui.router_session import replace_session_router

# Accessing a private helper to validate override mapping is intentional here.


def test_chat_router_override_flag_roundtrip() -> None:
    """Roundtrip: no router -> None; router present -> dict with key."""
    # Ensure a clean session state
    st.session_state.clear()

    # Import the chat page module
    mod = importlib.import_module("src.pages.01_chat")

    # When no router engine is present, override should be None
    get_override = mod._get_settings_override
    assert get_override() is None

    # Insert a dummy router engine and expect a mapping
    dummy = object()
    replace_session_router(
        st.session_state,
        dummy,
        runtime_generation=mod.settings.cache_version,
    )
    override = get_override()
    assert isinstance(override, dict)
    assert override.get("router_engine") is dummy


def test_chat_override_ignores_raw_retrieval_components() -> None:
    """The v2 coordinator boundary forwards only the prebuilt router."""
    st.session_state.clear()
    mod = importlib.import_module("src.pages.01_chat")

    st.session_state["vector_index"] = object()
    st.session_state["graphrag_index"] = object()
    assert mod._get_settings_override() is None

    router = object()
    replace_session_router(
        st.session_state,
        router,
        runtime_generation=mod.settings.cache_version,
    )
    assert mod._get_settings_override() == {"router_engine": router}
