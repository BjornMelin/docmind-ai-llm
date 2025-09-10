"""Unit test for Chat page router override wiring.

Verifies that when a router engine is present in Streamlit session state,
the Chat page computes a settings_override dict for the coordinator.
"""

from __future__ import annotations

import importlib

import streamlit as st

# Accessing a private helper to validate override mapping is intentional here.
# pylint: disable=protected-access


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
    st.session_state["router_engine"] = dummy
    override = get_override()
    assert isinstance(override, dict)
    assert override.get("router_engine") is dummy


def test_chat_override_forwards_optional_retrieval_components() -> None:
    """When retrieval components exist in session, they are forwarded."""
    st.session_state.clear()
    mod = importlib.import_module("src.pages.01_chat")

    st.session_state["vector_index"] = object()
    st.session_state["hybrid_retriever"] = object()
    st.session_state["graphrag_index"] = object()
    override = mod._get_settings_override()
    assert isinstance(override, dict)
    assert "vector" in override
    assert override["vector"] is st.session_state["vector_index"]
    assert "retriever" in override
    assert override["retriever"] is st.session_state["hybrid_retriever"]
    assert "kg" in override
    assert override["kg"] is st.session_state["graphrag_index"]
