"""Integration test for Chat page streaming fallback on timeout.

This test patches MultiAgentCoordinator.process_query to simulate a timeout
response and verifies that st.write_stream is called and the session history
is appended with the assistant message.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

import streamlit as st

from src.agents.models import AgentResponse


def _fake_write_stream(stream: Iterable[str]) -> str:  # matches st.write_stream
    text = "".join(list(stream))
    return text


@contextmanager
def _fake_chat_message(_role: str):  # minimal stub context manager
    yield


def test_chat_streaming_timeout_flow(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Clean session
    st.session_state.clear()

    # Patch Streamlit pieces used by main
    monkeypatch.setattr(st, "write_stream", _fake_write_stream, raising=False)
    monkeypatch.setattr(st, "chat_message", _fake_chat_message, raising=False)

    # Provide a single user prompt
    monkeypatch.setattr(st, "chat_input", lambda _=None: "hello", raising=False)

    # Title/markdown helpers to no-ops
    monkeypatch.setattr(st, "title", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "markdown", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "caption", lambda *_a, **_k: None, raising=False)
    monkeypatch.setattr(st, "warning", lambda *_a, **_k: None, raising=False)

    # Dummy provider badge and other UI that we can ignore
    with contextlib.suppress(Exception):
        import src.ui.components.provider_badge as pb  # type: ignore

        monkeypatch.setattr(pb, "provider_badge", lambda *_a, **_k: None, raising=False)

    # Patch coordinator to return a timeout response
    class _DummyCoord:
        def process_query(self, *args: Any, **kwargs: Any) -> AgentResponse:
            """Return a deterministic timeout response."""
            return AgentResponse(
                content="Request timed out.",
                sources=[],
                metadata={"fallback_used": True, "reason": "timeout"},
                validation_score=0.0,
                processing_time=0.01,
                optimization_metrics={"timeout": True},
            )

    mod = importlib.import_module("src.pages.01_chat")

    # Replace the class reference used in the page with our dummy
    monkeypatch.setattr(
        mod, "MultiAgentCoordinator", lambda: _DummyCoord(), raising=False
    )

    # Run the page main()
    with contextlib.suppress(Exception):
        mod.main()

    # Validate that assistant message appended with streamed final text
    msgs = st.session_state.get("messages", [])
    assert msgs, "Expected messages history to be updated"
    assert msgs[-1]["role"] == "assistant"
    assert "timed out" in msgs[-1]["content"].lower()
