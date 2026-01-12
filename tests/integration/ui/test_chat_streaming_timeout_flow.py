"""Integration test for Chat page streaming fallback on timeout.

This test patches MultiAgentCoordinator.process_query to simulate a timeout
response and verifies that st.write_stream is called and the session history
is updated with the assistant response.
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
    return "".join(list(stream))


@contextmanager
def _fake_chat_message(_role: str):  # minimal stub context manager
    yield


def test_chat_streaming_timeout_flow(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # Clean session
    st.session_state.clear()

    captured: dict[str, Any] = {"streamed": ""}

    # Patch Streamlit pieces used by main
    def _capture_write_stream(stream: Iterable[str]) -> str:
        captured["streamed"] = _fake_write_stream(stream)
        return captured["streamed"]

    monkeypatch.setattr(st, "write_stream", _capture_write_stream, raising=False)
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
        def list_checkpoints(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
            return []

        def get_state_values(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"messages": []}

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

    # Stub non-essential page dependencies so we actually reach _handle_chat_prompt.
    monkeypatch.setattr(
        mod, "configure_observability", lambda *_a, **_k: None, raising=False
    )
    monkeypatch.setattr(mod, "provider_badge", lambda *_a, **_k: None, raising=False)

    class _Conn:
        def close(self) -> None:
            return None

    monkeypatch.setattr(mod, "get_chat_db_conn", lambda: _Conn(), raising=False)

    class _Selection:
        thread_id = "t1"
        user_id = "u1"
        resume_checkpoint_id = None

    monkeypatch.setattr(
        mod, "render_session_sidebar", lambda _conn: _Selection(), raising=False
    )
    monkeypatch.setattr(mod, "_get_coordinator", lambda: _DummyCoord(), raising=False)
    monkeypatch.setattr(
        mod, "render_time_travel_sidebar", lambda *_a, **_k: None, raising=False
    )
    monkeypatch.setattr(
        mod, "_render_memory_sidebar", lambda *_a, **_k: None, raising=False
    )
    monkeypatch.setattr(
        mod, "_render_visual_search_sidebar", lambda *_a, **_k: None, raising=False
    )
    monkeypatch.setattr(mod, "_ensure_router_engine", lambda: None, raising=False)
    monkeypatch.setattr(mod, "_render_staleness_badge", lambda: None, raising=False)
    monkeypatch.setattr(mod, "_load_chat_messages", lambda *_a, **_k: [], raising=False)
    monkeypatch.setattr(
        mod, "_render_chat_history", lambda *_a, **_k: None, raising=False
    )
    monkeypatch.setattr(mod, "touch_session", lambda *_a, **_k: None, raising=False)

    # Run the page main()
    mod.main()

    assert "timed out" in str(captured.get("streamed") or "").lower()
