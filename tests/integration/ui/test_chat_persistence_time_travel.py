"""Integration tests for chat persistence + time travel (SPEC-041 / ADR-058).

These tests use Streamlit AppTest and a lightweight coordinator stub that
persists messages via LangGraph SqliteSaver so the UI can restore history and
resume from prior checkpoints without hitting any external LLM backends.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from streamlit.testing.v1 import AppTest

from src.agents.models import AgentResponse, MultiAgentGraphState


def _build_echo_graph(*, checkpointer: SqliteSaver):
    graph: StateGraph = StateGraph(MultiAgentGraphState)

    def _respond(state: dict[str, Any]) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_human = next(
            (m for m in reversed(messages) if isinstance(m, HumanMessage)),
            None,
        )
        prompt = getattr(last_human, "content", "")
        return {"messages": [AIMessage(content=f"Echo: {prompt}")]}

    graph.add_node("respond", _respond)
    graph.set_entry_point("respond")
    graph.set_finish_point("respond")
    return graph.compile(checkpointer=checkpointer)


class _CoordinatorStub:
    """Minimal coordinator compatible with the Chat page surface area."""

    def __init__(self, *_, checkpointer: Any = None, store: Any = None, **__):
        if checkpointer is None:
            raise RuntimeError("CoordinatorStub requires a SqliteSaver checkpointer")
        self._checkpointer = checkpointer
        self._graph = _build_echo_graph(checkpointer=checkpointer)

    def process_query(
        self,
        *,
        query: str,
        context: Any | None = None,
        settings_override: dict[str, Any] | None = None,
        thread_id: str = "default",
        user_id: str = "local",
        checkpoint_id: str | None = None,
    ) -> AgentResponse:
        cfg: dict[str, Any] = {"thread_id": str(thread_id), "user_id": str(user_id)}
        if checkpoint_id:
            cfg["checkpoint_id"] = str(checkpoint_id)
        out = self._graph.invoke(
            {"messages": [HumanMessage(content=str(query))]},
            config={"configurable": cfg},
        )
        msgs = out.get("messages", []) if isinstance(out, dict) else []
        last = msgs[-1] if msgs else None
        content = getattr(last, "content", "")
        return AgentResponse(
            content=str(content),
            sources=[],
            metadata={"stub": True},
            validation_score=0.0,
            processing_time=0.0,
            optimization_metrics={},
        )

    def get_state_values(
        self,
        *,
        thread_id: str,
        user_id: str = "local",
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        cfg: dict[str, Any] = {"thread_id": str(thread_id), "user_id": str(user_id)}
        if checkpoint_id:
            cfg["checkpoint_id"] = str(checkpoint_id)
        snap = self._graph.get_state({"configurable": cfg})
        values = getattr(snap, "values", None)
        return values if isinstance(values, dict) else {}

    def list_checkpoints(
        self, *, thread_id: str, user_id: str = "local", limit: int = 20
    ) -> list[dict[str, Any]]:
        cfg: dict[str, Any] = {"thread_id": str(thread_id), "user_id": str(user_id)}
        out: list[dict[str, Any]] = []
        for snap in self._graph.get_state_history(
            {"configurable": cfg}, limit=int(limit)
        ):
            config = getattr(snap, "config", None)
            conf = config.get("configurable") if isinstance(config, dict) else None
            conf = conf if isinstance(conf, dict) else {}
            out.append({
                "checkpoint_id": conf.get("checkpoint_id"),
                "checkpoint_ns": conf.get("checkpoint_ns", ""),
            })
        return out


@pytest.fixture
def chat_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[AppTest]:
    """Create an AppTest instance for the Chat page with a coordinator stub."""
    from src.config.settings import settings as _settings  # local import

    original_data_dir = _settings.data_dir
    original_chat_path = _settings.chat.sqlite_path
    original_ops_path = _settings.database.sqlite_db_path
    original_autoload = _settings.graphrag_cfg.autoload_policy

    _settings.data_dir = tmp_path
    _settings.chat.sqlite_path = tmp_path / "chat.db"
    _settings.database.sqlite_db_path = tmp_path / "docmind.db"
    _settings.graphrag_cfg.autoload_policy = "ignore"

    st.cache_resource.clear()
    st.cache_data.clear()

    mod = ModuleType("src.agents.coordinator")
    mod.MultiAgentCoordinator = _CoordinatorStub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.agents.coordinator", mod)

    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    at = AppTest.from_file(str(page_path))
    at.default_timeout = 8
    try:
        yield at
    finally:
        _settings.data_dir = original_data_dir
        _settings.chat.sqlite_path = original_chat_path
        _settings.database.sqlite_db_path = original_ops_path
        _settings.graphrag_cfg.autoload_policy = original_autoload
        st.cache_resource.clear()
        st.cache_data.clear()


def _user_texts(app: AppTest) -> list[str]:
    texts: list[str] = []
    for msg in app.chat_message:
        if getattr(msg, "avatar", None) != "user":
            continue
        if msg.markdown:
            texts.append(str(msg.markdown[0].value))
    return texts


@pytest.mark.integration
def test_chat_persists_history_across_restart(chat_app: AppTest) -> None:
    app = chat_app.run()
    assert not app.exception
    assert _user_texts(app) == []

    app = app.chat_input[0].set_value("one").run()
    assert not app.exception
    assert _user_texts(app) == ["one"]

    # Simulate a restart by clearing caches and creating a new AppTest session.
    st.cache_resource.clear()
    st.cache_data.clear()
    root = Path(__file__).resolve().parents[3]
    page_path = root / "src" / "pages" / "01_chat.py"
    app2 = AppTest.from_file(str(page_path)).run(timeout=8)
    assert not app2.exception
    assert "one" in _user_texts(app2)


@pytest.mark.integration
def test_chat_time_travel_fork_drops_future_messages(
    chat_app: AppTest, tmp_path: Path
) -> None:
    app = chat_app.run()
    assert not app.exception

    app = app.chat_input[0].set_value("one").run()
    assert not app.exception
    app = app.chat_input[0].set_value("two").run()
    assert not app.exception
    assert _user_texts(app) == ["one", "two"]

    thread_id = str(app.session_state["chat_thread_id"] or "")
    assert thread_id

    checkpoint_select = next(
        sb for sb in app.selectbox if str(getattr(sb, "label", "")) == "Checkpoint"
    )
    candidate_ids = [str(opt) for opt in getattr(checkpoint_select, "options", [])]
    assert candidate_ids, "expected checkpoint options to be present"

    # Find a checkpoint (visible in the UI selectbox) whose state includes "one"
    # but not "two".
    conn = sqlite3.connect(str(tmp_path / "chat.db"), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        graph = _build_echo_graph(checkpointer=saver)
        fork_checkpoint: str | None = None
        for candidate in candidate_ids:
            snap = graph.get_state({
                "configurable": {"thread_id": thread_id, "checkpoint_id": candidate}
            })
            values = getattr(snap, "values", None)
            if not isinstance(values, dict):
                continue
            user_msgs = [
                getattr(m, "content", "")
                for m in values.get("messages", [])
                if isinstance(m, HumanMessage)
            ]
            if "one" in user_msgs and "two" not in user_msgs:
                assert "two" not in user_msgs
                fork_checkpoint = candidate
                break
        assert fork_checkpoint, "expected to locate a fork checkpoint"
    finally:
        conn.close()

    # Drive the time-travel UI to resume from the chosen checkpoint.
    checkpoint_select.set_value(fork_checkpoint)
    # Prevent AppTest from re-submitting the previous chat input trigger value on reruns.
    app.chat_input[0].set_value(None)
    resume_btn = next(
        b
        for b in app.button
        if str(getattr(b, "label", "")) == "Resume from checkpoint"
    )
    app = resume_btn.click().run()
    assert app.session_state["chat_resume_checkpoint_id"] == fork_checkpoint

    app = app.chat_input[0].set_value("forked").run()
    assert not app.exception
    assert _user_texts(app) == ["one", "forked"]
