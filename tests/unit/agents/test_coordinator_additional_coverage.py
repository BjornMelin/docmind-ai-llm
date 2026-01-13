"""Additional coverage-focused tests for MultiAgentCoordinator.

Note: This test file intentionally accesses protected members of MultiAgentCoordinator
to test internal behavior and state transitions.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest

from src.agents.coordinator import (
    MultiAgentCoordinator,
    _supports_native_llamaindex_retries,
)
from src.agents.models import AgentResponse

pytestmark = pytest.mark.unit


def _make_langchain_model() -> Mock:
    model = Mock()
    model.invoke = Mock(return_value="ok")
    model.ainvoke = Mock(return_value="ok")
    model.stream = Mock(return_value=iter(()))
    return model


def test_supports_native_llamaindex_retries_detects_field_maps() -> None:
    class _LLM:
        model_fields: ClassVar[dict[str, object]] = {"max_retries": object()}

    class _Legacy:
        __fields__: ClassVar[dict[str, object]] = {"max_retries": object()}

    class _Nope:
        model_fields: ClassVar[dict[str, object]] = {}

    assert _supports_native_llamaindex_retries(_LLM()) is True
    assert _supports_native_llamaindex_retries(_Legacy()) is True
    assert _supports_native_llamaindex_retries(_Nope()) is False


@patch("src.config.setup_llamaindex")
@patch("llama_index.core.Settings")
def test_ensure_setup_configures_native_retry_when_supported(
    mock_settings: Mock, mock_setup: Mock
) -> None:
    class _LLM:
        model_fields: ClassVar[dict[str, object]] = {"max_retries": object()}

        def __init__(self) -> None:
            self.max_retries = None

    llamaindex_llm = _LLM()
    mock_settings.llm = llamaindex_llm

    with (
        patch("src.agents.coordinator.is_dspy_available", return_value=False),
        patch("src.agents.coordinator._shared_llm_retries", return_value=2),
        patch(
            "src.agents.coordinator.build_chat_model",
            return_value=_make_langchain_model(),
        ),
        patch.object(MultiAgentCoordinator, "_setup_agent_graph", return_value=None),
    ):
        coord = MultiAgentCoordinator(use_shared_llm_client=True)
        assert coord._ensure_setup() is True
        assert llamaindex_llm.max_retries == 2
        assert coord._shared_llm_wrapper is None


@patch("src.config.setup_llamaindex")
@patch("llama_index.core.Settings")
def test_ensure_setup_wraps_llm_when_native_retry_assignment_fails(
    mock_settings: Mock, mock_setup: Mock
) -> None:
    """Test that LLM is wrapped when native max_retries assignment fails."""

    class _LLM:
        """Test LLM that raises on max_retries assignment."""

        model_fields: ClassVar[dict[str, object]] = {"max_retries": object()}

        @property
        def max_retries(self) -> int:  # type: ignore[no-untyped-def]
            return 0

        @max_retries.setter
        def max_retries(self, _v: int) -> None:
            raise ValueError("nope")

    llamaindex_llm = _LLM()
    mock_settings.llm = llamaindex_llm

    with (
        patch("src.agents.coordinator.is_dspy_available", return_value=False),
        patch("src.agents.coordinator._shared_llm_attempts", return_value=3),
        patch("src.agents.coordinator._shared_llm_retries", return_value=1),
        patch(
            "src.agents.coordinator.build_chat_model",
            return_value=_make_langchain_model(),
        ),
        patch.object(MultiAgentCoordinator, "_setup_agent_graph", return_value=None),
    ):
        coord = MultiAgentCoordinator(use_shared_llm_client=True)
        assert coord._ensure_setup() is True
        assert coord._shared_llm_wrapper is not None
        assert coord.llamaindex_llm == coord._shared_llm_wrapper


def test_run_agent_workflow_marks_timeout_and_handles_model_dump(monkeypatch) -> None:
    coord = MultiAgentCoordinator(max_agent_timeout=1.0)

    class _State:
        """Test mock implementing state protocol."""

        def model_dump(self) -> dict:  # type: ignore[no-untyped-def]
            return {"messages": []}

    compiled = Mock()
    compiled.stream.return_value = iter([_State()])
    coord.compiled_graph = compiled

    monkeypatch.setattr("src.agents.coordinator.time.perf_counter", lambda: 10.0)
    out = coord._run_agent_workflow(
        {"total_start_time": 0.0},
        thread_id="t",
        user_id="u",
        checkpoint_id=None,
        runtime_context=None,
    )
    assert out.get("timed_out") is True
    assert out.get("deadline_s") == 1.0


def test_run_agent_workflow_returns_initial_state_when_stream_empty() -> None:
    """Test _run_agent_workflow returns init state when graph stream is empty."""
    coord = MultiAgentCoordinator(max_agent_timeout=1.0)
    compiled = Mock()
    compiled.stream.return_value = iter(())
    coord.compiled_graph = compiled
    initial = {"total_start_time": 0.0, "messages": []}
    out = coord._run_agent_workflow(
        initial,
        thread_id="t",
        user_id="u",
        checkpoint_id=None,
        runtime_context=None,
    )
    assert out == initial


def test_process_query_timeout_uses_fallback_once(monkeypatch) -> None:
    """Test process_query marks fallback_used on timeout + increments fallback count."""

    class _Registry:
        """Test mock implementing ToolRegistry protocol."""

        def build_tools_data(self, _ov):  # type: ignore[no-untyped-def]
            return {}

        def get_router_tools(self, _ctx=None):  # type: ignore[no-untyped-def]
            return []

        def get_planner_tools(self, _ctx=None):  # type: ignore[no-untyped-def]
            return []

        def get_retrieval_tools(self, _ctx=None):  # type: ignore[no-untyped-def]
            return []

        def get_synthesis_tools(self, _ctx=None):  # type: ignore[no-untyped-def]
            return []

        def get_validation_tools(self, _ctx=None):  # type: ignore[no-untyped-def]
            return []

    coord = MultiAgentCoordinator(enable_fallback=True, tool_registry=_Registry())  # type: ignore[arg-type]
    monkeypatch.setattr(coord, "_ensure_setup", lambda: True)
    monkeypatch.setattr(
        coord,
        "_build_initial_state",
        lambda _q, start_time, tools_data: {
            "messages": [],
            "total_start_time": start_time,
            "tools_data": tools_data,
        },
    )
    monkeypatch.setattr(
        coord, "_run_agent_workflow", lambda *_a, **_k: {"timed_out": True}
    )
    monkeypatch.setattr(coord, "_start_span", lambda *_a, **_k: Mock())
    monkeypatch.setattr(coord, "_record_query_metrics", lambda *_a, **_k: None)

    resp = coord.process_query("q", context=None, thread_id="t", user_id="u")
    assert isinstance(resp, AgentResponse)
    assert resp.metadata.get("fallback_used") is True
    assert resp.metadata.get("reason") == "timeout"
    assert coord.fallback_queries == 1


def test_get_state_values_and_list_checkpoints_use_compiled_graph(monkeypatch) -> None:
    """Test get_state_values and list_checkpoints methods use compiled_graph."""
    coord = MultiAgentCoordinator()
    monkeypatch.setattr(coord, "_ensure_setup", lambda: True)

    compiled = Mock()
    compiled.get_state.return_value = SimpleNamespace(values={"messages": ["x"]})
    compiled.get_state_history.return_value = iter([
        SimpleNamespace(
            config={"configurable": {"checkpoint_id": "c1", "checkpoint_ns": "ns"}}
        ),
        SimpleNamespace(config={"configurable": {"checkpoint_id": "c0"}}),
    ])
    coord.compiled_graph = compiled

    values = coord.get_state_values(thread_id="t", user_id="u")
    assert values == {"messages": ["x"]}

    cps = coord.list_checkpoints(thread_id="t", user_id="u", limit=2)
    assert cps[0]["checkpoint_id"] == "c1"
    assert cps[0]["checkpoint_ns"] == "ns"
    assert cps[1]["checkpoint_id"] == "c0"

    status = coord.validate_system_status()
    assert status["model_configured"] is True
