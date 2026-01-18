"""Additional coverage-focused tests for MultiAgentCoordinator.

Note: This test file intentionally accesses protected members of MultiAgentCoordinator
to test internal behavior and state transitions.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock, patch

import pytest

from src.agents.coordinator import (
    MultiAgentCoordinator,
    _as_state_dict,
    _coerce_deadline_ts,
    _ensure_event_loop,
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
    """Test that _supports_native_llamaindex_retries detects field maps."""

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
    """Test that _ensure_setup configures native retry when supported.

    Args:
        mock_settings: Mocked llama_index Settings object.
        mock_setup: Mocked setup_llamaindex callable.

    Returns:
        None.
    """

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


@pytest.fixture
def workflow_result_fixture() -> dict[str, object]:
    """Reusable fixture for coordinator state/result shape."""
    return {
        "messages": [],
        "total_start_time": 0.0,
        "timed_out": False,
        "deadline_s": 0.0,
        # Add other common fields as needed
    }


def test_run_agent_workflow_marks_timeout_and_handles_model_dump(
    monkeypatch, workflow_result_fixture
) -> None:
    """Test that _run_agent_workflow marks timeout and handles model dump.

    Args:
        monkeypatch: Pytest fixture for patching and mocking.
        workflow_result_fixture: Fixture providing base state dict for coordinator.

    Returns:
        None.
    """
    coord = MultiAgentCoordinator(max_agent_timeout=1.0)

    class _State:
        """Test mock implementing state protocol."""

        def model_dump(self) -> dict:  # type: ignore[no-untyped-def]
            return workflow_result_fixture.copy()

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


def test_run_agent_workflow_returns_initial_state_when_stream_empty(
    workflow_result_fixture,
) -> None:
    """Test _run_agent_workflow returns init state when graph stream is empty."""
    coord = MultiAgentCoordinator(max_agent_timeout=1.0)
    compiled = Mock()
    compiled.stream.return_value = iter(())
    coord.compiled_graph = compiled
    initial = workflow_result_fixture.copy()
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


def test_ensure_event_loop_creates_loop_when_missing() -> None:
    """_ensure_event_loop should set a loop when none is present."""
    import threading

    errors: list[Exception] = []

    def _worker() -> None:
        try:
            with contextlib.suppress(RuntimeError):
                asyncio.set_event_loop(None)
            _ensure_event_loop()
            loop = asyncio.get_event_loop_policy().get_event_loop()
            assert loop is not None
            loop.close()
            with contextlib.suppress(RuntimeError):
                asyncio.set_event_loop(None)
        except Exception as exc:  # pragma: no cover - surfaced by main thread
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    assert not errors


def test_as_state_dict_normalizes_inputs() -> None:
    """_as_state_dict handles dicts, model_dump, and fallbacks."""
    assert _as_state_dict({"a": 1}) == {"a": 1}

    class _State:
        def model_dump(self):  # type: ignore[no-untyped-def]
            return {"b": 2}

    class _BadState:
        def model_dump(self):  # type: ignore[no-untyped-def]
            return ["nope"]

    assert _as_state_dict(_State()) == {"b": 2}
    assert _as_state_dict(_BadState()) == {}


def test_coerce_deadline_ts_handles_invalid_values() -> None:
    """_coerce_deadline_ts returns valid floats or None."""
    assert _coerce_deadline_ts({"deadline_ts": "12.5"}) == 12.5
    assert _coerce_deadline_ts({"deadline_ts": None}) is None
    assert _coerce_deadline_ts({"deadline_ts": "bad"}) is None


def test_record_query_metrics_emits_histogram_and_counter(monkeypatch) -> None:
    """_record_query_metrics uses OTEL meter to record data.

    Args:
        monkeypatch: Pytest fixture for patching and mocking.

    Returns:
        None.
    """
    calls = {"record": 0, "add": 0}

    class _Hist:
        def record(self, _val, attributes=None):  # type: ignore[no-untyped-def]
            assert attributes is not None
            calls["record"] += 1

    class _Counter:
        def add(self, _val, attributes=None):  # type: ignore[no-untyped-def]
            assert attributes is not None
            calls["add"] += 1

    class _Meter:
        def create_histogram(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return _Hist()

        def create_counter(self, *_a, **_k):  # type: ignore[no-untyped-def]
            return _Counter()

    record_globals = MultiAgentCoordinator._record_query_metrics.__globals__
    monkeypatch.setattr(
        record_globals["metrics"], "get_meter", lambda *_a, **_k: _Meter()
    )

    with patch.dict(
        record_globals,
        {"_COORDINATOR_LATENCY": None, "_COORDINATOR_COUNTER": None},
        clear=False,
    ):
        inst = MultiAgentCoordinator()
        inst._record_query_metrics(0.12, success=True)
        assert calls["record"] == 1
        assert calls["add"] == 1


def test_handle_timeout_response_without_fallback() -> None:
    """_handle_timeout_response returns timeout response when fallback disabled."""
    coord = MultiAgentCoordinator(enable_fallback=False)
    response, used = coord._handle_timeout_response("q", context=None, start_time=0.0)
    assert used is True
    assert response.metadata.get("fallback_used") is True
    assert response.metadata.get("reason") == "timeout"


def test_semantic_cache_lookup_skips_when_disabled() -> None:
    """_maybe_semantic_cache_lookup returns early when cache disabled."""
    coord = MultiAgentCoordinator()
    response, key = coord._maybe_semantic_cache_lookup(
        query="q",
        thread_id="t",
        user_id="u",
        checkpoint_id=None,
        start_time=0.0,
        span=Mock(),
    )
    assert response is None
    assert key is None


def test_semantic_cache_store_skips_when_disabled() -> None:
    """_maybe_semantic_cache_store returns without side effects when disabled."""
    coord = MultiAgentCoordinator()
    coord._maybe_semantic_cache_store(
        semantic_cache_key=Mock(), query="q", response_text="r"
    )


def test_handle_workflow_result_timeout_path(monkeypatch) -> None:
    """_handle_workflow_result routes timeout results to fallback handler."""
    coord = MultiAgentCoordinator()
    monkeypatch.setattr(
        coord,
        "_handle_timeout_response",
        lambda *_a, **_k: (AgentResponse(content="x"), True),
    )
    monkeypatch.setattr(coord, "_record_query_metrics", lambda *_a, **_k: None)
    response, timed_out, used_fallback = coord._handle_workflow_result(
        {"timed_out": True}, "q", None, 0.0, 0.0
    )
    assert timed_out is True
    assert used_fallback is True
    assert isinstance(response, AgentResponse)


def test_handle_workflow_result_non_timeout(monkeypatch) -> None:
    """_handle_workflow_result returns extracted response for non-timeout."""
    coord = MultiAgentCoordinator()
    fake = AgentResponse(content="ok")
    monkeypatch.setattr(coord, "_extract_response", lambda *_a, **_k: fake)
    response, timed_out, used_fallback = coord._handle_workflow_result(
        {"messages": []}, "q", None, 0.0, 0.0
    )
    assert response is fake
    assert timed_out is False
    assert used_fallback is False


def test_annotate_span_sets_attributes() -> None:
    """_annotate_span records final status attributes on span."""
    coord = MultiAgentCoordinator()
    recorded: dict[str, object] = {}

    class _Span:
        def set_attribute(self, key: str, value: object) -> None:
            recorded[key] = value

    coord._annotate_span(
        _Span(), workflow_timed_out=False, used_fallback=False, processing_time=0.5
    )
    assert recorded["coordinator.workflow_timeout"] is False
    assert recorded["coordinator.success"] is True
    assert "coordinator.processing_time_ms" in recorded


def test_update_metrics_after_response_increments_success() -> None:
    """_update_metrics_after_response increments success count on non-timeouts."""
    coord = MultiAgentCoordinator()
    coord.successful_queries = 0
    coord._update_metrics_after_response(
        workflow_timed_out=False,
        used_fallback=False,
        processing_time=0.1,
        coordination_time=0.02,
    )
    assert coord.successful_queries == 1


def test_get_state_values_and_list_checkpoints_use_compiled_graph(monkeypatch) -> None:
    """Test get_state_values and list_checkpoints methods use compiled_graph."""
    coord = MultiAgentCoordinator()
    monkeypatch.setattr(coord, "_ensure_setup", lambda: True)

    compiled = Mock()
    compiled.get_state.return_value = SimpleNamespace(values={"messages": ["x"]})
    compiled.get_state_history.return_value = iter(
        [
            SimpleNamespace(
                config={"configurable": {"checkpoint_id": "c1", "checkpoint_ns": "ns"}}
            ),
            SimpleNamespace(config={"configurable": {"checkpoint_id": "c0"}}),
        ]
    )
    coord.compiled_graph = compiled

    values = coord.get_state_values(thread_id="t", user_id="u")
    assert values == {"messages": ["x"]}

    cps = coord.list_checkpoints(thread_id="t", user_id="u", limit=2)
    assert cps[0]["checkpoint_id"] == "c1"
    assert cps[0]["checkpoint_ns"] == "ns"
    assert cps[1]["checkpoint_id"] == "c0"

    status = coord.validate_system_status()
    assert status["model_configured"] is True
