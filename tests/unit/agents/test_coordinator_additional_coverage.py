"""Additional coverage-focused tests for MultiAgentCoordinator.

Note: This test file intentionally accesses protected members of MultiAgentCoordinator
to test internal behavior and state transitions.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import pytest
from langgraph.store.memory import InMemoryStore
from opentelemetry.trace import Span

from src.agents.coordinator import (
    MultiAgentCoordinator,
    _as_state_dict,
    _AsyncGraphRunner,
    _require_deadline_ts,
)
from src.agents.models import AgentResponse
from src.agents.tools.memory import (
    MemoryCandidate,
    capture_memory_namespace_generations,
    is_memory_namespace_tombstoned,
    memory_namespace_generation,
    memory_namespace_lock,
)
from src.persistence.checkpoint_identity import (
    checkpoint_thread_id,
    memory_namespace,
)

pytestmark = pytest.mark.unit


def test_admitted_turn_memory_mutations_are_fenced_across_scope_purges(  # noqa: PLR0915
    monkeypatch,
) -> None:
    """A paused turn cannot mutate either namespace after manual purges."""
    from src.agents.tools import memory as memory_tools

    user_id = "generation-fence-user"
    thread_id = "generation-fence-thread"
    session_namespace = memory_namespace(user_id=user_id, thread_id=thread_id)
    user_namespace = memory_namespace(user_id=user_id)
    existing_user_memory = "existing-user-memory"
    store = InMemoryStore()
    store.put(
        user_namespace,
        existing_user_memory,
        {"content": "keep me", "kind": "fact", "origin": "explicit"},
    )
    entered_graph = threading.Event()
    release_graph = threading.Event()
    payloads: list[tuple[dict[str, Any], dict[str, Any]]] = []
    events: list[dict[str, object]] = []

    class _PausedGraph:
        def copy(self, _config):  # type: ignore[no-untyped-def]
            return self

        async def astream(  # type: ignore[no-untyped-def]
            self, initial_state, *, config, **_kwargs
        ):
            entered_graph.set()
            while not release_graph.is_set():
                await asyncio.sleep(0.005)
            runtime = SimpleNamespace(store=store, config=config)
            payloads.append(
                (
                    json.loads(
                        memory_tools.remember.func(  # type: ignore[attr-defined]
                            "remember after purge",
                            scope="session",
                            state=initial_state,
                            runtime=runtime,
                        )
                    ),
                    json.loads(
                        memory_tools.forget_memory.func(  # type: ignore[attr-defined]
                            existing_user_memory,
                            scope="user",
                            state=initial_state,
                            runtime=runtime,
                        )
                    ),
                )
            )
            yield initial_state

        async def aget_state(self, _config):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                config={"configurable": {"checkpoint_id": "terminal"}}
            )

    monkeypatch.setattr(memory_tools, "log_jsonl", events.append)
    coordinator = MultiAgentCoordinator(store=store, max_agent_timeout=2.0)
    coordinator.compiled_graph = cast(Any, _PausedGraph())
    first_result: dict[str, Any] = {}
    first_error: list[BaseException] = []

    def _run_first_turn() -> None:
        try:
            first_result.update(
                coordinator._run_agent_workflow(
                    coordinator._build_initial_state("first", time.perf_counter()),
                    thread_id=thread_id,
                    user_id=user_id,
                    checkpoint_id=None,
                    runtime_context=None,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            first_error.append(exc)

    worker = threading.Thread(target=_run_first_turn)
    try:
        worker.start()
        assert entered_graph.wait(timeout=1.0)
        memory_tools.advance_memory_namespace_generation(session_namespace)
        memory_tools.advance_memory_namespace_generation(user_namespace)
        release_graph.set()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        assert not first_error
        assert first_result
        assert payloads[0] == (
            {"ok": False, "error": "memory mutation invalidated"},
            {"ok": False, "error": "memory mutation invalidated"},
        )
        stale_events = json.dumps(events[:2])
        assert user_id not in stale_events
        assert thread_id not in stale_events
        assert store.search(session_namespace) == []
        assert store.get(user_namespace, existing_user_memory) is not None

        coordinator._run_agent_workflow(
            coordinator._build_initial_state("second", time.perf_counter()),
            thread_id=thread_id,
            user_id=user_id,
            checkpoint_id=None,
            runtime_context=None,
        )
        assert payloads[1][0]["ok"] is True
        assert payloads[1][1] == {"ok": True}
        assert len(store.search(session_namespace)) == 1
        assert store.get(user_namespace, existing_user_memory) is None
    finally:
        release_graph.set()
        worker.join(timeout=2.0)
        coordinator.close()


def test_hard_purge_fence_blocks_inflight_and_future_consolidation(
    monkeypatch,
) -> None:
    """Extraction crossing a purge boundary cannot resurrect the session."""
    from src.agents import coordinator as coordinator_module

    extraction_started = threading.Event()
    release_extraction = threading.Event()
    applied = threading.Event()

    def _extract(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        extraction_started.set()
        assert release_extraction.wait(timeout=2)
        return [
            MemoryCandidate(
                content="stale memory",
                kind="fact",
                importance=0.8,
                source_checkpoint_id="checkpoint-1",
            )
        ]

    monkeypatch.setattr(coordinator_module, "extract_memory_candidates", _extract)

    def _apply_if_current(*args, expected_generation, **_kwargs):  # type: ignore[no-untyped-def]
        namespace = args[2]
        if (
            not is_memory_namespace_tombstoned(namespace)
            and memory_namespace_generation(namespace) == expected_generation
        ):
            applied.set()

    monkeypatch.setattr(
        coordinator_module,
        "consolidate_and_apply_memory_candidates",
        _apply_if_current,
    )

    coordinator = MultiAgentCoordinator(store=object())
    coordinator.llm = cast(Any, object())
    thread_id = "fence-thread"
    user_id = "fence-user"
    admitted_generations = capture_memory_namespace_generations(
        user_id=user_id,
        thread_id=thread_id,
    )
    generation = admitted_generations["session"]
    worker = threading.Thread(
        target=coordinator._consolidate_memories,
        args=({}, thread_id, user_id, generation, "checkpoint-1"),
    )
    try:
        worker.start()
        assert extraction_started.wait(timeout=1)
        monkeypatch.setattr(coordinator_module, "delete_persisted_session", Mock())
        assert coordinator.purge_session(
            conn=cast(Any, object()), thread_id=thread_id, user_id=user_id
        )
        release_extraction.set()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert not applied.is_set()

        submit = Mock()
        coordinator._memory_executor.submit = submit
        coordinator._schedule_memory_consolidation(
            {
                "_terminal_checkpoint_id": "checkpoint-2",
                "memory_generations": admitted_generations,
            },
            thread_id=thread_id,
            user_id=user_id,
        )
        submit.assert_not_called()
    finally:
        release_extraction.set()
        worker.join(timeout=2)
        coordinator.close()


def test_purge_session_drains_and_permanently_blocks_thread(monkeypatch) -> None:
    """Durable purge waits for the matching run and rejects future runs."""
    coordinator = MultiAgentCoordinator()
    persistence_id = checkpoint_thread_id(
        thread_id="public-thread",
        user_id="local-user",
    )
    active = coordinator._begin_active_run(persistence_id)
    assert active is not None
    monkeypatch.setattr(
        "src.agents.coordinator.delete_persisted_session",
        Mock(),
    )

    def _finish_when_drained() -> None:
        deadline = time.monotonic() + 1.0
        while active.control.drain_reason is None and time.monotonic() < deadline:
            time.sleep(0.001)
        coordinator._finish_active_run(persistence_id, active)

    finisher = threading.Thread(target=_finish_when_drained)
    try:
        finisher.start()
        assert coordinator.purge_session(
            conn=cast(Any, object()),
            thread_id="public-thread",
            user_id="local-user",
            timeout_s=1.0,
        )
        finisher.join(timeout=1)
        assert active.control.drain_reason == "session_purged"
        assert not finisher.is_alive()
        assert coordinator._begin_active_run(persistence_id) is None
    finally:
        coordinator._finish_active_run(persistence_id, active)
        finisher.join(timeout=1)
        coordinator.close()


def test_purge_session_defers_delete_when_run_does_not_exit(monkeypatch) -> None:
    coordinator = MultiAgentCoordinator()
    persistence_id = checkpoint_thread_id(
        thread_id="blocked-thread",
        user_id="local-user",
    )
    active = coordinator._begin_active_run(persistence_id)
    assert active is not None
    delete = Mock()
    monkeypatch.setattr(
        "src.agents.coordinator.delete_persisted_session",
        delete,
    )
    try:
        assert not coordinator.purge_session(
            conn=cast(Any, object()),
            thread_id="blocked-thread",
            user_id="local-user",
            timeout_s=0.01,
        )
        assert active.control.drain_reason == "session_purged"
        delete.assert_not_called()
    finally:
        coordinator._finish_active_run(persistence_id, active)
        coordinator.close()


def test_purge_session_returns_false_for_closed_sqlite_connection() -> None:
    """Invalid SQLite lifecycle state cannot escape the purge bool contract."""
    coordinator = MultiAgentCoordinator()
    conn = sqlite3.connect(":memory:")
    conn.close()
    try:
        assert not coordinator.purge_session(
            conn=conn,
            thread_id="closed-connection-thread",
            user_id="closed-connection-user",
            timeout_s=1.0,
        )
    finally:
        coordinator.close()


def test_purge_budget_bounds_namespace_fence_acquisition(monkeypatch) -> None:
    """A busy memory namespace returns deferred within the total purge budget."""
    coordinator = MultiAgentCoordinator()
    thread_id = "busy-namespace-thread"
    user_id = "busy-namespace-user"
    namespace = memory_namespace(user_id=user_id, thread_id=thread_id)
    deleted = Mock()
    monkeypatch.setattr(
        "src.agents.coordinator.delete_persisted_session",
        deleted,
    )
    result: list[bool] = []
    worker = threading.Thread(
        target=lambda: result.append(
            coordinator.purge_session(
                conn=cast(Any, object()),
                thread_id=thread_id,
                user_id=user_id,
                timeout_s=0.01,
            )
        )
    )
    try:
        with memory_namespace_lock(namespace):
            worker.start()
            worker.join(timeout=0.2)
            assert not worker.is_alive()
        assert result == [False]
        deleted.assert_not_called()
    finally:
        worker.join(timeout=1.0)
        coordinator.close()


def test_fork_rejects_active_identity_but_allows_independent_identities() -> None:
    """Forks share the graph-run fence only for the exact persisted identity."""
    graph = SimpleNamespace(
        get_state=Mock(return_value=SimpleNamespace(values={"messages": []})),
        update_state=Mock(
            return_value={"configurable": {"checkpoint_id": "fork-head"}}
        ),
    )
    coordinator = MultiAgentCoordinator()
    coordinator.compiled_graph = cast(Any, graph)
    coordinator._setup_complete = True
    thread_id = "active-fork-thread"
    user_id = "active-fork-user"
    persistence_id = checkpoint_thread_id(thread_id=thread_id, user_id=user_id)
    active = coordinator._begin_active_run(persistence_id)
    assert active is not None
    try:
        assert (
            coordinator.fork_from_checkpoint(
                thread_id=thread_id,
                user_id=user_id,
                checkpoint_id="source",
            )
            is None
        )
        graph.get_state.assert_not_called()

        for independent_thread, independent_user in (
            (thread_id, "independent-user"),
            ("independent-thread", user_id),
        ):
            assert (
                coordinator.fork_from_checkpoint(
                    thread_id=independent_thread,
                    user_id=independent_user,
                    checkpoint_id="source",
                )
                == "fork-head"
            )
        assert graph.update_state.call_count == 2
    finally:
        coordinator._finish_active_run(persistence_id, active)
        coordinator.close()


def test_inflight_fork_fences_new_turn_until_branch_write_finishes() -> None:
    """A fork reserves the exact run identity across its read/write window."""
    fork_started = threading.Event()
    allow_fork = threading.Event()
    fork_result: list[str | None] = []

    class _Graph:
        def get_state(self, _config):  # type: ignore[no-untyped-def]
            fork_started.set()
            assert allow_fork.wait(timeout=1.0)
            return SimpleNamespace(values={"messages": []})

        def update_state(self, _config, _values, *, as_node):  # type: ignore[no-untyped-def]
            assert as_node == "__copy__"
            return {"configurable": {"checkpoint_id": "fork-head"}}

    coordinator = MultiAgentCoordinator()
    coordinator.compiled_graph = _Graph()
    coordinator._setup_complete = True
    forker = threading.Thread(
        target=lambda: fork_result.append(
            coordinator.fork_from_checkpoint(
                thread_id="fork-first-thread",
                user_id="fork-first-user",
                checkpoint_id="source",
            )
        )
    )
    try:
        forker.start()
        assert fork_started.wait(timeout=1.0)

        turn_result = coordinator._run_agent_workflow(
            {
                "total_start_time": 0.0,
                "deadline_ts": time.monotonic() + 1.0,
            },
            thread_id="fork-first-thread",
            user_id="fork-first-user",
            checkpoint_id=None,
            runtime_context=None,
        )

        assert turn_result["workflow_stopped"] is True
        assert turn_result["cancel_reason"] == "previous_run_active"
        allow_fork.set()
        forker.join(timeout=1.0)
        assert fork_result == ["fork-head"]
        assert coordinator._active_runs == {}
    finally:
        allow_fork.set()
        forker.join(timeout=1.0)
        coordinator.close()


def test_purge_serializes_with_inflight_fork_and_deletes_its_new_head(
    monkeypatch,
) -> None:
    """A fork already inside the mutation lock completes before purge deletes it."""
    fork_started = threading.Event()
    allow_fork = threading.Event()
    events: list[str] = []
    fork_result: list[str | None] = []
    purge_result: list[bool] = []

    class _Graph:
        def get_state(self, _config):  # type: ignore[no-untyped-def]
            fork_started.set()
            assert allow_fork.wait(timeout=1.0)
            return SimpleNamespace(values={"messages": []})

        def update_state(self, _config, _values, *, as_node):  # type: ignore[no-untyped-def]
            assert as_node == "__copy__"
            events.append("fork-write")
            return {"configurable": {"checkpoint_id": "fork-head"}}

    coordinator = MultiAgentCoordinator()
    coordinator.compiled_graph = _Graph()
    coordinator._setup_complete = True
    monkeypatch.setattr(
        "src.agents.coordinator.delete_persisted_session",
        lambda *_a, **_k: events.append("purge-delete"),
    )
    forker = threading.Thread(
        target=lambda: fork_result.append(
            coordinator.fork_from_checkpoint(
                thread_id="race-thread",
                user_id="race-user",
                checkpoint_id="source",
            )
        )
    )
    purger = threading.Thread(
        target=lambda: purge_result.append(
            coordinator.purge_session(
                conn=cast(Any, object()),
                thread_id="race-thread",
                user_id="race-user",
            )
        )
    )
    try:
        forker.start()
        assert fork_started.wait(timeout=1.0)
        purger.start()
        allow_fork.set()
        forker.join(timeout=1.0)
        purger.join(timeout=1.0)

        assert not forker.is_alive()
        assert not purger.is_alive()
        assert fork_result == ["fork-head"]
        assert purge_result == [True]
        assert events == ["fork-write", "purge-delete"]
        assert (
            coordinator.fork_from_checkpoint(
                thread_id="race-thread",
                user_id="race-user",
                checkpoint_id="source",
            )
            is None
        )
    finally:
        allow_fork.set()
        forker.join(timeout=1.0)
        purger.join(timeout=1.0)
        coordinator.close()


def test_memory_submit_failure_releases_capacity(monkeypatch) -> None:
    coordinator = MultiAgentCoordinator(store=object())
    thread_id = "submit-failure-thread"
    user_id = "submit-failure-user"
    monkeypatch.setattr(
        coordinator._memory_executor,
        "submit",
        Mock(side_effect=RuntimeError("executor closed")),
    )
    try:
        coordinator._schedule_memory_consolidation(
            {
                "_terminal_checkpoint_id": "checkpoint-submit",
                "memory_generations": capture_memory_namespace_generations(
                    user_id=user_id,
                    thread_id=thread_id,
                ),
            },
            thread_id=thread_id,
            user_id=user_id,
        )
        assert coordinator._memory_consolidation_semaphore.acquire(blocking=False)
        assert coordinator._memory_consolidation_semaphore.acquire(blocking=False)
        assert not coordinator._memory_consolidation_semaphore.acquire(blocking=False)
        coordinator._memory_consolidation_semaphore.release()
        coordinator._memory_consolidation_semaphore.release()
    finally:
        coordinator.close()


def test_memory_consolidation_requires_immutable_checkpoint_provenance() -> None:
    coordinator = MultiAgentCoordinator(store=object())
    submit = Mock()
    coordinator._memory_executor.submit = submit
    try:
        coordinator._schedule_memory_consolidation(
            {
                "memory_generations": capture_memory_namespace_generations(
                    user_id="missing-provenance-user",
                    thread_id="missing-provenance-thread",
                )
            },
            thread_id="missing-provenance-thread",
            user_id="missing-provenance-user",
        )
        coordinator._schedule_memory_consolidation(
            {"_terminal_checkpoint_id": "checkpoint-without-generation"},
            thread_id="missing-generation-thread",
            user_id="missing-generation-user",
        )
        submit.assert_not_called()
    finally:
        coordinator.close()


def test_memory_consolidation_rejects_prepurge_generation_before_schedule() -> None:
    """A completed pre-purge turn cannot schedule work using a fresh fence."""
    from src.agents.tools import memory as memory_tools

    thread_id = "purge-before-schedule-thread"
    user_id = "purge-before-schedule-user"
    namespace = memory_namespace(user_id=user_id, thread_id=thread_id)
    admitted_generations = capture_memory_namespace_generations(
        user_id=user_id,
        thread_id=thread_id,
    )
    coordinator = MultiAgentCoordinator(store=object())
    submit = Mock()
    coordinator._memory_executor.submit = submit
    try:
        memory_tools.advance_memory_namespace_generation(namespace)
        coordinator._schedule_memory_consolidation(
            {
                "_terminal_checkpoint_id": "prepurge-checkpoint",
                "memory_generations": admitted_generations,
            },
            thread_id=thread_id,
            user_id=user_id,
        )
        submit.assert_not_called()
    finally:
        coordinator.close()


def test_memory_timeout_holds_capacity_until_worker_exits(monkeypatch) -> None:
    from src.agents import coordinator as coordinator_module

    worker_started = threading.Event()
    release_worker = threading.Event()

    def _block(*_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
        worker_started.set()
        release_worker.wait(timeout=2)

    coordinator = MultiAgentCoordinator(store=object())
    monkeypatch.setattr(coordinator, "_consolidate_memories", _block)
    monkeypatch.setattr(coordinator_module, "MEMORY_CONSOLIDATION_TIMEOUT_S", 0.01)
    try:
        thread_id = "timeout-thread"
        user_id = "timeout-user"
        coordinator._schedule_memory_consolidation(
            {
                "_terminal_checkpoint_id": "checkpoint-timeout",
                "memory_generations": capture_memory_namespace_generations(
                    user_id=user_id,
                    thread_id=thread_id,
                ),
            },
            thread_id=thread_id,
            user_id=user_id,
        )
        assert worker_started.wait(timeout=1)
        time.sleep(0.05)

        assert coordinator._memory_consolidation_semaphore.acquire(blocking=False)
        assert not coordinator._memory_consolidation_semaphore.acquire(blocking=False)
        coordinator._memory_consolidation_semaphore.release()

        release_worker.set()
        assert coordinator._memory_consolidation_semaphore.acquire(timeout=1)
        assert coordinator._memory_consolidation_semaphore.acquire(timeout=1)
        coordinator._memory_consolidation_semaphore.release()
        coordinator._memory_consolidation_semaphore.release()
    finally:
        release_worker.set()
        coordinator.close()


def test_stuck_memory_worker_does_not_hold_interpreter_shutdown() -> None:
    """A provider call that never returns cannot block Python process exit."""
    script = """
import threading
from src.agents.coordinator import MultiAgentCoordinator

started = threading.Event()
blocked = threading.Event()

def never_returns():
    started.set()
    blocked.wait()

coordinator = MultiAgentCoordinator(store=object())
coordinator._memory_executor.submit(never_returns)
assert started.wait(2.0)
coordinator.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=8.0,
    )
    assert completed.returncode == 0, completed.stderr


@pytest.fixture
def workflow_result_fixture() -> dict[str, Any]:
    """Reusable fixture for coordinator state/result shape."""
    return {
        "messages": [],
        "total_start_time": 0.0,
        "deadline_ts": time.monotonic() + 1.0,
        "workflow_stopped": False,
        "timed_out": False,
        "deadline_s": 0.0,
        # Add other common fields as needed
    }


def test_run_agent_workflow_uses_per_run_async_graph_copy(
    workflow_result_fixture,
) -> None:
    """Each run gets an isolated step timeout and supports model_dump state."""
    coord = MultiAgentCoordinator(max_agent_timeout=1.0)

    class _State:
        """Test mock implementing state protocol."""

        def model_dump(self) -> dict:  # type: ignore[no-untyped-def]
            return workflow_result_fixture.copy()

    class _Compiled:
        update: dict[str, object] | None = None

        def copy(self, update: dict[str, object]):
            self.update = update
            return self

        async def astream(self, *_args, **_kwargs):
            yield _State()

    compiled = _Compiled()
    coord.compiled_graph = compiled
    try:
        out = coord._run_agent_workflow(
            {
                "total_start_time": 0.0,
                "deadline_ts": time.monotonic() + 1.0,
            },
            thread_id="t",
            user_id="u",
            checkpoint_id=None,
            runtime_context=None,
        )
        assert out.get("timed_out") is False
        assert compiled.update is not None
        step_timeout = compiled.update["step_timeout"]
        assert isinstance(step_timeout, float)
        assert 0 < step_timeout <= 1.0
    finally:
        coord.close()


def test_run_agent_workflow_classifies_empty_stream_as_stopped(
    workflow_result_fixture,
) -> None:
    """An empty graph stream is a stopped run rather than a false success."""
    coord = MultiAgentCoordinator(max_agent_timeout=1.0)

    class _Compiled:
        def copy(self, _update: dict[str, object]):
            return self

        async def astream(self, *_args, **_kwargs):
            if False:
                yield None

    compiled = _Compiled()
    coord.compiled_graph = compiled
    initial = workflow_result_fixture.copy()
    try:
        out = coord._run_agent_workflow(
            initial,
            thread_id="t",
            user_id="u",
            checkpoint_id=None,
            runtime_context=None,
        )
        assert out["workflow_stopped"] is True
        assert out["timed_out"] is False
        assert out["cancel_reason"] == "workflow_no_result"
    finally:
        coord.close()


def test_stream_exit_at_deadline_is_classified_as_timeout(monkeypatch) -> None:
    """A clean stream exit at the deadline cannot return a partial state."""
    clock = SimpleNamespace(now=100.0)
    real_perf_counter = time.perf_counter
    fake_time = SimpleNamespace(
        monotonic=lambda: clock.now,
        perf_counter=real_perf_counter,
    )
    monkeypatch.setattr("src.agents.coordinator.time", fake_time)

    class _Compiled:
        async def astream(self, *_args, **_kwargs):
            yield {"total_start_time": 0.0, "deadline_ts": 101.0}
            clock.now = 101.0

    coord = MultiAgentCoordinator(max_agent_timeout=1.0)
    active_run = coord._begin_active_run("deadline-exit")
    assert active_run is not None

    try:
        result = asyncio.run(
            coord._consume_agent_workflow(
                _Compiled(),
                {"total_start_time": 0.0, "deadline_ts": 101.0},
                config={"configurable": {"thread_id": "deadline-exit"}},
                runtime_context=None,
                deadline_ts=101.0,
                persistence_id="deadline-exit",
                public_thread_id="deadline-exit",
                active_run=active_run,
            )
        )
    finally:
        coord.close()

    assert result["workflow_stopped"] is True
    assert result["timed_out"] is True
    assert result["cancel_reason"] == "deadline_exceeded"


def test_process_query_timeout_returns_timeout_response(monkeypatch) -> None:
    """Test process_query returns the canonical timeout response."""
    coord = MultiAgentCoordinator()
    monkeypatch.setattr(coord, "_ensure_setup", lambda: True)
    monkeypatch.setattr(
        coord,
        "_build_initial_state",
        lambda _q, start_time: {
            "messages": [],
            "total_start_time": start_time,
        },
    )
    monkeypatch.setattr(
        coord,
        "_run_agent_workflow",
        lambda *_a, **_k: {
            "workflow_stopped": True,
            "timed_out": True,
            "cancel_reason": "deadline_exceeded",
        },
    )
    monkeypatch.setattr(coord, "_start_span", lambda *_a, **_k: Mock())
    monkeypatch.setattr(coord, "_record_query_metrics", lambda *_a, **_k: None)

    resp = coord.process_query("q", thread_id="t", user_id="u")
    assert isinstance(resp, AgentResponse)
    assert resp.metadata.get("reason") == "timeout"
    assert resp.optimization_metrics.get("timeout") is True
    assert resp.validation_score == 0.0


def test_process_query_non_timeout_stop_skips_success_side_effects(monkeypatch) -> None:
    """Capacity/fence stops skip memory consolidation and success metrics."""
    coord = MultiAgentCoordinator()
    memory_store = Mock()
    record_metrics = Mock()
    span = Mock()
    monkeypatch.setattr(coord, "_ensure_setup", lambda: True)
    monkeypatch.setattr(coord, "_start_span", lambda *_a, **_k: span)
    monkeypatch.setattr(
        coord,
        "_run_agent_workflow",
        lambda *_a, **_k: {
            "workflow_stopped": True,
            "timed_out": False,
            "cancel_reason": "runner_saturated",
        },
    )
    monkeypatch.setattr(coord, "_schedule_memory_consolidation", memory_store)
    monkeypatch.setattr(coord, "_record_query_metrics", record_metrics)

    response = coord.process_query("q", thread_id="t", user_id="u")

    assert response.metadata["reason"] == "workflow_stopped"
    assert response.metadata["cancel_reason"] == "runner_saturated"
    assert response.optimization_metrics["stopped"] is True
    memory_store.assert_not_called()
    assert record_metrics.call_count == 1
    assert record_metrics.call_args.args[1] is False
    assert any(
        call.args == ("coordinator.success", False)
        for call in span.set_attribute.call_args_list
    )
    coord.close()


def test_ensure_setup_derives_provider_cap_from_coordinator_override() -> None:
    """Coordinator setup passes its owned deadline to the provider factory."""
    coordinator = MultiAgentCoordinator(max_agent_timeout=3.5)
    chat_model = Mock()

    with (
        patch("src.config.setup_llamaindex"),
        patch("llama_index.core.Settings") as llamaindex_settings,
        patch(
            "src.agents.coordinator.build_chat_model",
            return_value=chat_model,
        ) as build_model,
        patch.object(
            coordinator,
            "_build_agent_graph_components",
            return_value=({}, object(), object()),
        ),
    ):
        llamaindex_settings._llm = Mock()

        assert coordinator._ensure_setup() is True

    assert build_model.call_args.kwargs["timeout_cap"] == 3.5
    coordinator.close()


def test_concurrent_first_use_builds_coordinator_once() -> None:
    """Concurrent callers share one serialized lazy setup publication."""
    coordinator = MultiAgentCoordinator()
    model_build_started = threading.Event()
    allow_model_build = threading.Event()
    second_lock_attempt = threading.Event()
    setup_results: list[bool] = []
    real_setup_lock = threading.Lock()
    lock_attempts = 0
    lock_attempts_guard = threading.Lock()

    class _ObservedLock:
        def __enter__(self):  # type: ignore[no-untyped-def]
            nonlocal lock_attempts
            with lock_attempts_guard:
                lock_attempts += 1
                if lock_attempts == 2:
                    second_lock_attempt.set()
            real_setup_lock.acquire()
            return self

        def __exit__(self, *_args):  # type: ignore[no-untyped-def]
            real_setup_lock.release()

    def build_model(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        model_build_started.set()
        assert allow_model_build.wait(timeout=1.0)
        return object()

    coordinator._setup_lock = cast(Any, _ObservedLock())
    with (
        patch("src.config.setup_llamaindex") as setup_llamaindex,
        patch("llama_index.core.Settings") as llamaindex_settings,
        patch("src.agents.coordinator.build_chat_model", side_effect=build_model),
        patch.object(
            coordinator,
            "_build_agent_graph_components",
            return_value=({}, object(), object()),
        ) as build_graph,
    ):
        llamaindex_settings._llm = Mock()
        first = threading.Thread(
            target=lambda: setup_results.append(coordinator._ensure_setup())
        )
        second = threading.Thread(
            target=lambda: setup_results.append(coordinator._ensure_setup())
        )
        try:
            first.start()
            assert model_build_started.wait(timeout=1.0)
            second.start()
            assert second_lock_attempt.wait(timeout=1.0)
            allow_model_build.set()
            first.join(timeout=1.0)
            second.join(timeout=1.0)
        finally:
            allow_model_build.set()
            first.join(timeout=1.0)
            second.join(timeout=1.0)
            coordinator.close()

    assert setup_results == [True, True]
    setup_llamaindex.assert_called_once()
    build_graph.assert_called_once()


def test_close_during_setup_rejects_late_component_publication() -> None:
    """Close stays bounded and a paused setup cannot reopen coordinator state."""
    coordinator = MultiAgentCoordinator()
    graph_build_started = threading.Event()
    allow_graph_build = threading.Event()
    close_finished = threading.Event()
    setup_results: list[bool] = []

    def build_components(**_kwargs):  # type: ignore[no-untyped-def]
        graph_build_started.set()
        assert allow_graph_build.wait(timeout=1.0)
        return {"agent": object()}, object(), object()

    with (
        patch("src.config.setup_llamaindex"),
        patch("llama_index.core.Settings") as llamaindex_settings,
        patch("src.agents.coordinator.build_chat_model", return_value=object()),
        patch.object(
            coordinator,
            "_build_agent_graph_components",
            side_effect=build_components,
        ),
    ):
        llamaindex_settings._llm = Mock()
        setup_thread = threading.Thread(
            target=lambda: setup_results.append(coordinator._ensure_setup())
        )
        closer = threading.Thread(
            target=lambda: (coordinator.close(), close_finished.set())
        )
        try:
            setup_thread.start()
            assert graph_build_started.wait(timeout=1.0)
            closer.start()
            assert close_finished.wait(timeout=0.5)
            assert setup_thread.is_alive()
            allow_graph_build.set()
            setup_thread.join(timeout=1.0)
            closer.join(timeout=1.0)
        finally:
            allow_graph_build.set()
            setup_thread.join(timeout=1.0)
            closer.join(timeout=1.0)
            coordinator.close()

    assert setup_results == [False]
    assert coordinator._setup_complete is False
    assert coordinator.llm is None
    assert coordinator.agents == {}
    assert coordinator.graph is None
    assert coordinator.compiled_graph is None
    assert coordinator._graph_runner is None
    assert coordinator._ensure_setup() is False


def test_close_during_owned_checkpointer_open_cleans_unpublished_runner(
    tmp_path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    """A saver opened after close is cleaned locally and never published."""
    open_started = threading.Event()
    allow_open = threading.Event()
    stack_closed = threading.Event()
    initialization_errors: list[RuntimeError] = []
    coordinator = MultiAgentCoordinator(checkpointer_path=tmp_path / "chat.db")

    async def close_stack() -> None:
        stack_closed.set()

    async def open_checkpointer(_path):  # type: ignore[no-untyped-def]
        open_started.set()
        while not allow_open.is_set():
            await asyncio.sleep(0.005)
        return object(), SimpleNamespace(aclose=close_stack)

    def initialize() -> None:
        try:
            coordinator._ensure_checkpointer()
        except RuntimeError as exc:
            initialization_errors.append(exc)

    monkeypatch.setattr(
        "src.agents.coordinator._open_async_sqlite_checkpointer",
        open_checkpointer,
    )
    initializer = threading.Thread(target=initialize)
    try:
        initializer.start()
        assert open_started.wait(timeout=1.0)
        coordinator.close()
        allow_open.set()
        initializer.join(timeout=1.0)
    finally:
        allow_open.set()
        initializer.join(timeout=1.0)
        coordinator.close()

    assert len(initialization_errors) == 1
    assert stack_closed.wait(timeout=0.5)
    assert coordinator.checkpointer is None
    assert coordinator._checkpointer_stack is None
    assert coordinator._graph_runner is None


def test_same_thread_timeout_is_fenced_until_async_cleanup() -> None:
    """A timed-out run retains its thread fence until wrapper cleanup finishes."""
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()

    class _Compiled:
        def copy(self, _update: dict[str, object]):
            return self

        async def astream(self, *_args, **_kwargs):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                cleanup_started.set()
                while not allow_cleanup.is_set():
                    try:
                        await asyncio.sleep(0.005)
                    except asyncio.CancelledError:
                        continue
                return
            yield {"done": True}

    coord = MultiAgentCoordinator(max_agent_timeout=0.02)
    coord.compiled_graph = _Compiled()
    try:
        first = coord._run_agent_workflow(
            {
                "total_start_time": 0.0,
                "deadline_ts": time.monotonic() + 0.02,
            },
            thread_id="same",
            user_id="u",
            checkpoint_id=None,
            runtime_context=None,
        )
        assert cleanup_started.wait(timeout=0.5)
        active_run = next(iter(coord._active_runs.values()))
        second = coord._run_agent_workflow(
            {
                "total_start_time": 0.0,
                "deadline_ts": time.monotonic() + 0.02,
            },
            thread_id="same",
            user_id="u",
            checkpoint_id=None,
            runtime_context=None,
        )

        assert first["cancel_reason"] == "deadline_exceeded"
        assert second["cancel_reason"] == "previous_run_active"
        assert second["workflow_stopped"] is True
        assert second["timed_out"] is False
        allow_cleanup.set()
        assert active_run.finished.wait(timeout=0.5)
    finally:
        allow_cleanup.set()
        coord.close()


def test_graph_runner_close_stops_owned_event_loop() -> None:
    """Coordinator close terminates its lazily owned graph event-loop thread."""
    coord = MultiAgentCoordinator()
    runner = coord._get_graph_runner()

    assert runner.is_alive
    coord.close()
    assert not runner.is_alive
    coord.close()


def test_close_does_not_wait_for_stuck_persistence_reader() -> None:
    """Shutdown remains bounded when a sync persistence bridge owns its lock."""
    coord = MultiAgentCoordinator()
    runner = coord._get_graph_runner()
    closed = threading.Event()
    coord._persistence_lock.acquire()
    closer = threading.Thread(target=lambda: (coord.close(), closed.set()))
    try:
        closer.start()
        assert closed.wait(timeout=0.5)
        assert not runner.is_alive
    finally:
        coord._persistence_lock.release()
        closer.join(timeout=1.0)
        coord.close()


def test_close_does_not_wait_for_checkpointer_initialization_lock() -> None:
    """Shutdown captures saver ownership without waiting on a stuck setup lock."""
    coord = MultiAgentCoordinator()
    runner = coord._get_graph_runner()
    closed = threading.Event()
    coord._checkpointer_lock.acquire()
    closer = threading.Thread(target=lambda: (coord.close(), closed.set()))
    try:
        closer.start()
        assert closed.wait(timeout=0.5)
        assert not runner.is_alive
    finally:
        coord._checkpointer_lock.release()
        closer.join(timeout=1.0)
        coord.close()


def test_close_does_not_wait_for_admitted_memory_mutation() -> None:
    """Shutdown skips generation invalidation when its namespace lock is busy."""
    coord = MultiAgentCoordinator(store=object())
    namespace = ("memories", "close-bounded", "session")
    with coord._memory_jobs_lock:
        coord._memory_jobs[namespace] = 1
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (coord.close(), closed.set()))

    with memory_namespace_lock(namespace):
        closer.start()
        assert closed.wait(timeout=0.5)

    closer.join(timeout=1.0)
    assert not closer.is_alive()
    coord.close()


def test_graph_runner_bounds_slow_async_cleanup(monkeypatch) -> None:
    """A stuck finalizer cannot keep the owned event-loop thread alive."""
    cleanup_started = threading.Event()
    allow_cleanup = threading.Event()
    cleanup_finished = threading.Event()
    runner = _AsyncGraphRunner()

    async def cleanup() -> None:
        cleanup_started.set()
        while not allow_cleanup.is_set():
            await asyncio.sleep(0.005)
        cleanup_finished.set()

    monkeypatch.setattr(
        "src.agents.coordinator.AGENT_GRAPH_RUNNER_CLOSE_GRACE_S", 0.001
    )

    runner.close(async_cleanup=cleanup)

    assert cleanup_started.wait(timeout=0.2)
    runner._thread.join(timeout=0.2)
    assert not runner.is_alive
    assert not cleanup_finished.is_set()
    allow_cleanup.set()


def test_graph_runner_cleans_up_despite_cancellation_resistant_task(
    monkeypatch,
) -> None:
    """A task suppressing cancellation cannot prevent saver cleanup or loop exit."""
    task_started = threading.Event()
    cleanup_started = threading.Event()
    runner = _AsyncGraphRunner()

    async def cancellation_resistant() -> dict[str, Any]:
        task_started.set()
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                continue

    async def cleanup() -> None:
        cleanup_started.set()

    monkeypatch.setattr("src.agents.coordinator.AGENT_GRAPH_RUNNER_CLOSE_GRACE_S", 0.01)
    future = runner.submit(cancellation_resistant())
    assert task_started.wait(timeout=0.2)

    runner.close(async_cleanup=cleanup)
    runner._thread.join(timeout=0.5)

    assert cleanup_started.is_set()
    assert not runner.is_alive
    future.cancel()


def test_non_deadline_stop_does_not_emit_deadline_event(monkeypatch) -> None:
    """Capacity and fencing stops do not masquerade as deadline overruns."""
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.agents.coordinator.log_jsonl", lambda event: events.append(dict(event))
    )
    coord = MultiAgentCoordinator(max_agent_timeout=1.0)

    result = coord._build_workflow_stopped_state(
        {"total_start_time": 0.0},
        cancel_reason="runner_saturated",
    )
    coord.close()

    assert result["workflow_stopped"] is True
    assert result["timed_out"] is False
    assert len(events) == 1
    assert events[0]["agent_workflow_stopped"] is True
    assert events[0]["cancel_reason"] == "runner_saturated"
    assert "agent_deadline_exceeded" not in events[0]


def test_early_dependency_timeout_does_not_claim_deadline_overrun(
    monkeypatch,
) -> None:
    """An early dependency timeout remains distinct from the caller deadline."""
    events: list[dict[str, object]] = []
    monkeypatch.setattr(
        "src.agents.coordinator.log_jsonl", lambda event: events.append(dict(event))
    )

    class _Compiled:
        def copy(self, _update: dict[str, object]):
            return self

        async def astream(self, *_args, **_kwargs):
            if False:
                yield {}
            raise TimeoutError("provider timeout")

    coord = MultiAgentCoordinator(max_agent_timeout=1.0)
    coord.compiled_graph = _Compiled()
    try:
        result = coord._run_agent_workflow(
            {
                "total_start_time": 0.0,
                "deadline_ts": time.monotonic() + 1.0,
            },
            thread_id="dependency-timeout",
            user_id="u",
            checkpoint_id=None,
            runtime_context=None,
        )
    finally:
        coord.close()

    assert result["cancel_reason"] == "dependency_timeout"
    assert result["workflow_stopped"] is True
    assert result["timed_out"] is True
    assert events[-1]["cancel_reason"] == "dependency_timeout"
    assert "agent_deadline_exceeded" not in events[-1]


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


def test_require_deadline_ts_rejects_missing_invalid_and_non_finite_values() -> None:
    """Every run requires one finite absolute monotonic deadline."""
    assert _require_deadline_ts({"deadline_ts": "12.5"}) == 12.5
    for value in (None, "bad", float("nan"), float("inf")):
        with pytest.raises(ValueError, match="deadline_ts"):
            _require_deadline_ts({"deadline_ts": value})


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


def test_handle_timeout_response() -> None:
    """_handle_timeout_response returns the canonical timeout response."""
    coord = MultiAgentCoordinator()
    response = coord._handle_timeout_response(
        start_time=0.0,
        cancel_reason="deadline_exceeded",
    )
    assert response.metadata.get("reason") == "timeout"
    assert response.metadata.get("cancel_reason") == "deadline_exceeded"
    assert response.optimization_metrics.get("timeout") is True


def test_handle_workflow_result_timeout_path(monkeypatch) -> None:
    """_handle_workflow_result routes timeout results to the timeout handler."""
    coord = MultiAgentCoordinator()
    monkeypatch.setattr(
        coord,
        "_handle_timeout_response",
        lambda *_a, **_k: AgentResponse(content="x"),
    )
    monkeypatch.setattr(coord, "_record_query_metrics", lambda *_a, **_k: None)
    response, stopped, timed_out = coord._handle_workflow_result(
        {
            "workflow_stopped": True,
            "timed_out": True,
            "cancel_reason": "dependency_timeout",
        },
        "q",
        0.0,
        0.0,
    )
    assert stopped is True
    assert timed_out is True
    assert isinstance(response, AgentResponse)


def test_handle_workflow_result_non_timeout(monkeypatch) -> None:
    """_handle_workflow_result returns extracted response for non-timeout."""
    coord = MultiAgentCoordinator()
    fake = AgentResponse(content="ok")
    monkeypatch.setattr(coord, "_extract_response", lambda *_a, **_k: fake)
    response, stopped, timed_out = coord._handle_workflow_result(
        {"messages": []}, "q", 0.0, 0.0
    )
    assert response is fake
    assert stopped is False
    assert timed_out is False


def test_annotate_span_sets_attributes() -> None:
    """_annotate_span records final status attributes on span."""
    coord = MultiAgentCoordinator()
    recorded: dict[str, object] = {}

    class _Span:
        def set_attribute(self, key: str, value: object) -> None:
            recorded[key] = value

    coord._annotate_span(
        cast(Span, _Span()),
        workflow_stopped=False,
        workflow_timed_out=False,
        processing_time=0.5,
    )
    assert recorded["coordinator.workflow_stopped"] is False
    assert recorded["coordinator.workflow_timeout"] is False
    assert recorded["coordinator.success"] is True
    assert "coordinator.processing_time_ms" in recorded


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

    state_config = compiled.get_state.call_args.args[0]["configurable"]
    history_config = compiled.get_state_history.call_args.args[0]["configurable"]
    assert state_config["thread_id"] == history_config["thread_id"]
    assert state_config["thread_id"] != "t"
    assert state_config["thread_id"].startswith("docmind:")
