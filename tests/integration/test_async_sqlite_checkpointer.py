"""Integration coverage for coordinator-owned async SQLite checkpoints."""

from __future__ import annotations

import asyncio
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import NotRequired, TypedDict, cast

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agents.coordinator import MultiAgentCoordinator
from src.persistence.chat_db import LegacyCheckpointIdentityError


class _CheckpointState(TypedDict):
    """Minimal state for persistence and concurrent-fence coverage."""

    total_start_time: float
    deadline_ts: float
    owner: str
    visits: NotRequired[int]
    workflow_stopped: NotRequired[bool]
    timed_out: NotRequired[bool]
    cancel_reason: NotRequired[str]


@pytest.mark.integration
def test_coordinator_owned_checkpointer_rejects_legacy_raw_thread_ids(
    tmp_path: Path,
) -> None:
    """Programmatic startup enforces the same v2 identity boundary as the UI."""
    db_path = tmp_path / "legacy-chat.db"

    async def seed_legacy_checkpoint() -> None:
        async with AsyncSqliteSaver.from_conn_string(str(db_path)) as saver:
            await saver.setup()
            await saver.conn.execute(
                """
                INSERT INTO checkpoints (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    parent_checkpoint_id,
                    type,
                    checkpoint,
                    metadata
                ) VALUES (?, '', ?, NULL, NULL, NULL, NULL)
                """,
                ("raw-v1-thread", "checkpoint-1"),
            )
            await saver.conn.commit()

    asyncio.run(seed_legacy_checkpoint())
    coordinator = MultiAgentCoordinator(checkpointer_path=db_path)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            coordinator._ensure_checkpointer()
        assert isinstance(exc_info.value.__cause__, LegacyCheckpointIdentityError)
    finally:
        coordinator.close()


@pytest.mark.integration
def test_async_sqlite_persists_isolated_users_and_closes_on_runner_loop(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real async graph runs persist per user and close their owned connection."""
    rendezvous: asyncio.Event | None = None
    entered = 0

    async def record_visit(state: _CheckpointState) -> dict[str, int]:
        nonlocal entered, rendezvous
        if rendezvous is None:
            rendezvous = asyncio.Event()
        entered += 1
        if entered == 2:
            rendezvous.set()
        await asyncio.wait_for(rendezvous.wait(), timeout=1.0)
        return {"visits": int(state.get("visits", 0)) + 1}

    builder = StateGraph(_CheckpointState)
    builder.add_node("record_visit", record_visit)
    builder.add_edge(START, "record_visit")
    builder.add_edge("record_visit", END)

    db_path = tmp_path / "chat.db"
    coordinator = MultiAgentCoordinator(
        max_agent_timeout=2.0,
        checkpointer_path=db_path,
    )
    coordinator._ensure_checkpointer()
    checkpointer = cast(AsyncSqliteSaver, coordinator.checkpointer)
    coordinator.compiled_graph = builder.compile(checkpointer=checkpointer)
    coordinator._setup_complete = True
    runner = coordinator._get_graph_runner()
    connection_thread = getattr(checkpointer.conn, "_thread", None)
    close_loops: list[asyncio.AbstractEventLoop] = []
    original_close = checkpointer.conn.close

    async def tracked_close() -> None:
        close_loops.append(asyncio.get_running_loop())
        await original_close()

    monkeypatch.setattr(checkpointer.conn, "close", tracked_close)

    async def read_connection_pragmas() -> tuple[str, int]:
        async with checkpointer.conn.execute("PRAGMA journal_mode;") as cursor:
            journal_row = await cursor.fetchone()
            assert journal_row is not None
            journal_mode = str(journal_row[0])
        async with checkpointer.conn.execute("PRAGMA busy_timeout;") as cursor:
            busy_timeout_row = await cursor.fetchone()
            assert busy_timeout_row is not None
            busy_timeout_ms = int(busy_timeout_row[0])
        return journal_mode, busy_timeout_ms

    def run_for(user_id: str) -> dict[str, object]:
        now = time.monotonic()
        return coordinator._run_agent_workflow(
            {
                "total_start_time": time.perf_counter(),
                "deadline_ts": now + 1.5,
                "owner": user_id,
            },
            thread_id="shared-thread",
            user_id=user_id,
            checkpoint_id=None,
            runtime_context=None,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            alice_future = executor.submit(run_for, "alice")
            bob_future = executor.submit(run_for, "bob")
            alice_result = alice_future.result(timeout=2.0)
            bob_result = bob_future.result(timeout=2.0)

        assert entered == 2
        assert alice_result["owner"] == "alice"
        assert bob_result["owner"] == "bob"
        assert alice_result["visits"] == 1
        assert bob_result["visits"] == 1
        assert not alice_result.get("workflow_stopped", False)
        assert not bob_result.get("workflow_stopped", False)

        alice_state = coordinator.get_state_values(
            thread_id="shared-thread", user_id="alice"
        )
        bob_state = coordinator.get_state_values(
            thread_id="shared-thread", user_id="bob"
        )
        assert alice_state["owner"] == "alice"
        assert bob_state["owner"] == "bob"

        alice_checkpoints = coordinator.list_checkpoints(
            thread_id="shared-thread", user_id="alice"
        )
        bob_checkpoints = coordinator.list_checkpoints(
            thread_id="shared-thread", user_id="bob"
        )
        assert alice_checkpoints
        assert bob_checkpoints
        assert (
            coordinator.fork_from_checkpoint(
                thread_id="shared-thread",
                user_id="alice",
                checkpoint_id=str(alice_checkpoints[0]["checkpoint_id"]),
            )
            is not None
        )
        assert checkpointer.loop is runner._loop
        assert connection_thread is not None
        assert connection_thread.is_alive()
        assert runner.run(read_connection_pragmas()) == ("wal", 5000)
    finally:
        coordinator.close()

    assert not runner.is_alive
    assert close_loops == [runner._loop]
    assert connection_thread is not None
    assert not connection_thread.is_alive()

    with closing(sqlite3.connect(db_path)) as conn, conn:
        persisted = conn.execute(
            """
            SELECT thread_id, checkpoint_ns, COUNT(*)
            FROM checkpoints
            GROUP BY thread_id, checkpoint_ns
            """
        ).fetchall()
    assert len({row[0] for row in persisted}) == 2
    assert all(str(row[0]).startswith("docmind:") for row in persisted)
    assert {row[1] for row in persisted} == {""}
