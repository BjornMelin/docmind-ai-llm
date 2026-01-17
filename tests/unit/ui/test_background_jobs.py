"""Unit tests for Streamlit background job manager."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest

from src.ui.background_jobs import JobCanceledError, JobManager, ProgressEvent

pytestmark = pytest.mark.unit


def _wait_for_terminal(manager: JobManager, job_id: str, *, owner_id: str) -> str:
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        state = manager.get(job_id, owner_id=owner_id)
        if state is None:
            return "missing"
        if state.status in ("succeeded", "failed", "canceled"):
            return state.status
        time.sleep(0.01)
    raise AssertionError("job did not reach terminal state in time")


def test_job_manager_runs_job_and_keeps_progress_bounded() -> None:
    manager = JobManager(max_workers=1, max_progress_queue_size=3, job_ttl_sec=0)

    def _work(cancel_event, report):  # type: ignore[no-untyped-def]
        for pct in range(0, 101, 10):
            report(
                ProgressEvent(
                    percent=pct,
                    phase="ingest",
                    message=f"pct={pct}",
                    timestamp=datetime.now(UTC),
                )
            )
        return {"ok": True}

    owner_id = "owner"
    job_id = manager.start_job(owner_id=owner_id, fn=_work)
    status = _wait_for_terminal(manager, job_id, owner_id=owner_id)
    assert status == "succeeded"

    state = manager.get(job_id, owner_id=owner_id)
    assert state is not None
    assert state.result == {"ok": True}

    events = manager.drain_progress(job_id, owner_id=owner_id, max_events=100)
    assert 1 <= len(events) <= 3
    assert all(isinstance(evt, ProgressEvent) for evt in events)
    assert events[-1].percent == 100


def test_job_manager_cancel_sets_status() -> None:
    manager = JobManager(max_workers=1, max_progress_queue_size=5, job_ttl_sec=0)

    def _work(cancel_event, report):  # type: ignore[no-untyped-def]
        report(
            ProgressEvent(
                percent=0,
                phase="save",
                message="start",
                timestamp=datetime.now(UTC),
            )
        )
        if cancel_event.is_set():
            raise JobCanceledError()
        time.sleep(0.05)
        return {"ok": True}

    owner_id = "owner"
    job_id = manager.start_job(owner_id=owner_id, fn=_work)
    assert manager.cancel(job_id, owner_id=owner_id) is True
    status = _wait_for_terminal(manager, job_id, owner_id=owner_id)
    assert status == "canceled"


def test_job_manager_cleanup_expires_done_jobs() -> None:
    manager = JobManager(max_workers=1, max_progress_queue_size=1, job_ttl_sec=0.01)

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        return {"ok": True}

    owner_id = "owner"
    job_id = manager.start_job(owner_id=owner_id, fn=_work)
    assert (
        manager.wait_for_completion(job_id, owner_id=owner_id, timeout_sec=1.0)
        == "succeeded"
    )
    # Simulate an orphaned job (not polled recently) by touching private state;
    # no public hook exists for forcing stale timestamps in tests.
    with manager._lock:
        state = manager._jobs.get(job_id)
        assert state is not None
        state.last_seen_at = state.created_at - timedelta(days=1)
    assert manager.get(job_id, owner_id=owner_id) is None
