"""Unit tests for Streamlit background job manager."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime, timedelta

import pytest

from src.ui.background_jobs import JobCanceledError, JobManager, ProgressEvent

pytestmark = pytest.mark.unit


def _wait_for_terminal(
    manager: JobManager,
    job_id: str,
    *,
    owner_id: str,
    timeout: float,
) -> str:
    """Wait for a job to reach a terminal state.

    Args:
        manager (JobManager): Job manager under test.
        job_id (str): Job identifier to poll.
        owner_id (str): Owner used for manager.get(job_id, owner_id=owner_id).
        timeout (float): Deadline for terminal status polling.

    Returns:
        str: "missing" or the terminal status from the job state.

    Raises:
        AssertionError: If the job fails to reach a terminal status before the
            deadline while polling manager.get(job_id, owner_id=owner_id) for a
            status in ("succeeded", "failed", "canceled").
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = manager.get(job_id, owner_id=owner_id)
        if state is None:
            return "missing"
        if state.status in ("succeeded", "failed", "canceled"):
            return state.status
        time.sleep(0.01)
    raise AssertionError("job did not reach terminal state in time")


def test_job_manager_runs_job_and_keeps_progress_bounded(
    default_timeout: float,
) -> None:
    """Verify job execution, bounded progress queue, and result handling.

    Args:
        default_timeout (float): Timeout fixture used by the test.

    Returns:
        None
    """
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
    status = _wait_for_terminal(
        manager,
        job_id,
        owner_id=owner_id,
        timeout=default_timeout,
    )
    assert status == "succeeded"

    state = manager.get(job_id, owner_id=owner_id)
    assert state is not None
    assert state.result == {"ok": True}

    events = manager.drain_progress(job_id, owner_id=owner_id, max_events=100)
    assert 1 <= len(events) <= 3
    assert all(isinstance(evt, ProgressEvent) for evt in events)
    assert events[-1].percent == 100


def test_job_manager_cancel_sets_status(default_timeout: float) -> None:
    """Verify cancellation updates job status and respects cancel events.

    Args:
        default_timeout (float): Timeout fixture used by the test.

    Returns:
        None
    """
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
        if cancel_event.wait(0.05):
            raise JobCanceledError()
        return {"ok": True}

    owner_id = "owner"
    job_id = manager.start_job(owner_id=owner_id, fn=_work)
    assert manager.cancel(job_id, owner_id=owner_id) is True
    status = _wait_for_terminal(
        manager,
        job_id,
        owner_id=owner_id,
        timeout=default_timeout,
    )
    assert status == "canceled"


def test_job_manager_success_wins_over_late_cancel(default_timeout: float) -> None:
    """Keep a committed return value when cancellation arrives after its checkpoint."""
    manager = JobManager(max_workers=1, max_progress_queue_size=1, job_ttl_sec=0)
    entered_commit = threading.Event()
    finish_commit = threading.Event()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        entered_commit.set()
        assert finish_commit.wait(default_timeout)
        return {"snapshot_id": "committed"}

    owner_id = "owner"
    job_id = manager.start_job(owner_id=owner_id, fn=_work)
    assert entered_commit.wait(default_timeout)
    assert manager.cancel(job_id, owner_id=owner_id) is True
    state = manager.get(job_id, owner_id=owner_id)
    assert state is not None
    assert state.status == "running"

    finish_commit.set()
    assert (
        manager.wait_for_completion(
            job_id,
            owner_id=owner_id,
            timeout_sec=default_timeout,
        )
        == "succeeded"
    )
    state = manager.get(job_id, owner_id=owner_id)
    assert state is not None
    assert state.result == {"snapshot_id": "committed"}


def test_queued_cancellation_still_runs_worker_cleanup(
    default_timeout: float,
) -> None:
    """Queued jobs enter their cooperative boundary instead of leaking staged data."""
    manager = JobManager(max_workers=1, max_progress_queue_size=1, job_ttl_sec=0)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    cleanup_ran = threading.Event()

    def _blocker(_cancel_event, _report):  # type: ignore[no-untyped-def]
        blocker_started.set()
        assert release_blocker.wait(default_timeout)

    def _queued(cancel_event, _report):  # type: ignore[no-untyped-def]
        if cancel_event.is_set():
            cleanup_ran.set()
            raise JobCanceledError()
        raise AssertionError("queued worker did not receive cancellation")

    owner_id = "owner"
    manager.start_job(owner_id=owner_id, fn=_blocker)
    assert blocker_started.wait(default_timeout)
    queued_id = manager.start_job(owner_id=owner_id, fn=_queued)
    assert manager.cancel(queued_id, owner_id=owner_id) is True

    release_blocker.set()
    assert (
        manager.wait_for_completion(
            queued_id,
            owner_id=owner_id,
            timeout_sec=default_timeout,
        )
        == "canceled"
    )
    assert cleanup_ran.is_set()


def test_job_manager_cleanup_expires_done_jobs(default_timeout: float) -> None:
    """Verify JobManager prunes jobs that have exceeded their TTL.

    Args:
        default_timeout (float): Timeout fixture used by the test.

    Returns:
        None
    """
    manager = JobManager(max_workers=1, max_progress_queue_size=1, job_ttl_sec=0.01)

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        return {"ok": True}

    owner_id = "owner"
    job_id = manager.start_job(owner_id=owner_id, fn=_work)
    assert (
        manager.wait_for_completion(
            job_id,
            owner_id=owner_id,
            timeout_sec=default_timeout,
        )
        == "succeeded"
    )
    # Simulate an orphaned job (not polled recently) by touching private state;
    # no public hook exists for forcing stale timestamps in tests.
    with manager._lock:
        state = manager._jobs.get(job_id)
        assert state is not None
        state.last_seen_at = state.created_at - timedelta(days=1)
    assert manager.get(job_id, owner_id=owner_id) is None
