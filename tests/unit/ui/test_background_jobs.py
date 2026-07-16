"""Unit tests for Streamlit background job manager."""

from __future__ import annotations

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import src.ui.cache as cache_mod
from src.ui.background_jobs import (
    ForegroundRuntimeConflictError,
    JobAdmissionPausedError,
    JobCanceledError,
    JobConflictError,
    JobManager,
    ProgressEvent,
    _reset_job_manager_for_tests,
    get_job_manager,
)

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


def test_consume_terminal_is_owner_authorized_and_releases_result(
    default_timeout: float,
) -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    started = threading.Event()
    release = threading.Event()
    result_refs: list[weakref.ReferenceType[object]] = []

    class _Result:
        pass

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        result = _Result()
        result_refs.append(weakref.ref(result))
        started.set()
        assert release.wait(default_timeout)
        return result

    try:
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert started.wait(default_timeout)
        assert not manager.consume_terminal(job_id, owner_id="owner")
        assert not manager.consume_terminal(job_id, owner_id="foreign")

        release.set()
        assert (
            manager.wait_for_completion(
                job_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
        state = manager.get(job_id, owner_id="owner")
        assert state is not None
        assert state.result is result_refs[0]()
        managed_state = manager._jobs[job_id]
        assert not manager.consume_terminal(job_id, owner_id="foreign")

        assert manager.consume_terminal(job_id, owner_id="owner")
        assert state.result is result_refs[0]()
        assert state.error is None
        assert managed_state.result is None
        assert managed_state.error is None
        assert manager.get(job_id, owner_id="owner") is None
        assert job_id not in manager._jobs
        assert job_id not in manager._futures
        del state
        gc.collect()
        assert result_refs[0]() is None
        assert not manager.consume_terminal(job_id, owner_id="owner")
    finally:
        release.set()
        manager.shutdown()


@pytest.mark.parametrize("terminal_status", ["failed", "canceled"])
def test_consume_terminal_removes_data_free_terminal_states(
    default_timeout: float,
    terminal_status: str,
) -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        if terminal_status == "canceled":
            raise JobCanceledError()
        raise RuntimeError("expected failure")

    try:
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert (
            manager.wait_for_completion(
                job_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == terminal_status
        )
        assert manager.consume_terminal(job_id, owner_id="owner")
        assert manager.get(job_id, owner_id="owner") is None
    finally:
        manager.shutdown()


@pytest.mark.parametrize("terminal_status", ["succeeded", "failed", "canceled"])
def test_terminal_view_is_payload_complete_and_immutable(
    default_timeout: float,
    terminal_status: str,
) -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    result = object()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        if terminal_status == "failed":
            raise RuntimeError("expected failure")
        if terminal_status == "canceled":
            raise JobCanceledError()
        return result

    try:
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert (
            manager.wait_for_completion(
                job_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == terminal_status
        )
        view = manager.get(job_id, owner_id="owner")
        assert view is not None
        assert view.status == terminal_status
        if terminal_status == "succeeded":
            assert view.result is result
            assert view.error is None
        elif terminal_status == "failed":
            assert view.result is None
            assert view.error is not None
            assert "RuntimeError" in view.error
        else:
            assert view.result is None
            assert view.error is None
        with pytest.raises(AttributeError):
            view.status = "running"  # type: ignore[misc]
    finally:
        manager.shutdown()


def test_shutdown_terminal_retries_consumption_until_future_done(
    default_timeout: float,
) -> None:
    manager = JobManager(
        max_workers=1,
        job_ttl_sec=0,
        shutdown_grace_period_sec=0,
    )
    started = threading.Event()
    release = threading.Event()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(default_timeout)
        return object()

    job_id = manager.start_job(owner_id="owner", fn=_work)
    assert started.wait(default_timeout)
    manager.shutdown()
    view = manager.get(job_id, owner_id="owner")
    assert view is not None
    assert view.status == "canceled"
    assert not manager.consume_terminal(job_id, owner_id="owner")

    release.set()
    assert (
        manager.wait_for_completion(
            job_id,
            owner_id="owner",
            timeout_sec=default_timeout,
        )
        == "canceled"
    )
    assert manager.consume_terminal(job_id, owner_id="owner")
    assert manager.get(job_id, owner_id="owner") is None
    manager._join_shutdown_for_tests()


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


def test_ttl_cleanup_clears_manager_payload_before_release(
    default_timeout: float,
) -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0.01)
    result_refs: list[weakref.ReferenceType[object]] = []

    class _Result:
        pass

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        result = _Result()
        result_refs.append(weakref.ref(result))
        return result

    try:
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert (
            manager.wait_for_completion(
                job_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
        view = manager.get(job_id, owner_id="owner")
        assert view is not None
        assert view.result is result_refs[0]()
        with manager._lock:
            managed = manager._jobs[job_id]
            managed.last_seen_at = managed.created_at - timedelta(days=1)

        manager.activity_snapshot()

        assert managed.result is None
        assert managed.error is None
        assert job_id not in manager._jobs
        assert job_id not in manager._futures
        assert view.result is result_refs[0]()
        del view
        gc.collect()
        assert result_refs[0]() is None
    finally:
        manager.shutdown()


def test_corpus_mutation_slot_covers_queued_and_running_jobs(
    default_timeout: float,
) -> None:
    """Serialize one keyed corpus mutation across owners until it is terminal."""
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    mutation_started = threading.Event()
    release_mutation = threading.Event()

    def _blocker(_cancel_event, _report):  # type: ignore[no-untyped-def]
        blocker_started.set()
        assert release_blocker.wait(default_timeout)

    def _mutation(_cancel_event, _report):  # type: ignore[no-untyped-def]
        mutation_started.set()
        assert release_mutation.wait(default_timeout)

    try:
        manager.start_job(owner_id="unrelated", fn=_blocker)
        assert blocker_started.wait(default_timeout)
        first_id = manager.start_job(
            owner_id="owner-one",
            fn=_mutation,
            exclusivity_key="corpus-mutation",
        )
        assert manager.is_exclusivity_active("corpus-mutation") is True

        with pytest.raises(JobConflictError):
            manager.start_job(
                owner_id="owner-one",
                fn=_mutation,
                exclusivity_key="corpus-mutation",
            )
        with pytest.raises(JobConflictError):
            manager.start_job(
                owner_id="owner-two",
                fn=_mutation,
                exclusivity_key="corpus-mutation",
            )

        release_blocker.set()
        assert mutation_started.wait(default_timeout)
        with pytest.raises(JobConflictError):
            manager.start_job(
                owner_id="owner-two",
                fn=_mutation,
                exclusivity_key="corpus-mutation",
            )

        release_mutation.set()
        assert (
            manager.wait_for_completion(
                first_id,
                owner_id="owner-one",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
        assert manager.is_exclusivity_active("corpus-mutation") is False
        second_id = manager.start_job(
            owner_id="owner-two",
            fn=lambda _cancel, _report: "done",
            exclusivity_key="corpus-mutation",
        )
        assert (
            manager.wait_for_completion(
                second_id,
                owner_id="owner-two",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
    finally:
        release_blocker.set()
        release_mutation.set()
        manager.shutdown()


def test_admission_quiescence_rejects_active_jobs_and_reopens(
    default_timeout: float,
) -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    started = threading.Event()
    release = threading.Event()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(default_timeout)

    try:
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert started.wait(default_timeout)
        assert manager.has_active_jobs()
        with pytest.raises(JobConflictError), manager.admission_quiescence():
            pytest.fail("quiescence admitted while a job was active")

        release.set()
        assert (
            manager.wait_for_completion(
                job_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
        assert not manager.has_active_jobs()
        with manager.admission_quiescence():
            pass
        next_id = manager.start_job(
            owner_id="owner",
            fn=lambda _cancel, _report: None,
        )
        assert (
            manager.wait_for_completion(
                next_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
    finally:
        release.set()
        manager.shutdown()


def test_admission_quiescence_closes_start_race() -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    try:
        with manager.admission_quiescence(), pytest.raises(JobAdmissionPausedError):
            manager.start_job(
                owner_id="owner",
                fn=lambda _cancel, _report: None,
            )
        job_id = manager.start_job(
            owner_id="owner",
            fn=lambda _cancel, _report: None,
        )
        assert manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
    finally:
        manager.shutdown()


@pytest.mark.parametrize("late_outcome", ["success", "failure"])
def test_shutdown_preserves_running_job_cancellation(
    default_timeout: float,
    late_outcome: str,
) -> None:
    manager = JobManager(
        max_workers=1,
        job_ttl_sec=0,
        shutdown_grace_period_sec=0,
    )
    started = threading.Event()
    release = threading.Event()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(default_timeout)
        if late_outcome == "failure":
            raise RuntimeError("late failure")
        return "late result"

    job_id = manager.start_job(owner_id="owner", fn=_work)
    assert started.wait(default_timeout)
    manager.shutdown()
    state = manager.get(job_id, owner_id="owner")
    assert state is not None
    assert state.status == "canceled"

    release.set()
    assert (
        manager.wait_for_completion(
            job_id,
            owner_id="owner",
            timeout_sec=default_timeout,
        )
        == "canceled"
    )
    assert state.status == "canceled"
    assert state.result is None
    assert state.error is None
    manager._join_shutdown_for_tests()


def test_shutdown_preserves_queued_job_cancellation(
    default_timeout: float,
) -> None:
    manager = JobManager(
        max_workers=1,
        job_ttl_sec=0,
        shutdown_grace_period_sec=0,
    )
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    queued_ran = threading.Event()

    def _blocker(_cancel_event, _report):  # type: ignore[no-untyped-def]
        blocker_started.set()
        assert release_blocker.wait(default_timeout)
        return "late blocker result"

    def _queued(_cancel_event, _report):  # type: ignore[no-untyped-def]
        queued_ran.set()
        return "late queued result"

    blocker_id = manager.start_job(owner_id="owner", fn=_blocker)
    assert blocker_started.wait(default_timeout)
    queued_id = manager.start_job(owner_id="owner", fn=_queued)
    manager.shutdown()
    release_blocker.set()

    for job_id in (blocker_id, queued_id):
        assert (
            manager.wait_for_completion(
                job_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == "canceled"
        )
        state = manager.get(job_id, owner_id="owner")
        assert state is not None
        assert state.result is None
        assert state.error is None
    assert not queued_ran.is_set()
    manager._join_shutdown_for_tests()


def test_foreground_runtime_and_maintenance_exclude_each_other() -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    try:
        with manager.foreground_runtime_activity():
            activity = manager.activity_snapshot()
            assert activity.foreground_runtime_active
            assert not activity.maintenance_active
            with (
                pytest.raises(ForegroundRuntimeConflictError),
                manager.admission_quiescence(),
            ):
                pytest.fail("maintenance entered during foreground runtime work")

        with manager.admission_quiescence():
            activity = manager.activity_snapshot()
            assert not activity.foreground_runtime_active
            assert activity.maintenance_active
            with (
                pytest.raises(JobAdmissionPausedError),
                manager.foreground_runtime_activity(),
            ):
                pytest.fail("foreground runtime work entered during maintenance")

        with pytest.raises(RuntimeError), manager.foreground_runtime_activity():
            raise RuntimeError("query failed")
        assert not manager.activity_snapshot().foreground_runtime_active
    finally:
        manager.shutdown()


def test_nested_foreground_leases_exclude_maintenance_until_outer_exit() -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    try:
        with manager.foreground_runtime_activity():
            with manager.foreground_runtime_activity():
                assert manager.activity_snapshot().foreground_runtime_active
            assert manager.activity_snapshot().foreground_runtime_active
            with (
                pytest.raises(ForegroundRuntimeConflictError),
                manager.admission_quiescence(),
            ):
                pytest.fail("maintenance entered before the outer reader exited")
        assert not manager.activity_snapshot().foreground_runtime_active
        with manager.admission_quiescence():
            assert manager.activity_snapshot().maintenance_active
    finally:
        manager.shutdown()


def test_activity_snapshot_distinguishes_jobs_from_maintenance(
    default_timeout: float,
) -> None:
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    started = threading.Event()
    release = threading.Event()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(default_timeout)

    try:
        initial = manager.activity_snapshot()
        assert not initial.has_active_jobs
        assert not initial.foreground_runtime_active
        assert not initial.maintenance_active
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert started.wait(default_timeout)
        active = manager.activity_snapshot()
        assert active.has_active_jobs
        assert not active.foreground_runtime_active
        assert not active.maintenance_active
        release.set()
        assert manager.wait_for_completion(job_id, owner_id="owner") == "succeeded"
        with manager.admission_quiescence():
            maintenance = manager.activity_snapshot()
            assert not maintenance.has_active_jobs
            assert not maintenance.foreground_runtime_active
            assert maintenance.maintenance_active
    finally:
        release.set()
        manager.shutdown()


def test_concurrent_first_retrieval_returns_one_process_manager(
    default_timeout: float,
) -> None:
    """Construct one manager under concurrent first access and share its key slot."""
    callers = 8
    barrier = threading.Barrier(callers + 1)
    started = threading.Event()
    release = threading.Event()
    _reset_job_manager_for_tests()

    def _retrieve() -> JobManager:
        barrier.wait(default_timeout)
        return get_job_manager()

    def _work(_cancel_event, _report):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(default_timeout)

    try:
        with ThreadPoolExecutor(max_workers=callers) as executor:
            futures = [executor.submit(_retrieve) for _ in range(callers)]
            barrier.wait(default_timeout)
            managers = [future.result(timeout=default_timeout) for future in futures]

        assert all(manager is managers[0] for manager in managers)
        job_id = managers[0].start_job(
            owner_id="owner-one",
            fn=_work,
            exclusivity_key="corpus-mutation",
        )
        assert started.wait(default_timeout)
        with pytest.raises(JobConflictError):
            managers[-1].start_job(
                owner_id="owner-two",
                fn=lambda _cancel, _report: None,
                exclusivity_key="corpus-mutation",
            )
        release.set()
        assert (
            managers[-1].wait_for_completion(
                job_id,
                owner_id="owner-one",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
    finally:
        release.set()
        _reset_job_manager_for_tests()


def test_submit_failure_releases_exclusivity_key(
    monkeypatch: pytest.MonkeyPatch,
    default_timeout: float,
) -> None:
    """Remove queued ownership when executor submission rejects a job."""
    manager = JobManager(max_workers=1, job_ttl_sec=0)
    original_submit = manager._executor.submit
    reject_next = True

    def _submit(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal reject_next
        if reject_next:
            reject_next = False
            raise RuntimeError("executor rejected submission")
        return original_submit(*args, **kwargs)

    monkeypatch.setattr(manager._executor, "submit", _submit)
    try:
        with pytest.raises(RuntimeError, match="executor rejected submission"):
            manager.start_job(
                owner_id="owner-one",
                fn=lambda _cancel, _report: None,
                exclusivity_key="corpus-mutation",
            )
        assert manager.is_exclusivity_active("corpus-mutation") is False

        job_id = manager.start_job(
            owner_id="owner-two",
            fn=lambda _cancel, _report: "done",
            exclusivity_key="corpus-mutation",
        )
        assert (
            manager.wait_for_completion(
                job_id,
                owner_id="owner-two",
                timeout_sec=default_timeout,
            )
            == "succeeded"
        )
    finally:
        manager.shutdown()


def test_job_manager_survives_streamlit_cache_clear(
    monkeypatch: pytest.MonkeyPatch,
    default_timeout: float,
) -> None:
    """Retain and cancel a running job after settings and Streamlit cache changes."""
    started = threading.Event()

    class _Cache:
        def clear(self) -> None:
            return None

    streamlit_stub = SimpleNamespace(
        session_state={},
        cache_data=_Cache(),
        cache_resource=_Cache(),
    )
    monkeypatch.setattr(cache_mod, "st", streamlit_stub)
    settings_stub = SimpleNamespace(cache_version=7)

    _reset_job_manager_for_tests()
    manager = get_job_manager()

    def _work(cancel_event, _report):  # type: ignore[no-untyped-def]
        started.set()
        if cancel_event.wait(default_timeout * 3):
            raise JobCanceledError()
        raise AssertionError("job was not canceled")

    try:
        job_id = manager.start_job(owner_id="owner", fn=_work)
        assert started.wait(default_timeout)
        with pytest.raises(JobConflictError):
            cache_mod.clear_caches(settings_stub)
        assert settings_stub.cache_version == 7

        same_manager = get_job_manager()
        assert same_manager is manager
        assert same_manager.get(job_id, owner_id="owner") is not None
        assert same_manager.cancel(job_id, owner_id="owner") is True
        assert (
            same_manager.wait_for_completion(
                job_id,
                owner_id="owner",
                timeout_sec=default_timeout,
            )
            == "canceled"
        )
        assert cache_mod.clear_caches(settings_stub) == 8
    finally:
        _reset_job_manager_for_tests()


def test_reset_allows_reentrant_retrieval_without_publishing_second_manager(
    default_timeout: float,
) -> None:
    """Publish only the closed old manager until its workers have fully drained."""
    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_retrieved = threading.Event()
    reentrant_managers: list[JobManager] = []
    admission_rejected: list[bool] = []
    _reset_job_manager_for_tests()
    old_manager = get_job_manager()

    def _reentrant_work(cancel_event, _report):  # type: ignore[no-untyped-def]
        worker_started.set()
        assert cancel_event.wait(default_timeout)
        reentrant = get_job_manager()
        reentrant_managers.append(reentrant)
        try:
            reentrant.start_job(owner_id="owner", fn=lambda *_args: None)
        except RuntimeError:
            admission_rejected.append(True)
        worker_retrieved.set()
        assert release_worker.wait(default_timeout * 3)

    job_id = old_manager.start_job(owner_id="owner", fn=_reentrant_work)
    assert worker_started.wait(default_timeout)
    state = old_manager.get(job_id, owner_id="owner")
    assert state is not None

    try:
        with ThreadPoolExecutor(max_workers=1) as reset_executor:
            reset_future = reset_executor.submit(_reset_job_manager_for_tests)
            assert worker_retrieved.wait(default_timeout)

            with pytest.raises(RuntimeError, match="shut down"):
                old_manager.start_job(owner_id="owner", fn=lambda *_args: None)

            concurrent_manager = get_job_manager()
            assert concurrent_manager is old_manager
            assert reentrant_managers == [old_manager]
            assert admission_rejected == [True]
            assert not reset_future.done()

            release_worker.set()
            reset_future.result(timeout=default_timeout)

        new_manager = get_job_manager()
        assert new_manager is not old_manager
        assert new_manager.get(job_id, owner_id="owner") is None
    finally:
        release_worker.set()
        _reset_job_manager_for_tests()


def test_reset_waits_for_outstanding_maintenance_lease(
    default_timeout: float,
) -> None:
    lease_entered = threading.Event()
    release_lease = threading.Event()
    _reset_job_manager_for_tests()
    old_manager = get_job_manager()

    def _hold_maintenance() -> None:
        with old_manager.admission_quiescence():
            lease_entered.set()
            assert release_lease.wait(default_timeout * 3)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            lease_future = executor.submit(_hold_maintenance)
            assert lease_entered.wait(default_timeout)
            reset_future = executor.submit(_reset_job_manager_for_tests)
            deadline = time.monotonic() + default_timeout
            while time.monotonic() < deadline:
                with old_manager._lock:
                    if old_manager._closed:
                        break
                time.sleep(0.005)
            else:
                pytest.fail("reset did not close old admission")

            assert get_job_manager() is old_manager
            assert old_manager.activity_snapshot().maintenance_active
            with pytest.raises(RuntimeError, match="shut down"):
                old_manager.start_job(owner_id="owner", fn=lambda *_args: None)
            assert not reset_future.done()

            release_lease.set()
            lease_future.result(timeout=default_timeout)
            reset_future.result(timeout=default_timeout)

        assert get_job_manager() is not old_manager
    finally:
        release_lease.set()
        _reset_job_manager_for_tests()
