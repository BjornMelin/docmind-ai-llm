"""Background job manager for Streamlit pages.

Implements a lightweight, local-only job runner for long-lived tasks (e.g. file
ingestion) so Streamlit UI threads remain responsive.

Non-negotiables:
- Worker threads must not call Streamlit APIs.
- Progress events must not include raw document text or secrets.
"""

from __future__ import annotations

import atexit
import contextlib
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor, wait
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import streamlit as st

MAX_PROGRESS_QUEUE_SIZE = 100
SHUTDOWN_GRACE_PERIOD_SEC = 5.0
_DEFAULT_JOB_TTL_SEC = 60.0 * 60.0

JobStatus = Literal["queued", "running", "succeeded", "failed", "canceled"]
JobPhase = Literal["save", "ingest", "index", "snapshot", "analysis", "done"]


class JobCanceledError(RuntimeError):
    """Raised by a worker when cooperative cancellation is requested."""


class JobConflictError(RuntimeError):
    """Raised when a mutually exclusive background job is already active."""


class JobAdmissionPausedError(JobConflictError):
    """Raised when runtime mutation has paused all new job admissions."""


class ForegroundRuntimeConflictError(JobConflictError):
    """Raised when maintenance races with foreground runtime work."""


@dataclass(frozen=True, slots=True)
class JobActivitySnapshot:
    """Atomic process activity visible to Streamlit controls."""

    has_active_jobs: bool
    foreground_runtime_active: bool
    maintenance_active: bool


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Progress update emitted by a background job."""

    percent: int
    phase: JobPhase
    message: str
    timestamp: datetime


@dataclass(slots=True)
class _JobState:
    """In-memory state for a submitted job."""

    job_id: str
    owner_id: str
    created_at: datetime
    last_seen_at: datetime
    status: JobStatus
    cancel_event: threading.Event
    progress_queue: queue.Queue[ProgressEvent]
    exclusivity_key: str | None = None
    result: Any | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class JobStateView:
    """Immutable shallow snapshot published to job observers."""

    job_id: str
    owner_id: str
    created_at: datetime
    last_seen_at: datetime
    status: JobStatus
    exclusivity_key: str | None
    result: Any | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _JobOutcome:
    """Worker outcome published only after its Future is done."""

    status: Literal["succeeded", "failed", "canceled"]
    result: Any | None = None
    error: str | None = None


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _queue_put_latest(
    q: queue.Queue[ProgressEvent], event: ProgressEvent, *, max_attempts: int = 2
) -> None:
    """Insert event without blocking; keep the newest events when full."""
    for _ in range(max_attempts):
        try:
            q.put_nowait(event)
            return
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                q.get_nowait()


class JobManager:
    """Thread-backed job manager with bounded progress queues."""

    def __init__(
        self,
        *,
        max_workers: int = 2,
        max_progress_queue_size: int = MAX_PROGRESS_QUEUE_SIZE,
        job_ttl_sec: float = _DEFAULT_JOB_TTL_SEC,
        shutdown_grace_period_sec: float = SHUTDOWN_GRACE_PERIOD_SEC,
    ) -> None:
        """Initialize the JobManager with a thread pool and bounded queues."""
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="docmind-job",
        )
        self._max_progress_queue_size = max(1, int(max_progress_queue_size))
        self._job_ttl_sec = float(job_ttl_sec)
        self._shutdown_grace_period_sec = float(shutdown_grace_period_sec)

        self._lock = threading.RLock()
        self._activity_condition = threading.Condition(self._lock)
        self._jobs: dict[str, _JobState] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._closed = False
        self._maintenance_active = False
        self._foreground_runtime_count = 0

    def start_job(
        self,
        *,
        owner_id: str,
        fn: Callable[[threading.Event, Callable[[ProgressEvent], None]], Any],
        exclusivity_key: str | None = None,
    ) -> str:
        """Submit a job to the internal executor and return a job id.

        Args:
            owner_id: Owner identifier for authorization checks.
            fn: Worker callable invoked with (cancel_event, report_progress).
            exclusivity_key: Optional process-wide mutual-exclusion key.

        Returns:
            Job identifier string.

        Raises:
            ValueError: When owner_id is empty.
            JobConflictError: When a job with the same key is queued or running.
            RuntimeError: When the JobManager is shut down.
        """
        if not owner_id:
            raise ValueError("owner_id is required")
        with self._lock:
            if self._closed:
                raise RuntimeError("JobManager is shut down")
            if self._maintenance_active:
                raise JobAdmissionPausedError(
                    "Background job admission is paused for runtime maintenance"
                )
            self._cleanup_expired_locked()
            if exclusivity_key is not None and self._is_exclusivity_active_locked(
                exclusivity_key
            ):
                raise JobConflictError(
                    f"A job with exclusivity key {exclusivity_key!r} is active"
                )
            job_id = str(uuid.uuid4())
            state = _JobState(
                job_id=job_id,
                owner_id=str(owner_id),
                created_at=_now_utc(),
                last_seen_at=_now_utc(),
                status="queued",
                cancel_event=threading.Event(),
                progress_queue=queue.Queue(maxsize=self._max_progress_queue_size),
                exclusivity_key=exclusivity_key,
            )
            self._jobs[job_id] = state

            try:
                fut = self._executor.submit(self._run_job, job_id, fn)
            except Exception:
                self._jobs.pop(job_id, None)
                raise
            self._futures[job_id] = fut
            fut.add_done_callback(
                lambda completed, completed_job_id=job_id: self._complete_job(
                    completed_job_id,
                    completed,
                )
            )
            return job_id

    def is_exclusivity_active(self, exclusivity_key: str) -> bool:
        """Return whether a queued or running job owns an exclusivity key.

        Args:
            exclusivity_key: Process-wide mutual-exclusion key to inspect.

        Returns:
            True when the key is owned by a queued or running job.
        """
        with self._lock:
            self._cleanup_expired_locked()
            return self._is_exclusivity_active_locked(exclusivity_key)

    def has_active_jobs(self) -> bool:
        """Return whether any queued or running process job exists."""
        return self.activity_snapshot().has_active_jobs

    def activity_snapshot(self) -> JobActivitySnapshot:
        """Return active-job and maintenance state from one lock acquisition."""
        with self._lock:
            self._cleanup_expired_locked()
            return self._activity_snapshot_locked()

    def exclusivity_activity_snapshot(
        self, exclusivity_key: str
    ) -> tuple[bool, JobActivitySnapshot]:
        """Return key occupancy and process activity from one lock acquisition."""
        with self._lock:
            self._cleanup_expired_locked()
            return (
                self._is_exclusivity_active_locked(exclusivity_key),
                self._activity_snapshot_locked(),
            )

    @contextlib.contextmanager
    def foreground_runtime_activity(self) -> Iterator[None]:
        """Register synchronous runtime work that maintenance must not retire."""
        with self._lock:
            if self._closed:
                raise RuntimeError("JobManager is shut down")
            if self._maintenance_active:
                raise JobAdmissionPausedError(
                    "Foreground runtime admission is paused for maintenance"
                )
            self._foreground_runtime_count += 1
        try:
            yield
        finally:
            with self._activity_condition:
                self._foreground_runtime_count -= 1
                self._activity_condition.notify_all()

    @contextlib.contextmanager
    def admission_quiescence(self) -> Iterator[None]:
        """Pause job admission while an idle runtime mutation is in progress."""
        with self._lock:
            if self._closed:
                raise RuntimeError("JobManager is shut down")
            self._cleanup_expired_locked()
            if self._maintenance_active:
                raise JobAdmissionPausedError(
                    "Background job admission is already paused"
                )
            if self._has_active_jobs_locked():
                raise JobConflictError(
                    "Runtime maintenance cannot start while background jobs are active"
                )
            if self._foreground_runtime_count:
                raise ForegroundRuntimeConflictError(
                    "Runtime maintenance cannot start while foreground runtime work "
                    "is active"
                )
            self._maintenance_active = True
        try:
            yield
        finally:
            with self._activity_condition:
                self._maintenance_active = False
                self._activity_condition.notify_all()

    def cancel(self, job_id: str, *, owner_id: str) -> bool:
        """Request best-effort cancellation for a job.

        Args:
            job_id: Job identifier to cancel.
            owner_id: Owner identifier for authorization checks.

        Returns:
            True when the cancellation request was accepted.
        """
        with self._lock:
            self._cleanup_expired_locked()
            state = self._jobs.get(job_id)
            if state is None or state.owner_id != owner_id:
                return False
            state.last_seen_at = _now_utc()
            state.cancel_event.set()
            return True

    def get(self, job_id: str, *, owner_id: str) -> JobStateView | None:
        """Return the job state (or None when missing/unauthorized).

        Args:
            job_id: Job identifier to fetch.
            owner_id: Owner identifier for authorization checks.

        Returns:
            Immutable job-state snapshot when available; otherwise None.
        """
        with self._lock:
            self._cleanup_expired_locked()
            state = self._jobs.get(job_id)
            if state is None or state.owner_id != owner_id:
                return None
            state.last_seen_at = _now_utc()
            return self._state_view_locked(state)

    def consume_terminal(self, job_id: str, *, owner_id: str) -> bool:
        """Atomically release one completed job owned by the caller.

        Args:
            job_id: Job identifier to consume.
            owner_id: Owner identifier for authorization checks.

        Returns:
            True when a terminal job and its completed future were removed.
        """
        with self._lock:
            self._cleanup_expired_locked()
            state = self._jobs.get(job_id)
            future = self._futures.get(job_id)
            if state is None or state.owner_id != owner_id:
                return False
            if state.status not in ("succeeded", "failed", "canceled"):
                return False
            if future is not None and not future.done():
                return False
            self._release_job_locked(job_id, state)
            return True

    def drain_progress(
        self, job_id: str, *, owner_id: str, max_events: int = 100
    ) -> list[ProgressEvent]:
        """Drain progress events for a job without blocking.

        Args:
            job_id: Job identifier to poll.
            owner_id: Owner identifier for authorization checks.
            max_events: Maximum number of events to drain.

        Returns:
            List of progress events in queue order.
        """
        with self._lock:
            self._cleanup_expired_locked()
            state = self._jobs.get(job_id)
            if state is None or state.owner_id != owner_id:
                return []
            state.last_seen_at = _now_utc()
            drained: list[ProgressEvent] = []
            for _ in range(max(0, int(max_events))):
                try:
                    drained.append(state.progress_queue.get_nowait())
                except queue.Empty:
                    break
            return drained

    def wait_for_completion(
        self, job_id: str, *, owner_id: str, timeout_sec: float = 10.0
    ) -> JobStatus | None:
        """Block until a job completes (primarily for unit tests).

        Args:
            job_id: Job identifier to await.
            owner_id: Owner identifier for authorization checks.
            timeout_sec: Maximum seconds to wait.

        Returns:
            Final job status or None when missing/unauthorized/timeout.
        """
        with self._lock:
            state = self._jobs.get(job_id)
            fut = self._futures.get(job_id)
            if state is None or state.owner_id != owner_id or fut is None:
                return None
        timeout = max(0.0, float(timeout_sec))
        deadline = time.monotonic() + timeout
        try:
            _ = fut.result(timeout=timeout)
        except FuturesTimeoutError:
            return None
        with self._activity_condition:
            remaining = max(0.0, deadline - time.monotonic())
            self._activity_condition.wait_for(
                lambda: (
                    (current := self._jobs.get(job_id)) is None
                    or current.owner_id != owner_id
                    or current.status in ("succeeded", "failed", "canceled")
                ),
                timeout=remaining,
            )
            st_state = self._jobs.get(job_id)
            if (
                st_state is None
                or st_state.owner_id != owner_id
                or st_state.status not in ("succeeded", "failed", "canceled")
            ):
                return None
            return st_state.status

    def shutdown(self) -> None:
        """Best-effort shutdown of the executor and in-flight jobs.

        Returns:
            None.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            # Ask all jobs to stop cooperatively.
            for state in self._jobs.values():
                state.cancel_event.set()

            futures = list(self._futures.values())
        if futures:
            wait(futures, timeout=max(0.0, self._shutdown_grace_period_sec))
        with contextlib.suppress(Exception):
            self._executor.shutdown(wait=False, cancel_futures=False)

        # Mark any non-terminal jobs as canceled.
        with self._lock:
            for state in self._jobs.values():
                if state.status in ("queued", "running"):
                    self._publish_terminal_locked(state, status="canceled")

    def _begin_shutdown_for_tests(self) -> None:
        """Close admission and request cancellation before a test drain."""
        with self._lock:
            self._closed = True
            for state in self._jobs.values():
                state.cancel_event.set()

    def _join_shutdown_for_tests(self) -> None:
        """Join workers and maintenance after test shutdown has been requested."""
        self._executor.shutdown(wait=True, cancel_futures=False)

        with self._activity_condition:
            self._activity_condition.wait_for(
                lambda: (
                    not self._maintenance_active and self._foreground_runtime_count == 0
                )
            )
            for state in self._jobs.values():
                if state.status in ("queued", "running"):
                    self._publish_terminal_locked(state, status="canceled")

    def _run_job(
        self,
        job_id: str,
        fn: Callable[[threading.Event, Callable[[ProgressEvent], None]], Any],
    ) -> _JobOutcome:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return _JobOutcome(status="canceled")
            if state.status in ("succeeded", "failed", "canceled"):
                return _JobOutcome(status="canceled")
            if self._closed:
                return _JobOutcome(status="canceled")
            state.status = "running"
            cancel_event = state.cancel_event

        def report_progress(event: ProgressEvent) -> None:
            with self._lock:
                st_state = self._jobs.get(job_id)
                if st_state is not None and not self._closed:
                    _queue_put_latest(st_state.progress_queue, event)

        try:
            return _JobOutcome(
                status="succeeded",
                result=fn(cancel_event, report_progress),
            )
        except JobCanceledError:
            return _JobOutcome(status="canceled")
        except Exception as exc:  # pragma: no cover - defensive
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="background_jobs.run")
            return _JobOutcome(
                status="failed",
                error=f"{type(exc).__name__} ({redaction.redacted})",
            )

    def _complete_job(self, job_id: str, future: Future[Any]) -> None:
        """Publish one payload-first terminal outcome after Future completion."""
        try:
            outcome = future.result()
            if not isinstance(outcome, _JobOutcome):  # pragma: no cover - defensive
                outcome = _JobOutcome(
                    status="failed",
                    error="RuntimeError ([redacted:invalid-job-outcome])",
                )
        except BaseException as exc:  # pragma: no cover - defensive callback boundary
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(
                str(exc),
                key_id="background_jobs.complete",
            )
            outcome = _JobOutcome(
                status="failed",
                error=f"{type(exc).__name__} ({redaction.redacted})",
            )
        with self._activity_condition:
            state = self._jobs.get(job_id)
            if state is not None and not self._preserve_canceled_terminal_locked(state):
                self._publish_terminal_locked(
                    state,
                    status=outcome.status,
                    result=outcome.result,
                    error=outcome.error,
                )
            self._activity_condition.notify_all()

    def _preserve_canceled_terminal_locked(self, state: _JobState) -> bool:
        """Keep manager closure or cancellation terminal and data-free."""
        if not self._closed and state.status != "canceled":
            return False
        self._publish_terminal_locked(state, status="canceled")
        return True

    @staticmethod
    def _publish_terminal_locked(
        state: _JobState,
        *,
        status: Literal["succeeded", "failed", "canceled"],
        result: Any | None = None,
        error: str | None = None,
    ) -> None:
        """Publish payload fields before terminal status, the release marker."""
        state.result = result
        state.error = error
        state.status = status

    @staticmethod
    def _state_view_locked(state: _JobState) -> JobStateView:
        """Copy one internally consistent observer view while holding the lock."""
        return JobStateView(
            job_id=state.job_id,
            owner_id=state.owner_id,
            created_at=state.created_at,
            last_seen_at=state.last_seen_at,
            status=state.status,
            exclusivity_key=state.exclusivity_key,
            result=state.result,
            error=state.error,
        )

    def _release_job_locked(self, job_id: str, state: _JobState) -> None:
        """Clear manager-owned terminal payloads before dropping both owners."""
        state.result = None
        state.error = None
        self._jobs.pop(job_id, None)
        self._futures.pop(job_id, None)

    def _cleanup_expired_locked(self) -> None:
        if self._job_ttl_sec <= 0:
            return
        now = _now_utc()
        expired: list[str] = []
        for job_id, st_state in self._jobs.items():
            age = (now - st_state.last_seen_at).total_seconds()
            fut = self._futures.get(job_id)
            if (
                age > self._job_ttl_sec
                and st_state.status in ("succeeded", "failed", "canceled")
                and (fut is None or fut.done())
            ):
                expired.append(job_id)
        for job_id in expired:
            state = self._jobs.get(job_id)
            if state is not None:
                self._release_job_locked(job_id, state)

    def _is_exclusivity_active_locked(self, exclusivity_key: str) -> bool:
        return any(
            state.exclusivity_key == exclusivity_key
            and state.status in ("queued", "running")
            for state in self._jobs.values()
        )

    def _has_active_jobs_locked(self) -> bool:
        return any(
            state.status in ("queued", "running") for state in self._jobs.values()
        )

    def _activity_snapshot_locked(self) -> JobActivitySnapshot:
        return JobActivitySnapshot(
            has_active_jobs=self._has_active_jobs_locked(),
            foreground_runtime_active=self._foreground_runtime_count > 0,
            maintenance_active=self._maintenance_active,
        )


_job_manager_lock = threading.Lock()
_job_manager_reset_lock = threading.Lock()
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    """Return the process-lifetime JobManager instance.

    Returns:
        Process-lifetime JobManager instance.
    """
    global _job_manager

    with _job_manager_lock:
        if _job_manager is None:
            _job_manager = JobManager()
        return _job_manager


def _reset_job_manager_for_tests() -> None:
    """Shut down and clear the process manager cache for deterministic tests."""
    global _job_manager

    with _job_manager_reset_lock:
        with _job_manager_lock:
            manager = _job_manager
            if manager is None:
                return
            manager._begin_shutdown_for_tests()

        manager._join_shutdown_for_tests()

        with _job_manager_lock:
            if _job_manager is manager:
                _job_manager = None


def _shutdown_job_manager_at_exit() -> None:
    """Shut down the current process manager at interpreter exit."""
    with _job_manager_lock:
        manager = _job_manager
    if manager is not None:
        manager.shutdown()


atexit.register(_shutdown_job_manager_at_exit)


def get_or_create_owner_id() -> str:
    """Return a stable owner id for the current Streamlit session.

    Returns:
        Owner identifier string.
    """
    key = "docmind_owner_id"
    current = st.session_state.get(key)
    if isinstance(current, str) and current:
        return current
    new_id = str(uuid.uuid4())
    st.session_state[key] = new_id
    return new_id


__all__ = [
    "MAX_PROGRESS_QUEUE_SIZE",
    "SHUTDOWN_GRACE_PERIOD_SEC",
    "ForegroundRuntimeConflictError",
    "JobActivitySnapshot",
    "JobAdmissionPausedError",
    "JobCanceledError",
    "JobConflictError",
    "JobManager",
    "JobPhase",
    "JobStateView",
    "JobStatus",
    "ProgressEvent",
    "get_job_manager",
    "get_or_create_owner_id",
]
