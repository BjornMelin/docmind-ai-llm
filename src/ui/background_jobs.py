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
import uuid
from collections.abc import Callable
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


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Progress update emitted by a background job."""

    percent: int
    phase: JobPhase
    message: str
    timestamp: datetime


@dataclass(slots=True)
class JobState:
    """In-memory state for a submitted job."""

    job_id: str
    owner_id: str
    created_at: datetime
    last_seen_at: datetime
    status: JobStatus
    cancel_event: threading.Event
    progress_queue: queue.Queue[ProgressEvent]
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

        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._closed = False

    def start_job(
        self,
        *,
        owner_id: str,
        fn: Callable[[threading.Event, Callable[[ProgressEvent], None]], Any],
    ) -> str:
        """Submit a job to the internal executor and return a job id.

        Args:
            owner_id: Owner identifier for authorization checks.
            fn: Worker callable invoked with (cancel_event, report_progress).

        Returns:
            Job identifier string.

        Raises:
            ValueError: When owner_id is empty.
            RuntimeError: When the JobManager is shut down.
        """
        if not owner_id:
            raise ValueError("owner_id is required")
        with self._lock:
            if self._closed:
                raise RuntimeError("JobManager is shut down")
            self._cleanup_expired_locked()
            job_id = str(uuid.uuid4())
            state = JobState(
                job_id=job_id,
                owner_id=str(owner_id),
                created_at=_now_utc(),
                last_seen_at=_now_utc(),
                status="queued",
                cancel_event=threading.Event(),
                progress_queue=queue.Queue(maxsize=self._max_progress_queue_size),
            )
            self._jobs[job_id] = state

            fut = self._executor.submit(self._run_job, job_id, fn)
            self._futures[job_id] = fut
            return job_id

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
            if state.status in ("queued", "running"):
                state.status = "canceled"
            fut = self._futures.get(job_id)
            if fut is not None:
                with contextlib.suppress(Exception):
                    fut.cancel()
            return True

    def get(self, job_id: str, *, owner_id: str) -> JobState | None:
        """Return the job state (or None when missing/unauthorized).

        Args:
            job_id: Job identifier to fetch.
            owner_id: Owner identifier for authorization checks.

        Returns:
            JobState when available; otherwise None.
        """
        with self._lock:
            self._cleanup_expired_locked()
            state = self._jobs.get(job_id)
            if state is None or state.owner_id != owner_id:
                return None
            state.last_seen_at = _now_utc()
            return state

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
        state = self.get(job_id, owner_id=owner_id)
        if state is None:
            return []
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
        try:
            _ = fut.result(timeout=max(0.0, float(timeout_sec)))
        except FuturesTimeoutError:
            return None
        with self._lock:
            st_state = self._jobs.get(job_id)
            if st_state is None or st_state.owner_id != owner_id:
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
            self._executor.shutdown(wait=False, cancel_futures=True)

        # Mark any non-terminal jobs as canceled.
        with self._lock:
            for state in self._jobs.values():
                if state.status in ("queued", "running"):
                    state.status = "canceled"

    def _run_job(
        self,
        job_id: str,
        fn: Callable[[threading.Event, Callable[[ProgressEvent], None]], Any],
    ) -> Any:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            if state.cancel_event.is_set() or state.status == "canceled":
                state.status = "canceled"
                return None
            state.status = "running"
            cancel_event = state.cancel_event

        def report_progress(event: ProgressEvent) -> None:
            with self._lock:
                st_state = self._jobs.get(job_id)
                if st_state is not None:
                    _queue_put_latest(st_state.progress_queue, event)

        final_result = None
        try:
            final_result = fn(cancel_event, report_progress)
            with self._lock:
                st_state = self._jobs.get(job_id)
                if st_state is not None:
                    if st_state.cancel_event.is_set():
                        st_state.status = "canceled"
                        st_state.result = None
                        final_result = None
                    else:
                        st_state.status = "succeeded"
                        st_state.result = final_result
                        st_state.error = None
        except JobCanceledError:
            with self._lock:
                st_state = self._jobs.get(job_id)
                if st_state is not None:
                    st_state.status = "canceled"
                    st_state.result = None
                    st_state.error = None
            final_result = None
        except Exception as exc:  # pragma: no cover - defensive
            with self._lock:
                st_state = self._jobs.get(job_id)
                if st_state is not None:
                    from src.utils.log_safety import build_pii_log_entry

                    redaction = build_pii_log_entry(
                        str(exc), key_id="background_jobs.run"
                    )
                    st_state.status = "failed"
                    st_state.result = None
                    st_state.error = f"{type(exc).__name__} ({redaction.redacted})"
            final_result = None

        return final_result

    def _cleanup_expired_locked(self) -> None:
        if self._job_ttl_sec <= 0:
            return
        now = _now_utc()
        expired: list[str] = []
        for job_id, st_state in self._jobs.items():
            age = (now - st_state.last_seen_at).total_seconds()
            fut = self._futures.get(job_id)
            if age > self._job_ttl_sec and (fut is None or fut.done()):
                expired.append(job_id)
        for job_id in expired:
            self._jobs.pop(job_id, None)
            self._futures.pop(job_id, None)


@st.cache_resource(show_spinner=False)
def get_job_manager(cache_version: int) -> JobManager:
    """Return a process-wide JobManager instance (cached by Streamlit).

    Args:
        cache_version: Cache-busting integer for Streamlit resource caching.

    Returns:
        Cached JobManager instance.
    """
    _ = cache_version  # cache bust
    manager = JobManager()
    atexit.register(manager.shutdown)
    return manager


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
    "JobCanceledError",
    "JobManager",
    "JobPhase",
    "JobState",
    "JobStatus",
    "ProgressEvent",
    "get_job_manager",
    "get_or_create_owner_id",
]
