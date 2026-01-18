"""Shared fakes for background job manager tests."""

from __future__ import annotations

import threading

import pytest

import src.ui.background_jobs as bg


class _State:
    def __init__(self, *, owner_id: str, status: str, result, error: str | None):
        self.owner_id = owner_id
        self.status = status
        self.result = result
        self.error = error


class _FakeJobManager:
    def __init__(self) -> None:
        self._events: dict[str, list[bg.ProgressEvent]] = {}
        self._states: dict[str, _State] = {}

    def start_job(self, *, owner_id: str, fn):  # type: ignore[no-untyped-def]
        job_id = "job-1"
        events: list[bg.ProgressEvent] = []

        def _report(evt: bg.ProgressEvent) -> None:
            events.append(evt)

        try:
            res = fn(threading.Event(), _report)
            state = _State(
                owner_id=owner_id, status="succeeded", result=res, error=None
            )
        except Exception as exc:  # pragma: no cover - defensive
            state = _State(
                owner_id=owner_id, status="failed", result=None, error=str(exc)
            )
        self._events[job_id] = events
        self._states[job_id] = state
        return job_id

    def get(self, job_id: str, *, owner_id: str):  # type: ignore[no-untyped-def]
        state = self._states.get(job_id)
        if state is None or state.owner_id != owner_id:
            return None
        return state

    def drain_progress(  # type: ignore[no-untyped-def]
        self,
        job_id: str,
        *,
        owner_id: str,
        max_events: int = 100,
    ):
        _ = max_events
        state = self._states.get(job_id)
        if state is None or state.owner_id != owner_id:
            return []
        events = self._events.get(job_id, [])
        self._events[job_id] = []
        return events

    def cancel(self, *_a, **_k):  # type: ignore[no-untyped-def]
        return True


@pytest.fixture
def fake_job_manager() -> _FakeJobManager:
    return _FakeJobManager()


@pytest.fixture
def fake_job_owner_id() -> str:
    return "owner"


__all__ = ["_FakeJobManager", "_State", "fake_job_manager", "fake_job_owner_id"]
