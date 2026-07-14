"""Cancellation and lifecycle tests for retrieval CPU work."""

from __future__ import annotations

import asyncio
import threading

import pytest

from src.retrieval.async_work import AsyncWorkCapacityError, AsyncWorkExecutor

pytestmark = pytest.mark.unit


async def _wait_for(event: threading.Event) -> None:
    async with asyncio.timeout(1.0):
        while not event.is_set():
            await asyncio.sleep(0.001)


async def test_cancelled_waiter_keeps_capacity_until_worker_finishes() -> None:
    executor = AsyncWorkExecutor(name="test-retrieval-cpu")
    started = threading.Event()
    release = threading.Event()
    worker_names: list[str] = []

    def _blocking_work() -> str:
        worker_names.append(threading.current_thread().name)
        started.set()
        release.wait()
        return "done"

    try:
        task = asyncio.create_task(executor.run(_blocking_work))
        await _wait_for(started)
        assert worker_names[0].startswith("test-retrieval-cpu")
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with pytest.raises(AsyncWorkCapacityError):
            await executor.run(lambda: "must not start")

        close_task = asyncio.create_task(executor.aclose())
        await asyncio.sleep(0)
        assert close_task.done() is False

        release.set()
        await asyncio.wait_for(close_task, timeout=1.0)
        with pytest.raises(RuntimeError, match="closed"):
            await executor.run(lambda: "must not start")
    finally:
        release.set()
        executor.close()


def test_sync_result_releases_capacity_before_completion_callback() -> None:
    """A sequential stage must be admissible as soon as run_sync returns."""
    executor = AsyncWorkExecutor(name="test-retrieval-sync")
    started = threading.Event()
    release_work = threading.Event()
    lock_held = threading.Event()
    release_lock = threading.Event()

    def _work() -> str:
        started.set()
        release_work.wait()
        return "done"

    def _hold_callback_lock() -> None:
        started.wait()
        executor._lock.acquire()
        lock_held.set()
        release_work.set()
        release_lock.wait()
        executor._lock.release()

    helper = threading.Thread(target=_hold_callback_lock)
    helper.start()
    try:
        assert executor.run_sync(_work, timeout=1.0) == "done"
        assert lock_held.is_set()
        assert executor._slots.acquire(blocking=False)
        executor._slots.release()
    finally:
        release_work.set()
        release_lock.set()
        helper.join(timeout=1.0)
        executor.close()


def test_invalid_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        AsyncWorkExecutor(name="invalid", max_workers=0)
