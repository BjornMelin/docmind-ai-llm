"""Bounded executor for retrieval and reranking CPU work."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from typing import Any, TypeVar

_T = TypeVar("_T")


class AsyncWorkCapacityError(RuntimeError):
    """Raised when all owned worker slots are still occupied."""


class AsyncWorkClosedError(RuntimeError):
    """Raised when work is submitted after executor shutdown."""


class AsyncWorkExecutor:
    """Run bounded synchronous retrieval work without using asyncio's pool."""

    def __init__(self, *, name: str, max_workers: int = 1) -> None:
        """Create an owned executor with no admission queue.

        Args:
            name: Worker thread name prefix.
            max_workers: Maximum admitted native calls.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be at least one")
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=name,
        )
        self._slots = threading.BoundedSemaphore(max_workers)
        self._lock = threading.Lock()
        self._futures: set[Future[Any]] = set()
        self._closed = False

    async def run(
        self,
        function: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        """Run one function while retaining capacity until its thread exits."""
        return await asyncio.wrap_future(self._submit(function, *args, **kwargs))

    def run_sync(
        self,
        function: Callable[..., _T],
        /,
        *args: Any,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> _T:
        """Run one function synchronously with the same retained capacity."""
        return self._submit(function, *args, **kwargs).result(timeout=timeout)

    def _submit(
        self,
        function: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Future[_T]:
        """Admit work only when a native worker is immediately available."""
        if not self._slots.acquire(blocking=False):
            raise AsyncWorkCapacityError("Owned CPU executor is at capacity")

        with self._lock:
            if self._closed:
                self._slots.release()
                raise AsyncWorkClosedError("Owned CPU executor is closed")

            def _run_with_slot_release() -> _T:
                try:
                    return function(*args, **kwargs)
                finally:
                    self._slots.release()

            try:
                future = self._executor.submit(_run_with_slot_release)
            except BaseException:
                self._slots.release()
                raise
            self._futures.add(future)

        def _discard(completed: Future[Any]) -> None:
            with self._lock:
                self._futures.discard(completed)
            if completed.cancelled():
                self._slots.release()

        future.add_done_callback(_discard)
        return future

    def close(self) -> None:
        """Reject new work and cancel work that has not started."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def aclose(self) -> None:
        """Close and wait for already-running work to leave its thread."""
        self.close()
        with self._lock:
            futures = tuple(self._futures)
        for future in futures:
            with suppress(Exception):
                await asyncio.shield(asyncio.wrap_future(future))


__all__ = [
    "AsyncWorkCapacityError",
    "AsyncWorkClosedError",
    "AsyncWorkExecutor",
]
