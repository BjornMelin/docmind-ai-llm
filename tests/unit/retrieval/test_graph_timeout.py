"""Unit tests for graph traversal timeout handling.

Verifies that retrieve_graph_nodes returns an empty list when the underlying
retriever exceeds the timeout and that it does not raise.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.retrieval.graph_config import traverse_graph


class _SlowRetriever:
    def __init__(self, delay: float) -> None:
        self.delay = delay

    def retrieve(
        self, _query: str
    ) -> list[Any]:  # pragma: no cover - exercised via thread
        import time

        time.sleep(self.delay)
        return ["late"]


class _IndexStub:
    def __init__(self, delay: float) -> None:
        self._delay = delay

    def as_retriever(
        self, include_text: bool = False, path_depth: int = 1
    ):  # pragma: no cover - trivial
        assert include_text is False
        assert path_depth >= 0
        return _SlowRetriever(self._delay)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_traverse_graph_timeout_returns_empty() -> None:
    index = _IndexStub(delay=0.2)
    out = await traverse_graph(index, "q", max_depth=1, timeout=0.05)
    assert out == []
