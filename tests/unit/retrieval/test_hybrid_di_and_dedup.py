"""Hybrid retriever tests for DI and dedup/sort behavior.

Inject a fake Qdrant client and validate deterministic deduplication and
ordering by score then id. Avoids any live DB calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.retrieval.hybrid import HybridParams, ServerHybridRetriever


class _Point:
    def __init__(self, pid: str, score: float, payload: dict[str, Any]):
        self.id = pid
        self.score = score
        self.payload = payload


class _ClientFake:
    def __init__(self, points: list[_Point]) -> None:
        self._points = points
        self.closed = False

    def query_points(self, **_kwargs: Any) -> Any:  # pragma: no cover - simple
        return type("_R", (), {"points": self._points})()

    def close(self) -> None:  # pragma: no cover - trivial
        self.closed = True


@pytest.mark.unit
def test_hybrid_di_and_dedup_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    # Avoid real embedding backends by patching retriever methods

    # Construct points with duplicate page_id and mixed scores
    pts = [
        _Point("a", 0.9, {"page_id": "p1", "text": "t1"}),
        _Point(
            "b", 0.95, {"page_id": "p1", "text": "t1b"}
        ),  # duplicate key; higher score wins
        _Point("c", 0.5, {"page_id": "p2", "text": "t2"}),
    ]

    client = _ClientFake(pts)
    retr = ServerHybridRetriever(HybridParams(collection="col"), client=client)
    retr._embed_dense = (  # type: ignore[attr-defined]
        lambda _t: np.asarray([0.0, 1.0], dtype=np.float32)
    )
    retr._encode_sparse = lambda _t: None  # type: ignore[attr-defined]
    out = retr.retrieve("hello world")
    # Dedup keeps one for p1 and c remains; total 2
    assert len(out) == 2
    # Ensure order by score desc then id
    assert out[0].score >= out[1].score
    retr.close()
    assert client.closed is True
