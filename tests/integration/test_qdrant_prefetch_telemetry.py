"""Integration-style test for Qdrant Prefetch+FusionQuery telemetry.

Mocks the Qdrant client to verify that hybrid retrieval emits minimal telemetry
fields expected by downstream observability. Marked as integration to reflect
component interaction (retriever + settings + telemetry).
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest


@pytest.mark.integration
@pytest.mark.retrieval
def test_qdrant_prefetch_fusionquery_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify hybrid retrieval emits correct telemetry with Qdrant prefetch."""
    hmod = importlib.import_module("src.retrieval.hybrid")

    # Minimal fake Qdrant response
    class _Point:
        """Minimal Qdrant point stub with predictable payload."""

        def __init__(self, i: int) -> None:
            self.id = i
            self.score = 0.9 - (i * 0.01)
            self.payload = {"page_id": f"p{i}", "text": f"chunk {i}"}

    class _Res:
        """Container for the stubbed Qdrant response."""

        def __init__(self) -> None:
            self.points = [_Point(i) for i in range(6)]

    class _Client:
        """Stubbed Qdrant client capturing query parameters."""

        def query_points(self, **_kwargs: Any) -> _Res:
            """Execute a query and verify prefetch and limit parameters."""
            assert "prefetch" in _kwargs
            assert "limit" in _kwargs
            return _Res()

        def close(self) -> None:
            """Close the client connection."""
            return None

    # Capture telemetry
    events: list[dict[str, Any]] = []

    def _cap(evt: dict[str, Any]) -> None:
        events.append(evt)

    monkeypatch.setattr(hmod, "log_jsonl", _cap)

    params = hmod.HybridParams(
        collection="c",
        fused_top_k=5,
        prefetch_sparse=8,
        prefetch_dense=8,
        fusion_mode="rrf",
        dedup_key="page_id",
    )
    retr = hmod.ServerHybridRetriever(params, client=_Client())
    out = retr.retrieve("query")
    assert out is not None
    assert len(out) <= 5
    assert events, "Expected telemetry events"
    e = events[-1]
    # Minimal keys
    assert e.get("retrieval.backend") == "qdrant"
    assert e.get("retrieval.fusion_mode") in {"rrf", "dbsf"}
    assert isinstance(e.get("retrieval.return_count"), int)
    assert isinstance(e.get("retrieval.latency_ms"), int)
