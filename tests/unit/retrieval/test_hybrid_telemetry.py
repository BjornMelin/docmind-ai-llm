"""Unit test for hybrid retrieval telemetry enrichment.

Creates a ServerHybridRetriever with a fake Qdrant client to trigger the
telemetry emission and asserts required keys are present in the captured
events. Avoids real Qdrant network calls.
"""

from __future__ import annotations

import importlib
from typing import Any


def test_hybrid_retrieval_telemetry_emits_backend_and_timeout(monkeypatch):  # type: ignore[no-untyped-def]
    """Validate backend, timeout, fusion params, and counts are logged."""
    hmod = importlib.import_module("src.retrieval.hybrid")

    # Patch settings for deterministic rrf_k and timeout
    from src.config import settings as cfg  # lazy import for test

    monkeypatch.setattr(cfg.retrieval, "fusion_mode", "rrf", raising=False)
    monkeypatch.setattr(cfg.retrieval, "fused_top_k", 5, raising=False)
    monkeypatch.setattr(cfg.retrieval, "prefetch_dense_limit", 10, raising=False)
    monkeypatch.setattr(cfg.retrieval, "prefetch_sparse_limit", 10, raising=False)
    monkeypatch.setattr(cfg.retrieval, "dedup_key", "page_id", raising=False)
    monkeypatch.setattr(cfg.retrieval, "rrf_k", 60, raising=False)
    monkeypatch.setattr(cfg.database, "qdrant_timeout", 42, raising=False)

    # Fake client and result
    class _Point:
        def __init__(self, i: int):
            self.id = i
            self.score = 0.9 - (i * 0.01)
            self.payload = {
                "page_id": f"p{i}",
                "text": f"chunk {i}",
            }

    class _Res:
        def __init__(self):
            self.points = [_Point(i) for i in range(8)]

    class _Client:
        def query_points(self, **_kwargs: Any):  # type: ignore[no-untyped-def]
            return _Res()

        def close(self):  # type: ignore[no-untyped-def]
            return None

    # Capture telemetry
    events: list[dict[str, Any]] = []

    def _cap(evt: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        events.append(evt)

    monkeypatch.setattr(hmod, "log_jsonl", _cap)

    class _Embed:
        def get_text_embedding(self, text: str) -> list[float]:  # type: ignore[no-untyped-def]
            return [0.1, 0.2]

        def get_query_embedding(self, text: str) -> list[float]:  # type: ignore[no-untyped-def]
            return [0.1, 0.2]

    monkeypatch.setattr(hmod, "get_settings_embed_model", lambda: _Embed())

    params = hmod._HybridParams(  # pylint: disable=protected-access
        collection="c",
        fused_top_k=5,
        prefetch_sparse=10,
        prefetch_dense=10,
        fusion_mode="rrf",
        dedup_key="page_id",
    )
    retr = hmod.ServerHybridRetriever(params, client=_Client())
    out = retr.retrieve("q")
    assert isinstance(out, list)

    # Check last emitted telemetry event
    assert events, "No telemetry events captured"
    e = events[-1]
    assert e.get("retrieval.backend") == "qdrant"
    assert e.get("retrieval.qdrant_timeout_s") == 42
    assert e.get("retrieval.fusion_mode") == "rrf"
    assert isinstance(e.get("retrieval.return_count"), int)
    assert isinstance(e.get("retrieval.latency_ms"), int)
