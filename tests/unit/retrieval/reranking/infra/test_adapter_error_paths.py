"""Reranker adapter error paths test ensures fail-open behavior."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _nws(nid: str, s: float = 0.0):
    return SimpleNamespace(
        node=SimpleNamespace(node_id=nid, text="t", metadata={}), score=s
    )


def test_siglip_fail_open_on_timeout(monkeypatch):
    # Patch _now_ms to force timeout immediately and avoid heavy model loads
    import src.retrieval.reranking as rr

    clock = {"t": 1000.0}

    def _tick() -> float:
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(rr, "_now_ms", _tick)
    monkeypatch.setattr(rr, "_load_images_for_siglip", lambda *_a, **_k: [])

    # Make processing always think timeout elapsed
    nodes = [_nws("x"), _nws("y")]
    out = rr._siglip_rescore("q", nodes, budget_ms=0)
    # Fail-open returns input list unchanged length
    assert len(out) == len(nodes)
