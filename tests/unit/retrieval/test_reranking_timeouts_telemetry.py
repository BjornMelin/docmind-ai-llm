"""Unit tests for reranking timeouts and telemetry emission.

Validates that the reranking module respects settings-driven timeout values and
emits the expected telemetry keys for the text and final stages.
"""

from __future__ import annotations

import importlib
from typing import Any


def test_reranking_timeouts_and_final_telemetry(monkeypatch):  # type: ignore[no-untyped-def]
    """Ensure settings-driven timeouts are respected and telemetry includes keys.

    The test avoids heavy dependencies by stubbing out the text reranker
    and emitting telemetry into an in-memory list.
    """
    # Import module under test lazily to ensure monkeypatch works
    rmod = importlib.import_module("src.retrieval.reranking")

    # Patch settings-driven timeouts
    monkeypatch.setattr(
        rmod.settings.retrieval, "text_rerank_timeout_ms", 321, raising=False
    )
    monkeypatch.setattr(
        rmod.settings.retrieval, "siglip_timeout_ms", 210, raising=False
    )
    monkeypatch.setattr(
        rmod.settings.retrieval, "colpali_timeout_ms", 432, raising=False
    )
    monkeypatch.setattr(
        rmod.settings.retrieval, "total_rerank_budget_ms", 999, raising=False
    )
    monkeypatch.setattr(rmod.settings.retrieval, "reranking_top_k", 3, raising=False)

    # Capture telemetry events
    events: list[dict[str, Any]] = []

    def _capture(evt: dict[str, Any]) -> None:  # type: ignore[no-untyped-def]
        events.append(evt)

    monkeypatch.setattr(rmod, "log_jsonl", _capture)

    # Stub build_text_reranker to avoid loading real models
    class _StubTextRerank:
        def __init__(self, top_n: int):
            self.top_n = top_n

        def postprocess_nodes(self, nodes, query_str: str):  # type: ignore[no-untyped-def]
            # Return at most top_n nodes unchanged
            return nodes[: self.top_n]

    def _stub_build_text_reranker(top_n: int):  # type: ignore[no-untyped-def]
        return _StubTextRerank(top_n)

    monkeypatch.setattr(rmod, "build_text_reranker", _stub_build_text_reranker)

    # Minimal node shims (text modality by default)
    class _Node:
        def __init__(self, node_id: str):
            self.node_id = node_id
            self.metadata = {"modality": "text"}

    class _NWS:
        def __init__(self, i: int):
            self.node = _Node(f"n{i}")
            self.score = 0.1 * i

    nodes = [_NWS(i) for i in range(5)]

    # Query bundle shim
    class _QB:
        def __init__(self, q: str):
            self.query_str = q

    qb = _QB("query")

    # Execute reranking
    rr = rmod.MultimodalReranker()
    out = rr._postprocess_nodes(nodes, qb)  # pylint: disable=protected-access
    assert isinstance(out, list)
    assert len(out) == rmod.settings.retrieval.reranking_top_k

    # Inspect captured telemetry
    # Expect at least a 'text' stage and a 'final' stage
    stages = {e.get("rerank.stage") for e in events if "rerank.stage" in e}
    assert "text" in stages
    assert "final" in stages
    final = next(e for e in events if e.get("rerank.stage") == "final")
    assert final.get("rerank.total_timeout_budget_ms") == 999
    assert isinstance(final.get("rerank.input_count"), int)
    assert isinstance(final.get("rerank.output_count"), int)
    assert "rerank.executor" in final
