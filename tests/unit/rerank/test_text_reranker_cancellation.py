"""Cancellable text reranker: early-exit returns deterministic subset.

These tests exercise the cooperative, batch-wise cancellation logic in the
unified text reranker adapter by simulating elapsed time after one batch.
"""

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import reranking as rr


def _mk_nodes(n: int) -> list[NodeWithScore]:
    return [NodeWithScore(node=TextNode(text=f"t-{i}"), score=0.0) for i in range(n)]


def test_cancellation_returns_full_batches_only(monkeypatch):
    nodes = _mk_nodes(10)

    # Pretend FlagEmbedding is available and inject a fast scoring model
    class _FastModel:
        def compute_score(self, pairs):
            # Deterministic increasing scores within batch
            return [float(i) for i in range(len(pairs))]

    def ensure_loaded(self):
        self._model = _FastModel()  # type: ignore[attr-defined]

    monkeypatch.setattr(rr._FlagTextReranker, "_ensure_loaded", ensure_loaded)

    # Simulate time crossing budget right after first batch completes
    start = rr._now_ms()
    calls = {"n": 0}

    def fake_now():
        calls["n"] += 1
        # Advance time after ~one batch worth of work
        return start if calls["n"] <= 4 else (start + rr.TEXT_RERANK_TIMEOUT_MS + 1)

    monkeypatch.setattr(rr, "_now_ms", fake_now)

    adapter = rr.build_text_reranker(
        top_n=10, batch_size=4, timeout_ms=rr.TEXT_RERANK_TIMEOUT_MS
    )
    out = adapter.postprocess_nodes(nodes, query_str="q")

    # Only the first full batch (size 4) should be processed
    assert len(out) == 4
    # Telemetry stats recorded on adapter
    assert adapter.last_stats["timeout"] is True
    assert adapter.last_stats["processed_batches"] == 1
    assert adapter.last_stats["processed_count"] == 4
