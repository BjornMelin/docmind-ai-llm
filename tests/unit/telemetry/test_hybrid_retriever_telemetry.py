import pytest


@pytest.mark.unit
def test_telemetry_fields_emitted(monkeypatch):
    from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams

    class _Res:
        def __init__(self):
            self.points = []

    class _FakeClient:
        def __init__(self, **_kwargs):
            pass

        def query_points(self, **_kwargs):
            return _Res()

    events = []

    def _capture(ev: dict):
        events.append(ev)

    monkeypatch.setattr("src.retrieval.hybrid.QdrantClient", _FakeClient)
    # Capture telemetry sink
    monkeypatch.setattr("src.retrieval.hybrid.log_jsonl", _capture)

    retr = ServerHybridRetriever(
        _HybridParams(
            collection="test",
            fused_top_k=5,
            prefetch_sparse=4,
            prefetch_dense=3,
            fusion_mode="rrf",
            dedup_key="page_id",
        )
    )
    monkeypatch.setattr(retr, "_embed_dense", lambda _t: [0.1, 0.2])
    monkeypatch.setattr(retr, "_encode_sparse", lambda _t: None)

    _ = retr.retrieve("q")

    assert events, "expected telemetry event"
    rec = events[-1]
    # Required keys
    for k in [
        "retrieval.fusion_mode",
        "retrieval.prefetch_dense_limit",
        "retrieval.prefetch_sparse_limit",
        "retrieval.fused_limit",
        "retrieval.return_count",
        "retrieval.latency_ms",
        "retrieval.sparse_fallback",
        "dedup.key",
        "dedup.input_count",
        "dedup.unique_count",
        "dedup.duplicates_removed",
    ]:
        assert k in rec
    # Sanity checks
    assert rec["retrieval.fusion_mode"] == "rrf"
    assert rec["retrieval.prefetch_dense_limit"] == 3
    assert rec["retrieval.prefetch_sparse_limit"] == 4
    assert rec["retrieval.fused_limit"] == 5
