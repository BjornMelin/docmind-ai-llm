"""Tests for telemetry schema assertions and JSONL output validation."""

import json

from src.utils import telemetry


def test_canonical_keys_present(tmp_path, monkeypatch):
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(telemetry, "_TELEM_PATH", out, raising=False)
    telemetry.log_jsonl(
        {
            "retrieval.fusion_mode": "rrf",
            "retrieval.prefetch_dense_limit": 200,
            "retrieval.prefetch_sparse_limit": 400,
            "retrieval.fused_limit": 60,
            "retrieval.return_count": 10,
            "retrieval.latency_ms": 12,
            "dedup.before": 12,
            "dedup.after": 10,
            "dedup.dropped": 2,
            "dedup.key": "page_id",
            "rerank.stage": "text",
            "rerank.topk": 5,
            "rerank.latency_ms": 5,
            "rerank.timeout": False,
            "rerank.batch_size": 4,
            "rerank.processed_count": 12,
            "rerank.processed_batches": 3,
        }
    )
    data = [
        __import__("json").loads(line)
        for line in out.read_text().splitlines()
        if line.strip()
    ]
    assert data
    e = data[-1]
    # Retrieval keys
    assert isinstance(e["retrieval.fusion_mode"], str)
    assert isinstance(e["retrieval.prefetch_dense_limit"], int)
    assert isinstance(e["retrieval.prefetch_sparse_limit"], int)
    assert isinstance(e["retrieval.fused_limit"], int)
    assert isinstance(e["retrieval.return_count"], int)
    assert isinstance(e["retrieval.latency_ms"], int)
    # Dedup keys
    assert e["dedup.before"] >= e["dedup.after"]
    assert e["dedup.key"] == "page_id"
    # Rerank keys
    assert e["rerank.stage"] in {"text", "visual", "colpali", "final"}
    assert isinstance(e["rerank.topk"], int)
    assert isinstance(e["rerank.timeout"], bool)
    # Optional extended keys when present
    if e["rerank.stage"] == "text":
        assert isinstance(e.get("rerank.batch_size", 0), int)
        assert isinstance(e.get("rerank.processed_count", 0), int)
        assert isinstance(e.get("rerank.processed_batches", 0), int)


def test_log_jsonl_writes_event(tmp_path, monkeypatch):
    """Test that log_jsonl writes event data to JSONL file."""
    out = tmp_path / "telemetry.jsonl"
    monkeypatch.setattr(telemetry, "_TELEM_PATH", out, raising=False)
    telemetry.log_jsonl({"retrieval.fusion_mode": "rrf", "retrieval.latency_ms": 12})
    assert out.exists()
    data = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert data
    assert data[0]["retrieval.fusion_mode"] == "rrf"
