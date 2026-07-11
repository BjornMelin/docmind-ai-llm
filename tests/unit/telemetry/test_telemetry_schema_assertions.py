"""Tests for telemetry schema assertions and JSONL output validation."""

import json
from pathlib import Path

import pytest

from src.config.settings import settings
from src.utils import telemetry

pytestmark = pytest.mark.unit


def test_server_hybrid_canonical_keys_present(
    tmp_path: Path, reset_settings_after_test: None
) -> None:
    out = tmp_path / "telemetry.jsonl"
    settings.telemetry.jsonl_path = out
    settings.telemetry.disabled = False
    settings.telemetry.sample = 1.0
    telemetry.log_jsonl(
        {
            "retrieval.fusion_mode": "rrf",
            "retrieval.prefetch_dense_limit": 200,
            "retrieval.prefetch_sparse_limit": 400,
            "retrieval.fused_limit": 60,
            "retrieval.return_count": 10,
            "retrieval.latency_ms": 12,
            "dedup.key": "page_id",
            "dedup.group_size": 1,
            "dedup.server_group_count": 10,
            "dedup.server_side": True,
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
    assert e["dedup.key"] == "page_id"
    assert e["dedup.group_size"] == 1
    assert e["dedup.server_group_count"] == 10
    assert e["dedup.server_side"] is True
    # Rerank keys
    assert e["rerank.stage"] in {"text", "visual", "colpali", "final"}
    assert isinstance(e["rerank.topk"], int)
    assert isinstance(e["rerank.timeout"], bool)
    # Optional extended keys when present
    if e["rerank.stage"] == "text":
        assert isinstance(e.get("rerank.batch_size", 0), int)
        assert isinstance(e.get("rerank.processed_count", 0), int)
        assert isinstance(e.get("rerank.processed_batches", 0), int)


def test_multimodal_client_dedup_keys_present(
    tmp_path: Path, reset_settings_after_test: None
) -> None:
    """Validate the client dedup stage reports real cardinality changes."""
    out = tmp_path / "telemetry.jsonl"
    settings.telemetry.jsonl_path = out
    settings.telemetry.disabled = False
    settings.telemetry.sample = 1.0
    telemetry.log_jsonl(
        {
            "retrieval.multimodal": True,
            "dedup.key": "page_id",
            "dedup.before": 12,
            "dedup.after": 9,
            "dedup.dropped": 3,
        }
    )

    event = json.loads(out.read_text().splitlines()[-1])
    assert event["retrieval.multimodal"] is True
    assert event["dedup.key"] == "page_id"
    assert event["dedup.before"] == 12
    assert event["dedup.after"] == 9
    assert event["dedup.dropped"] == 3
    assert event["dedup.before"] - event["dedup.after"] == event["dedup.dropped"]


def test_log_jsonl_writes_event(
    tmp_path: Path, reset_settings_after_test: None
) -> None:
    """Test that log_jsonl writes event data to JSONL file."""
    out = tmp_path / "telemetry.jsonl"
    settings.telemetry.jsonl_path = out
    settings.telemetry.disabled = False
    settings.telemetry.sample = 1.0
    telemetry.log_jsonl({"retrieval.fusion_mode": "rrf", "retrieval.latency_ms": 12})
    assert out.exists()
    data = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    assert data
    assert data[0]["retrieval.fusion_mode"] == "rrf"
