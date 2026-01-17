"""Integration tests for semantic cache wiring in the coordinator (SPEC-038)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import AgentResponse
from src.config.settings import settings
from src.persistence.langchain_embeddings import LlamaIndexEmbeddingsAdapter

pytestmark = pytest.mark.integration


def test_coordinator_semantic_cache_hit_short_circuits(
    monkeypatch, tmp_path: Path
) -> None:
    # Configure settings for deterministic, in-memory cache.
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "data_dir", tmp_path, raising=False)

    monkeypatch.setattr(settings.semantic_cache, "enabled", True, raising=False)
    monkeypatch.setattr(settings.semantic_cache, "provider", "qdrant", raising=False)
    monkeypatch.setattr(
        settings.semantic_cache, "collection_name", "test_semcache", raising=False
    )
    monkeypatch.setattr(settings.semantic_cache, "namespace", "default", raising=False)
    monkeypatch.setattr(settings.semantic_cache, "ttl_seconds", 3600, raising=False)
    monkeypatch.setattr(
        settings.semantic_cache, "max_response_bytes", 10_000, raising=False
    )
    monkeypatch.setattr(settings.embedding, "dimension", 3, raising=False)

    # Deterministic embedding output.
    monkeypatch.setattr(
        LlamaIndexEmbeddingsAdapter,
        "embed_query",
        lambda self, _text: [1.0, 0.0, 0.0],
    )

    client = QdrantClient(location=":memory:")

    @contextmanager
    def _client_cm():  # type: ignore[no-untyped-def]
        yield client

    import src.utils.storage as storage_mod

    monkeypatch.setattr(storage_mod, "create_sync_client", _client_cm)

    coord = MultiAgentCoordinator(tool_registry=None, checkpointer=object(), store=None)

    # Avoid heavy setup/graph execution; force "fresh" response on first call.
    monkeypatch.setattr(coord, "_ensure_setup", lambda: True)
    monkeypatch.setattr(coord, "list_checkpoints", lambda **_k: [])
    monkeypatch.setattr(coord, "_run_agent_workflow", lambda *_a, **_k: {})
    monkeypatch.setattr(coord, "_schedule_memory_consolidation", lambda *_a, **_k: None)
    monkeypatch.setattr(coord, "_record_query_metrics", lambda *_a, **_k: None)

    def _handle_workflow_result(_result, _query, _context, _start, _coord_time):  # type: ignore[no-untyped-def]
        return (
            AgentResponse(
                content="fresh", sources=[], metadata={}, processing_time=0.0
            ),
            False,
            False,
        )

    monkeypatch.setattr(coord, "_handle_workflow_result", _handle_workflow_result)

    first = coord.process_query("hello", thread_id="t", user_id="u")
    assert first.content == "fresh"

    # Second call should hit cache and bypass workflow execution.
    monkeypatch.setattr(
        coord,
        "_run_agent_workflow",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("_run_agent_workflow should not be called on cache hit")
        ),
    )
    second = coord.process_query("hello", thread_id="t", user_id="u")
    assert second.content == "fresh"
    assert isinstance(second.metadata, dict)
    assert second.metadata.get("semantic_cache", {}).get("hit") is True
