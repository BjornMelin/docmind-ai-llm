"""Unit tests for Qdrant-backed semantic cache (SPEC-038)."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from src.config.settings import SemanticCacheConfig
from src.utils.semantic_cache import SemanticCache, build_cache_key, build_prompt_key

pytestmark = pytest.mark.unit


def _cfg(**overrides: object) -> SemanticCacheConfig:
    values: dict[str, object] = {
        "enabled": True,
        "provider": "qdrant",
        "collection_name": "test_semcache",
        "score_threshold": 0.85,
        "ttl_seconds": 3600,
        "top_k": 5,
        "max_response_bytes": 1000,
        "namespace": "default",
        "allow_semantic_for_templates": None,
    }
    values.update(overrides)
    return SemanticCacheConfig(**values)


def test_prompt_key_is_stable() -> None:
    a = build_prompt_key({"a": 1, "b": 2})
    b = build_prompt_key({"b": 2, "a": 1})
    assert a == b
    c = build_prompt_key({"a": 1, "b": 3})
    assert a != c


def test_exact_hit_and_invalidation() -> None:
    client = QdrantClient(location=":memory:")

    embed = lambda _q: [1.0, 0.0, 0.0]  # noqa: E731
    cache = SemanticCache(client=client, cfg=_cfg(), vector_dim=3, embed_query=embed)

    key = build_cache_key(
        query="hello",
        namespace="default",
        model_id="m",
        template_id="chat",
        template_version="1",
        temperature=0.0,
        corpus_hash="c1",
        config_hash="k1",
    )
    cache.store(key=key, query="hello", response_text="world")

    hit = cache.lookup(key=key, query="hello")
    assert hit is not None
    assert hit.kind == "exact"
    assert hit.response_text == "world"

    invalid = build_cache_key(
        query="hello",
        namespace="default",
        model_id="m",
        template_id="chat",
        template_version="1",
        temperature=0.0,
        corpus_hash="c2",
        config_hash="k1",
    )
    assert cache.lookup(key=invalid, query="hello") is None


def test_semantic_hit_requires_threshold() -> None:
    client = QdrantClient(location=":memory:")

    # Constant embedding => cosine similarity 1.0.
    embed = lambda _q: [1.0, 0.0, 0.0]  # noqa: E731
    cache = SemanticCache(
        client=client,
        cfg=_cfg(score_threshold=0.99),
        vector_dim=3,
        embed_query=embed,
    )

    key_a = build_cache_key(
        query="q1",
        namespace="default",
        model_id="m",
        template_id="chat",
        template_version="1",
        temperature=0.0,
        corpus_hash="c1",
        config_hash="k1",
    )
    cache.store(key=key_a, query="q1", response_text="answer-a")

    # Different prompt_key, same metadata => exact miss but semantic hit.
    key_b = build_cache_key(
        query="q2",
        namespace="default",
        model_id="m",
        template_id="chat",
        template_version="1",
        temperature=0.0,
        corpus_hash="c1",
        config_hash="k1",
    )
    hit = cache.lookup(key=key_b, query="q2")
    assert hit is not None
    assert hit.kind == "semantic"
    assert hit.response_text == "answer-a"
