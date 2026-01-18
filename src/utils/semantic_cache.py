"""Semantic response cache (Qdrant-backed, guardrailed).

Implements SPEC-038 with strict invalidation by:
- corpus_hash (uploaded corpus files)
- config_hash (retrieval/ingestion knobs)
- model + prompt template + params signature

Security posture:
- Never stores raw prompt text (only an HMAC-free hash key + embedding vector).
- Does not log prompts or responses (telemetry is metadata-only).
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from grpc import RpcError
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config.settings import SemanticCacheConfig
from src.persistence.hashing import compute_config_hash

_DEFAULT_COLLECTION = "docmind_semcache"


@dataclass(frozen=True, slots=True)
class CacheKey:
    """Metadata key used for strict invalidation and filtering.

    Args:
        prompt_key: Stable HMAC-free hash of the prompt and parameters.
        namespace: Cache namespace for isolation.
        model_id: Identifier for the LLM model.
        template_id: Prompt template identifier.
        template_version: Prompt template version identifier.
        temperature: LLM temperature setting.
        corpus_hash: Hash of the uploaded corpus files.
        config_hash: Hash of retrieval/ingestion configuration.

    Returns:
        A CacheKey instance used for strict invalidation and filtering.
    """

    prompt_key: str
    namespace: str
    model_id: str
    template_id: str
    template_version: str
    temperature: float
    corpus_hash: str
    config_hash: str


@dataclass(frozen=True, slots=True)
class CacheHit:
    """Cache lookup result.

    Args:
        kind: Type of match ("exact" or "semantic").
        score: Similarity score for semantic matches (None for exact).
        response_text: The cached response string.

    Returns:
        A CacheHit instance representing a successful lookup.
    """

    kind: str  # "exact" | "semantic"
    score: float | None
    response_text: str


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_prompt_key(payload: dict[str, Any]) -> str:
    """Return a stable prompt key for exact-match lookups.

    Args:
        payload: Canonicalizable payload used to build the cache key.

    Returns:
        Stable SHA-256 hex digest for exact-match lookups.
    """
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_hex(canonical)


def build_cache_key(
    *,
    query: str,
    namespace: str,
    model_id: str,
    template_id: str,
    template_version: str,
    temperature: float,
    corpus_hash: str,
    config_hash: str,
) -> CacheKey:
    """Build a strict cache key without persisting raw prompt text.

    Args:
        query: User query text used in the prompt.
        namespace: Cache namespace for isolation.
        model_id: Identifier for the LLM model.
        template_id: Prompt template identifier.
        template_version: Prompt template version identifier.
        temperature: LLM temperature setting.
        corpus_hash: Hash of the uploaded corpus.
        config_hash: Hash of retrieval/ingestion config.

    Returns:
        CacheKey with all strict invalidation fields populated.
    """
    prompt_key = build_prompt_key(
        {
            "query": str(query),
            "namespace": str(namespace),
            "model_id": str(model_id),
            "template_id": str(template_id),
            "template_version": str(template_version),
            "temperature": float(temperature),
            "corpus_hash": str(corpus_hash),
            "config_hash": str(config_hash),
        }
    )
    return CacheKey(
        prompt_key=prompt_key,
        namespace=str(namespace),
        model_id=str(model_id),
        template_id=str(template_id),
        template_version=str(template_version),
        temperature=float(temperature),
        corpus_hash=str(corpus_hash),
        config_hash=str(config_hash),
    )


def _point_id_for_prompt(prompt_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"docmind:semcache:{prompt_key}"))


def _now_epoch_s() -> float:
    return float(time.time())


def _expires_at_epoch_s(ttl_seconds: int) -> float:
    return _now_epoch_s() + max(0, int(ttl_seconds))


def _collection_name(cfg: SemanticCacheConfig) -> str:
    name = (cfg.collection_name or "").strip()
    return name or _DEFAULT_COLLECTION


def _must_filters(
    key: CacheKey, *, require_prompt_key: bool
) -> list[qmodels.Condition]:
    must: list[qmodels.Condition] = []
    if require_prompt_key:
        must.append(
            qmodels.FieldCondition(
                key="prompt_key",
                match=qmodels.MatchValue(value=key.prompt_key),
            )
        )
    must.extend(
        [
            qmodels.FieldCondition(
                key="namespace",
                match=qmodels.MatchValue(value=key.namespace),
            ),
            qmodels.FieldCondition(
                key="model_id",
                match=qmodels.MatchValue(value=key.model_id),
            ),
            qmodels.FieldCondition(
                key="template_id",
                match=qmodels.MatchValue(value=key.template_id),
            ),
            qmodels.FieldCondition(
                key="template_version",
                match=qmodels.MatchValue(value=key.template_version),
            ),
            qmodels.FieldCondition(
                key="temperature",
                range=qmodels.Range(gte=key.temperature, lte=key.temperature),
            ),
            qmodels.FieldCondition(
                key="corpus_hash",
                match=qmodels.MatchValue(value=key.corpus_hash),
            ),
            qmodels.FieldCondition(
                key="config_hash",
                match=qmodels.MatchValue(value=key.config_hash),
            ),
            qmodels.FieldCondition(
                key="expires_at",
                range=qmodels.Range(gte=_now_epoch_s()),
            ),
        ]
    )
    return must


def _payload_from_hit(hit: qmodels.Record | qmodels.ScoredPoint) -> dict[str, Any]:
    payload = getattr(hit, "payload", None)
    if isinstance(payload, dict):
        return payload
    return {}


def _safe_response_from_payload(payload: dict[str, Any]) -> str | None:
    value = payload.get("response_text")
    if isinstance(value, str) and value:
        return value
    return None


def _allow_semantic(cfg: SemanticCacheConfig, *, template_id: str) -> bool:
    allowlist = cfg.allow_semantic_for_templates
    if allowlist is None:
        return True
    return str(template_id) in {str(x) for x in allowlist}


class SemanticCache:
    """Semantic cache backed by a Qdrant collection."""

    def __init__(
        self,
        *,
        client: QdrantClient,
        cfg: SemanticCacheConfig,
        vector_dim: int,
        embed_query: Callable[[str], list[float]],
    ) -> None:
        """Initialize the semantic cache.

        Args:
            client: Qdrant client instance.
            cfg: Semantic cache configuration.
            vector_dim: Embedding dimension for query vectors.
            embed_query: Callable to embed query strings.

        Returns:
            None.
        """
        self._client = client
        self._cfg = cfg
        self._collection = _collection_name(cfg)
        self._vector_dim = int(vector_dim)
        self._embed_query = embed_query

    def ensure_ready(self) -> None:
        """Create the collection if it does not exist."""
        if self._client.collection_exists(self._collection):
            return
        try:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        except (UnexpectedResponse, RpcError):
            if self._client.collection_exists(self._collection):
                return
            raise

    def lookup(self, *, key: CacheKey, query: str) -> CacheHit | None:
        """Attempt exact-match then semantic lookup; returns None on miss."""
        if not self._cfg.enabled or self._cfg.provider != "qdrant":
            return None

        self.ensure_ready()
        start = time.perf_counter()

        # 1) Exact match by prompt_key.
        exact = self._exact_lookup(key)
        if exact is not None:
            self._log_event("semantic_cache_hit", kind="exact", score=None, start=start)
            return exact

        if not _allow_semantic(self._cfg, template_id=key.template_id):
            self._log_event("semantic_cache_miss", kind="none", score=None, start=start)
            return None

        # 2) Semantic match by vector + strict metadata filter.
        semantic = self._semantic_lookup(key, query=query)
        if semantic is not None:
            self._log_event(
                "semantic_cache_hit",
                kind="semantic",
                score=semantic.score,
                start=start,
            )
            return semantic

        self._log_event("semantic_cache_miss", kind="none", score=None, start=start)
        return None

    def store(self, *, key: CacheKey, query: str, response_text: str) -> None:
        """Store a response when enabled and within size limits."""
        if not self._cfg.enabled or self._cfg.provider != "qdrant":
            return
        if not isinstance(response_text, str) or not response_text.strip():
            return

        response_bytes = len(response_text.encode("utf-8"))
        if response_bytes > int(self._cfg.max_response_bytes):
            return

        self.ensure_ready()
        start = time.perf_counter()
        vec = self._embed_query(str(query))
        if len(vec) != self._vector_dim:
            raise ValueError(
                f"Semantic cache embedding dimension mismatch: got={len(vec)} "
                f"expected={self._vector_dim}"
            )

        payload: dict[str, Any] = {
            "prompt_key": key.prompt_key,
            "namespace": key.namespace,
            "model_id": key.model_id,
            "template_id": key.template_id,
            "template_version": key.template_version,
            "temperature": float(key.temperature),
            "corpus_hash": key.corpus_hash,
            "config_hash": key.config_hash,
            "created_at": _now_epoch_s(),
            "expires_at": _expires_at_epoch_s(int(self._cfg.ttl_seconds)),
            "response_text": response_text,
            "response_bytes": int(response_bytes),
        }

        self._client.upsert(
            collection_name=self._collection,
            points=[
                qmodels.PointStruct(
                    id=_point_id_for_prompt(key.prompt_key),
                    vector=vec,
                    payload=payload,
                )
            ],
        )
        self._log_event(
            "semantic_cache_store",
            kind="store",
            score=None,
            start=start,
            bytes_written=response_bytes,
        )

    def _exact_lookup(self, key: CacheKey) -> CacheHit | None:
        flt = qmodels.Filter(must=_must_filters(key, require_prompt_key=True))
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        payload = _payload_from_hit(points[0])
        resp = _safe_response_from_payload(payload)
        if resp is None:
            return None
        return CacheHit(kind="exact", score=None, response_text=resp)

    def _semantic_lookup(self, key: CacheKey, *, query: str) -> CacheHit | None:
        flt = qmodels.Filter(must=_must_filters(key, require_prompt_key=False))
        vec = self._embed_query(str(query))
        if len(vec) != self._vector_dim:
            return None
        response = self._client.query_points(
            collection_name=self._collection,
            query=vec,
            limit=int(self._cfg.top_k),
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
            score_threshold=float(self._cfg.score_threshold),
        )
        hits = list(getattr(response, "points", []) or [])
        if not hits:
            return None
        best = hits[0]
        score = float(getattr(best, "score", 0.0) or 0.0)
        payload = _payload_from_hit(best)
        resp = _safe_response_from_payload(payload)
        if resp is None:
            return None
        return CacheHit(kind="semantic", score=score, response_text=resp)

    def _log_event(
        self,
        event: str,
        *,
        kind: str,
        score: float | None,
        start: float,
        bytes_written: int | None = None,
    ) -> None:
        duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
        payload: dict[str, Any] = {
            "event": str(event),
            "kind": str(kind),
            "duration_ms": duration_ms,
        }
        if score is not None:
            payload["score"] = float(score)
        if bytes_written is not None:
            payload["bytes"] = int(bytes_written)
        try:
            from src.utils.telemetry import log_jsonl

            log_jsonl(payload)
        except Exception as exc:  # pragma: no cover - best effort
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="semantic_cache.telemetry")
            logger.debug(
                "semantic cache telemetry skipped (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )


def config_hash_for_semcache(settings_obj: Any) -> str:
    """Return the config hash used for semantic cache invalidation.

    Args:
        settings_obj: Settings object to hash.

    Returns:
        Stable hash string for semantic cache invalidation.
    """
    from src.persistence.snapshot_utils import current_config_dict

    return compute_config_hash(current_config_dict(settings_obj))


__all__ = [
    "CacheHit",
    "CacheKey",
    "SemanticCache",
    "build_cache_key",
    "build_prompt_key",
    "config_hash_for_semcache",
]
