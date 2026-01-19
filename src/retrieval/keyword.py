"""Keyword/lexical retrieval via sparse-only Qdrant Query API.

Implements SPEC-025 / ADR-044:
- Encode query text into a Qdrant ``SparseVector`` via FastEmbed (best-effort).
- Query Qdrant using named sparse vector ``text-sparse`` (sparse-only).
- Return ``NodeWithScore`` with deterministic ordering and PII-safe telemetry.

This retriever is intentionally minimal and fails open:
if sparse encoding or Qdrant query fails, it returns an empty list.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llama_index.core.schema import NodeWithScore, QueryBundle
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.common.client_exceptions import ResourceExhaustedResponse

from src.retrieval.sparse_query import encode_to_qdrant
from src.utils.log_safety import build_pii_log_entry
from src.utils.qdrant_exceptions import QDRANT_SCHEMA_EXCEPTIONS
from src.utils.qdrant_utils import QDRANT_PAYLOAD_FIELDS, nodes_from_query_result
from src.utils.storage import get_client_config
from src.utils.telemetry import log_jsonl

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qdrant_client.http.models import QueryResponse as QdrantQueryResponse


@dataclass(frozen=True)
class KeywordParams:
    """Configuration for sparse-only keyword retrieval."""

    collection: str
    top_k: int = 10
    using: str = "text-sparse"
    with_payload: tuple[str, ...] = QDRANT_PAYLOAD_FIELDS
    rate_limit_retries: int = 2
    rate_limit_backoff_base_s: float = 0.5
    rate_limit_backoff_max_s: float = 8.0


class KeywordSparseRetriever:
    """Sparse-only retriever for exact-term / keyword lookups (Qdrant)."""

    def __init__(
        self,
        params: KeywordParams,
        *,
        client: QdrantClient | None = None,
        client_factory: Callable[[], QdrantClient] | None = None,
    ) -> None:
        """Initialize retriever with query parameters and an optional client.

        Args:
            params: Keyword retrieval parameters (collection, top_k, payload fields).
            client: Optional pre-configured QdrantClient instance.
            client_factory: Optional factory to lazily construct a QdrantClient.
        """
        self.params = params
        cfg = get_client_config()
        self._client: QdrantClient | None = client
        self._client_factory = client_factory or (lambda: QdrantClient(**cfg))
        # Cache timeout for telemetry; keep consistent with configured client.
        try:
            self._qdrant_timeout_s = int(cfg.get("timeout", 60))
        except (TypeError, ValueError):  # pragma: no cover - defensive
            self._qdrant_timeout_s = 60

    def close(self) -> None:
        """Close the underlying client (best-effort)."""
        if self._client is None:
            return
        with suppress(Exception):  # pragma: no cover - defensive
            self._client.close()

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Retrieve nodes via sparse-only search against ``text-sparse``.

        Args:
            query: Query string or ``QueryBundle``.

        Returns:
            A list of ``NodeWithScore`` results (may be empty).
        """
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)
        t0 = time.time()

        sparse_vec = encode_to_qdrant(qtext)
        if sparse_vec is None:
            self._emit_telemetry(t0=t0, return_count=0, sparse_fallback=True)
            return []

        try:
            result = self._query_points_with_retry(sparse_vec)
        except (
            *QDRANT_SCHEMA_EXCEPTIONS,
            ResourceExhaustedResponse,
        ) as exc:
            logger.warning(
                "Qdrant keyword query failed (error_type=%s)", type(exc).__name__
            )
            self._emit_telemetry(
                t0=t0,
                return_count=0,
                sparse_fallback=False,
                error_type=type(exc).__name__,
            )
            return []

        nodes = nodes_from_query_result(
            result,
            top_k=int(self.params.top_k),
            id_keys=("chunk_id", "page_id", "doc_id"),
            prefer_point_id=True,
        )

        self._emit_telemetry(
            t0=t0, return_count=len(nodes), sparse_fallback=False, error_type=None
        )
        return nodes

    def _query_points_with_retry(self, sparse_vec: Any) -> QdrantQueryResponse:
        """Query Qdrant with rate-limit aware retries (best-effort)."""
        retries = max(0, int(self.params.rate_limit_retries))
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self._get_client().query_points(
                    collection_name=self.params.collection,
                    query=sparse_vec,
                    using=self.params.using,
                    limit=int(self.params.top_k),
                    with_payload=list(self.params.with_payload),
                )
            except ResourceExhaustedResponse as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                delay = self._rate_limit_delay(exc, attempt)
                time.sleep(delay)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Qdrant rate limit retry exhausted without exception")

    def _rate_limit_delay(self, exc: Exception, attempt: int) -> float:
        retry_after = getattr(exc, "retry_after", None)
        if retry_after is None:
            try:
                if len(exc.args) > 1:
                    retry_after = exc.args[1]
            except (AttributeError, TypeError):
                retry_after = None
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except (TypeError, ValueError):
                pass
        base = max(0.0, float(self.params.rate_limit_backoff_base_s))
        delay = min(
            base * (2**attempt),
            float(self.params.rate_limit_backoff_max_s),
        )
        # jitter in [0.5x, 1.0x]
        return delay * (0.5 + (random.random() * 0.5))  # noqa: S311

    def _emit_telemetry(
        self,
        *,
        t0: float,
        return_count: int,
        sparse_fallback: bool,
        error_type: str | None = None,
    ) -> None:
        """Emit PII-safe JSONL telemetry (no query text)."""
        try:
            latency_ms = int((time.time() - t0) * 1000)

            ev: dict[str, Any] = {
                "retrieval.backend": "qdrant",
                "retrieval.tool": "keyword_search",
                "retrieval.vector": self.params.using,
                "retrieval.top_k": int(self.params.top_k),
                "retrieval.return_count": int(return_count),
                "retrieval.latency_ms": int(latency_ms),
                "retrieval.sparse_fallback": bool(sparse_fallback),
                "retrieval.qdrant_timeout_s": int(self._qdrant_timeout_s),
            }
            if error_type:
                ev["retrieval.error_type"] = error_type
            log_jsonl(ev)
        except (OSError, ValueError, RuntimeError) as exc:
            redaction = build_pii_log_entry(str(exc), key_id="keyword.telemetry.emit")
            logger.debug(
                "Keyword telemetry emit skipped (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )


__all__ = ["KeywordParams", "KeywordSparseRetriever"]
