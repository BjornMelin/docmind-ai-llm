"""Server-side hybrid retriever using Qdrant Query API (RRF/DBSF).

Library-first, standalone retriever that composes dense + sparse queries via
Qdrant's Query API with Prefetch + FusionQuery. Returns LlamaIndex
NodeWithScore list with deterministic ordering and de-duplication.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np
from llama_index.core.schema import NodeWithScore, QueryBundle
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from src.config.integrations import get_settings_embed_model
from src.retrieval.sparse_query import encode_to_qdrant as _encode_sparse_query
from src.utils.exceptions import IMPORT_EXCEPTIONS
from src.utils.qdrant_exceptions import QDRANT_SCHEMA_EXCEPTIONS
from src.utils.qdrant_utils import (
    QDRANT_PAYLOAD_FIELDS,
    build_text_nodes,
)
from src.utils.storage import ensure_hybrid_collection, get_client_config
from src.utils.telemetry import log_jsonl


@dataclass
class HybridParams:
    """Configuration for server-side hybrid retrieval (RRF/DBSF fusion)."""

    collection: str
    fused_top_k: int = 60
    prefetch_sparse: int = 400
    prefetch_dense: int = 200
    fusion_mode: str = "rrf"  # or "dbsf"
    dedup_key: str = "page_id"


class ServerHybridRetriever:
    """Hybrid retriever using Qdrant's server-side fusion (RRF/DBSF).

    Computes dense and sparse query representations, prefetches candidates
    for both named vectors (``text-dense``/``text-sparse``), applies
    server-side fusion via Qdrant Query API, then performs deterministic
    de-duplication before the final cut.

    Notes:
        - Fusion modes supported: Reciprocal Rank Fusion (RRF) and
          Distribution-Based Score Fusion (DBSF).
        - Deduplication key defaults to ``page_id`` but is configurable.
    """

    def __init__(
        self,
        params: HybridParams,
        client: QdrantClient | None = None,
        client_factory: Callable[[], QdrantClient] | None = None,
    ) -> None:
        """Initialize retriever with parameters and client.

        Args:
            params: Retrieval parameters including collection, fusion mode,
                prefetch limits, and de-duplication key.
            client: Optional pre-configured QdrantClient instance.
            client_factory: Optional factory function to create QdrantClient.
        """
        self.params = params
        if client is not None:
            self._client = client
        else:
            factory = client_factory or (lambda: QdrantClient(**get_client_config()))
            self._client = factory()
        # Ensure hybrid schema exists for upgrade safety (idempotent)
        try:
            ensure_hybrid_collection(self._client, self.params.collection)
        except (
            QDRANT_SCHEMA_EXCEPTIONS
        ) as exc:  # pragma: no cover - defensive best-effort
            logger.debug("Hybrid schema ensure skipped: %s", exc)

    def close(self) -> None:
        """Close underlying client (best-effort)."""
        with suppress(Exception):  # pragma: no cover - defensive
            self._client.close()

    def _embed_dense(self, text: str) -> np.ndarray:
        """Embed text into a dense vector using configured model.

        Args:
            text: Input text to embed.

        Returns:
            Dense vector representation as ``np.ndarray[float32]``.

        Raises:
            RuntimeError: If ``Settings.embed_model`` is not configured.
        """
        embed = get_settings_embed_model()
        if embed is None:
            raise RuntimeError(
                "Settings.embed_model is not configured. Initialize integrations or "
                "setup embedding before using ServerHybridRetriever."
            )
        vec = embed.get_query_embedding(text)  # type: ignore[attr-defined]
        return np.asarray(vec, dtype=np.float32)

    def _encode_sparse(self, text: str) -> qmodels.SparseVector | None:
        """Encode text into a Qdrant ``SparseVector`` when available.

        Args:
            text: Input text to encode sparsely.

        Returns:
            A ``SparseVector`` when a sparse encoder is available and succeeds;
            otherwise ``None`` to indicate dense-only fallback.
        """
        try:
            return _encode_sparse_query(text)
        except (ValueError, TypeError, AttributeError):  # pragma: no cover - defensive
            return None

    def _fusion(self) -> qmodels.FusionQuery:
        """Build a ``FusionQuery`` based on configured fusion mode.

        Returns:
            ``FusionQuery`` configured for ``DBSF`` when ``fusion_mode`` is
            ``"dbsf"`` (case-insensitive); otherwise defaults to ``RRF``.
        """
        mode = (self.params.fusion_mode or "rrf").lower()
        if mode == "dbsf":
            return qmodels.FusionQuery(fusion=qmodels.Fusion.DBSF)
        return qmodels.FusionQuery(fusion=qmodels.Fusion.RRF)

    def _build_prefetch(
        self,
        dense_vec: np.ndarray,
        sparse_vec: qmodels.SparseVector | None,
    ) -> list[qmodels.Prefetch]:
        """Build prefetch queries for dense and optional sparse vectors."""
        prefetch: list[qmodels.Prefetch] = []
        if sparse_vec is not None:
            if isinstance(sparse_vec, dict):
                try:
                    idxs, vals = (
                        zip(*sorted(sparse_vec.items()), strict=False)
                        if sparse_vec
                        else ([], [])
                    )
                except (ValueError, TypeError):  # pragma: no cover - defensive
                    idxs, vals = ([], [])
                sparse_vec = qmodels.SparseVector(indices=list(idxs), values=list(vals))
            prefetch.append(
                qmodels.Prefetch(
                    query=sparse_vec,
                    using="text-sparse",
                    limit=self.params.prefetch_sparse,
                )
            )

        d_query: Any = (
            dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)
        )
        prefetch.append(
            qmodels.Prefetch(
                query=d_query,  # type: ignore[arg-type]
                using="text-dense",
                limit=self.params.prefetch_dense,
            )
        )
        return prefetch

    def _query_qdrant(
        self, prefetch: list[qmodels.Prefetch], fused_fetch_k: int
    ) -> Any:
        """Execute the fused Qdrant query."""
        return self._client.query_points(
            collection_name=self.params.collection,
            prefetch=prefetch,
            query=self._fusion(),
            limit=fused_fetch_k,
            with_payload=list(QDRANT_PAYLOAD_FIELDS),
        )

    def _dedup_points(
        self, points: list[Any], key_name: str
    ) -> tuple[list[tuple[float, Any]], int]:
        """Deduplicate points by key and return sorted list plus count.

        Args:
            points: List of Qdrant points to deduplicate.
            key_name: Metadata key used for deduplication.
        """
        best: dict[str, tuple[float, Any]] = {}
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            score = float(getattr(p, "score", 0.0))
            key = str(payload.get(key_name) or p.id)
            cur = best.get(key)
            if cur is None or score > cur[0]:
                best[key] = (score, p)
        dedup_sorted = sorted(
            best.values(), key=lambda x: (-x[0], str(getattr(x[1], "id", "")))
        )
        return dedup_sorted, len(best)

    def _build_nodes(
        self, dedup_sorted: list[tuple[float, Any]]
    ) -> list[NodeWithScore]:
        """Convert deduplicated points into NodeWithScore items."""
        points = [p for _, p in dedup_sorted[: self.params.fused_top_k]]
        return build_text_nodes(
            points,
            top_k=int(self.params.fused_top_k),
            id_keys=("chunk_id", "page_id", "doc_id"),
            prefer_point_id=True,
        )

    def _emit_telemetry(
        self,
        *,
        t0: float,
        nodes: list[NodeWithScore],
        sparse_vec: Any | None,
        key_name: str,
        input_count: int,
        unique_count: int,
    ) -> None:
        """Emit retrieval telemetry in a PII-safe form."""
        try:
            latency_ms = int((time.time() - t0) * 1000)
            fusion_mode = (self.params.fusion_mode or "rrf").lower()
            try:
                from src.config import settings as _settings  # local import

                rrf_k_val = int(getattr(_settings.retrieval, "rrf_k", 60))
                qdrant_timeout_s = int(
                    getattr(_settings.database, "qdrant_timeout", 60)
                )
            except IMPORT_EXCEPTIONS:  # pragma: no cover - defensive
                rrf_k_val = 60
                qdrant_timeout_s = 60
            dropped = max(0, input_count - unique_count)
            log_jsonl(
                {
                    "retrieval.backend": "qdrant",
                    "retrieval.fusion_mode": fusion_mode,
                    "retrieval.rrf_k": rrf_k_val,
                    "retrieval.prefetch_dense_limit": self.params.prefetch_dense,
                    "retrieval.prefetch_sparse_limit": self.params.prefetch_sparse,
                    "retrieval.fused_top_k": self.params.fused_top_k,
                    "retrieval.return_count": len(nodes),
                    "retrieval.latency_ms": latency_ms,
                    "retrieval.sparse_fallback": sparse_vec is None,
                    "retrieval.qdrant_timeout_s": qdrant_timeout_s,
                    "dedup.key": key_name,
                    "dedup.before": input_count,
                    "dedup.after": unique_count,
                    "dedup.dropped": dropped,
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("Telemetry emit skipped: %s", exc)

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Execute server-side hybrid retrieval.

        Computes dense and sparse queries, prefetches candidates, fuses scores
        on the server, performs deterministic de-duplication, and returns
        ``NodeWithScore`` results ordered by score (descending) and id (stable).

        Args:
            query: Query text or ``QueryBundle``.

        Returns:
            A list of ``NodeWithScore`` instances up to ``fused_top_k``.

        Notes:
            On network or query errors, the retriever fails open and returns an
            empty list rather than raising, to avoid breaking calling pipelines.
        """
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)

        t0 = time.time()
        dense_vec = self._embed_dense(qtext)
        sparse_vec = self._encode_sparse(qtext)

        prefetch = self._build_prefetch(dense_vec, sparse_vec)

        # Optional owner filter via env/telemetry could be added here
        # Headroom for server-side limit to mitigate post-fusion dedup underfill
        fused_fetch_k = max(
            self.params.prefetch_dense,
            self.params.prefetch_sparse,
            self.params.fused_top_k,
        )

        try:
            result = self._query_qdrant(prefetch, fused_fetch_k)
        # Network/remote path: treat common connectivity or query errors as empty result
        except QDRANT_SCHEMA_EXCEPTIONS as exc:  # pragma: no cover
            logger.warning("Qdrant hybrid query failed: %s", exc)
            # Dense-only fallback via vector index is not available here; return empty
            return []

        # Deterministic dedup by key
        key_name = (self.params.dedup_key or "page_id").strip()
        points = getattr(result, "points", []) or getattr(result, "result", [])
        input_count = len(points)
        dedup_sorted, unique_count = self._dedup_points(points, key_name)
        nodes = self._build_nodes(dedup_sorted)

        # Telemetry (PII-safe; no query text)
        self._emit_telemetry(
            t0=t0,
            nodes=nodes,
            sparse_vec=sparse_vec,
            key_name=key_name,
            input_count=input_count,
            unique_count=unique_count,
        )

        return nodes


__all__ = ["HybridParams", "ServerHybridRetriever"]
