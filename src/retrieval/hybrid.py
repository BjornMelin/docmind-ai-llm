"""Server-side hybrid retriever using Qdrant Query API (RRF/DBSF).

Library-first, standalone retriever that composes dense + sparse queries via
Qdrant's Query API with Prefetch plus native RRF/DBSF queries. Returns LlamaIndex
NodeWithScore list with deterministic ordering and server-side grouping.
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

from src.config import settings
from src.config.integrations import get_settings_embed_model
from src.retrieval.sparse_query import encode_to_qdrant as _encode_sparse_query
from src.utils.exceptions import IMPORT_EXCEPTIONS
from src.utils.log_safety import build_pii_log_entry
from src.utils.qdrant_exceptions import QDRANT_SCHEMA_EXCEPTIONS
from src.utils.qdrant_utils import (
    QDRANT_PAYLOAD_FIELDS,
    build_text_nodes,
    order_points,
)
from src.utils.storage import (
    QdrantCollectionIncompatibleError,
    ensure_hybrid_collection,
    get_client_config,
)
from src.utils.telemetry import log_jsonl


@dataclass
class HybridParams:
    """Configuration for server-side hybrid retrieval (RRF/DBSF fusion)."""

    collection: str
    fused_top_k: int = 60
    prefetch_sparse: int = 400
    prefetch_dense: int = 200
    fusion_mode: str = "rrf"  # or "dbsf"
    rrf_k: int = 60
    dedup_key: str = "page_id"


class ServerHybridRetriever:
    """Hybrid retriever using Qdrant's server-side fusion (RRF/DBSF).

    Computes dense and sparse query representations, prefetches candidates
    for both named vectors (``text-dense``/``text-sparse``), applies
    server-side fusion and grouping via Qdrant Query API.

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
        compatibility = ensure_hybrid_collection(
            self._client,
            self.params.collection,
            dense_dim=settings.embedding.dimension,
        )
        if not compatibility.compatible:
            raise QdrantCollectionIncompatibleError(
                self.params.collection,
                compatibility,
            )

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

    def _fusion(self) -> qmodels.FusionQuery | qmodels.RrfQuery:
        """Build the configured native Qdrant fusion query.

        Returns:
            ``FusionQuery`` for DBSF, otherwise an ``RrfQuery`` carrying the
            configured RRF k-constant.
        """
        mode = (self.params.fusion_mode or "rrf").lower()
        if mode == "dbsf":
            return qmodels.FusionQuery(fusion=qmodels.Fusion.DBSF)
        return qmodels.RrfQuery(rrf=qmodels.Rrf(k=int(self.params.rrf_k)))

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

    def _query_qdrant(self, prefetch: list[qmodels.Prefetch], key_name: str) -> Any:
        """Execute fused retrieval grouped by the canonical payload key."""
        return self._client.query_points_groups(
            collection_name=self.params.collection,
            prefetch=prefetch,
            query=self._fusion(),
            group_by=key_name,
            group_size=1,
            limit=self.params.fused_top_k,
            with_payload=list(QDRANT_PAYLOAD_FIELDS),
        )

    def _build_nodes(self, points: list[Any]) -> list[NodeWithScore]:
        """Convert grouped points into deterministically ordered nodes."""
        return build_text_nodes(
            order_points(points),
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
        server_group_count: int,
    ) -> None:
        """Emit retrieval telemetry in a PII-safe form."""
        try:
            latency_ms = int((time.perf_counter() - t0) * 1000)
            fusion_mode = (self.params.fusion_mode or "rrf").lower()
            try:
                from src.config import settings as _settings  # local import

                qdrant_timeout_s = int(
                    getattr(_settings.database, "qdrant_timeout", 60)
                )
            except IMPORT_EXCEPTIONS:  # pragma: no cover - defensive
                qdrant_timeout_s = 60
            event = {
                "retrieval.backend": "qdrant",
                "retrieval.fusion_mode": fusion_mode,
                "retrieval.prefetch_dense_limit": self.params.prefetch_dense,
                "retrieval.prefetch_sparse_limit": self.params.prefetch_sparse,
                "retrieval.fused_limit": self.params.fused_top_k,
                "retrieval.return_count": len(nodes),
                "retrieval.latency_ms": latency_ms,
                "retrieval.sparse_fallback": sparse_vec is None,
                "retrieval.qdrant_timeout_s": qdrant_timeout_s,
                "dedup.key": key_name,
                "dedup.group_size": 1,
                "dedup.server_group_count": server_group_count,
                "dedup.server_side": True,
            }
            if fusion_mode == "rrf":
                event["retrieval.rrf_k"] = int(self.params.rrf_k)
            log_jsonl(event)
        except (OSError, ValueError, RuntimeError) as exc:
            redaction = build_pii_log_entry(str(exc), key_id="hybrid.telemetry.emit")
            logger.debug(
                "Telemetry emit skipped (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Execute server-side hybrid retrieval.

        Computes dense and sparse queries, prefetches candidates, fuses scores
        on the server, groups by the configured payload key, and returns
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

        t0 = time.perf_counter()
        dense_vec = self._embed_dense(qtext)
        sparse_vec = self._encode_sparse(qtext)

        prefetch = self._build_prefetch(dense_vec, sparse_vec)

        key_name = (self.params.dedup_key or "page_id").strip()

        try:
            result = self._query_qdrant(prefetch, key_name)
        # Network/remote path: treat common connectivity or query errors as empty result
        except QDRANT_SCHEMA_EXCEPTIONS as exc:  # pragma: no cover
            redaction = build_pii_log_entry(str(exc), key_id="hybrid.qdrant.query")
            logger.warning(
                "Qdrant hybrid query failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            # Dense-only fallback via vector index is not available here; return empty
            return []

        groups = getattr(result, "groups", []) or []
        points = [group.hits[0] for group in groups if getattr(group, "hits", None)]
        nodes = self._build_nodes(points)

        # Telemetry (PII-safe; no query text)
        self._emit_telemetry(
            t0=t0,
            nodes=nodes,
            sparse_vec=sparse_vec,
            key_name=key_name,
            server_group_count=len(groups),
        )

        return nodes


__all__ = ["HybridParams", "ServerHybridRetriever"]
