"""Server-side hybrid retriever using Qdrant Query API (RRF/DBSF).

Library-first, standalone retriever that composes dense + sparse queries via
Qdrant's Query API with Prefetch + FusionQuery. Returns LlamaIndex
NodeWithScore list with deterministic ordering and de-duplication.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.exceptions import (
    ResponseHandlingException,
    UnexpectedResponse,
)

from src.retrieval.sparse_query import encode_to_qdrant as _encode_sparse_query
from src.utils.storage import ensure_hybrid_collection, get_client_config
from src.utils.telemetry import log_jsonl


@dataclass
class _HybridParams:
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
        params: _HybridParams,
        client: QdrantClient | None = None,
        client_factory: Callable[[], QdrantClient] | None = None,
    ) -> None:
        """Initialize retriever with parameters and client.

        Args:
            params: Retrieval parameters including collection, fusion mode,
                prefetch limits, and de-duplication key.
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
            ResponseHandlingException,
            UnexpectedResponse,
            ConnectionError,
            TimeoutError,
            OSError,
            ValueError,
            AttributeError,
        ) as exc:  # pragma: no cover - defensive best-effort
            logger.debug("Hybrid schema ensure skipped: %s", exc)

    def close(self) -> None:
        """Close underlying client (best-effort)."""
        try:
            self._client.close()
        except Exception:  # pragma: no cover - defensive
            pass

    def _embed_dense(self, text: str) -> np.ndarray:
        """Embed text into a dense vector using configured model.

        Args:
            text: Input text to embed.

        Returns:
            Dense vector representation as ``np.ndarray[float32]``.

        Raises:
            RuntimeError: If ``Settings.embed_model`` is not configured.
        """
        embed = getattr(Settings, "embed_model", None)
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

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:  # pylint: disable=too-many-locals
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

        prefetch: list[qmodels.Prefetch] = []
        if sparse_vec is not None:
            # Ensure typed SparseVector (some encoders return dict)
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

        # Support ndarray or python list (monkeypatched in tests)
        d_query: Any = (
            dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)
        )
        # Use raw list for dense query input to maximize compatibility

        prefetch.append(
            qmodels.Prefetch(
                query=d_query,  # type: ignore[arg-type]
                using="text-dense",
                limit=self.params.prefetch_dense,
            )
        )

        # Optional owner filter via env/telemetry could be added here
        # Headroom for server-side limit to mitigate post-fusion dedup underfill
        fused_fetch_k = max(
            self.params.prefetch_dense,
            self.params.prefetch_sparse,
            self.params.fused_top_k,
        )

        try:
            result = self._client.query_points(
                collection_name=self.params.collection,
                prefetch=prefetch,
                query=self._fusion(),
                limit=fused_fetch_k,
                with_payload=[
                    "doc_id",
                    "page_id",
                    "chunk_id",
                    "text",
                    "modality",
                    "image_path",
                ],
            )
        # Network/remote path: treat common connectivity or query errors as empty result
        except (ConnectionError, TimeoutError, ValueError) as exc:  # pragma: no cover
            logger.warning("Qdrant hybrid query failed: %s", exc)
            # Dense-only fallback via vector index is not available here; return empty
            return []

        # Deterministic dedup by key
        key_name = (self.params.dedup_key or "page_id").strip()
        best: dict[str, tuple[float, Any]] = {}
        points = getattr(result, "points", []) or getattr(result, "result", [])
        input_count = len(points)
        for p in points:
            payload = getattr(p, "payload", {}) or {}
            score = float(getattr(p, "score", 0.0))
            key = str(payload.get(key_name) or p.id)
            cur = best.get(key)
            if cur is None or score > cur[0]:
                best[key] = (score, p)

        # Stable ordering: score desc, id asc
        dedup_sorted = sorted(
            best.values(), key=lambda x: (-x[0], str(getattr(x[1], "id", "")))
        )
        unique_count = len(dedup_sorted)
        nodes: list[NodeWithScore] = []
        for score, p in dedup_sorted[: self.params.fused_top_k]:
            payload = getattr(p, "payload", {}) or {}
            text = payload.get("text") or ""
            nid = (
                str(p.id)
                if getattr(p, "id", None) is not None
                else str(
                    payload.get("chunk_id")
                    or payload.get("page_id")
                    or payload.get("doc_id")
                    or ""
                )
            )
            node = TextNode(text=text, id_=nid)
            node.metadata.update({k: v for k, v in payload.items() if k != "text"})
            nodes.append(NodeWithScore(node=node, score=score))

        # Telemetry (PII-safe; no query text)
        try:
            latency_ms = int((time.time() - t0) * 1000)
            fusion_mode = (self.params.fusion_mode or "rrf").lower()
            log_jsonl(
                {
                    "retrieval.fusion_mode": fusion_mode,
                    "retrieval.prefetch_dense_limit": self.params.prefetch_dense,
                    "retrieval.prefetch_sparse_limit": self.params.prefetch_sparse,
                    "retrieval.fused_limit": self.params.fused_top_k,
                    "retrieval.return_count": len(nodes),
                    "retrieval.latency_ms": latency_ms,
                    "retrieval.sparse_fallback": sparse_vec is None,
                    "dedup.key": key_name,
                    "dedup.input_count": input_count,
                    "dedup.unique_count": unique_count,
                    "dedup.duplicates_removed": max(0, input_count - unique_count),
                }
            )
        except (OSError, ValueError, RuntimeError) as exc:
            logger.debug("Telemetry emit skipped: %s", exc)

        return nodes


__all__ = ["ServerHybridRetriever"]
