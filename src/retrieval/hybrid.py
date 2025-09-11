"""Server-side hybrid retriever using Qdrant Query API (RRF/DBSF).

Library-first, standalone retriever that composes dense + sparse queries via
Qdrant's Query API with Prefetch + FusionQuery. Returns LlamaIndex
NodeWithScore list with deterministic ordering and de-duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from llama_index.core import Settings
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from src.retrieval.sparse_query import encode_to_qdrant as _encode_sparse_query
from src.utils.storage import get_client_config


@dataclass
class _HybridParams:
    collection: str
    fused_top_k: int = 60
    prefetch_sparse: int = 400
    prefetch_dense: int = 200
    fusion_mode: str = "rrf"  # or "dbsf"
    dedup_key: str = "page_id"


class ServerHybridRetriever:
    """Retriever that uses Qdrant Query API server-side fusion (RRF/DBSF).

    - Computes dense + sparse for the query text.
    - Prefetches sparse + dense using named vectors 'text-sparse'/'text-dense'.
    - De-dups by the configured key before final cut.
    - Returns LlamaIndex NodeWithScore list with deterministic ordering.
    """

    def __init__(self, params: _HybridParams) -> None:
        """Initialize with retrieval parameters and Qdrant client."""
        self.params = params
        self._client = QdrantClient(**get_client_config())

    def _embed_dense(self, text: str) -> np.ndarray:
        vec = Settings.embed_model.get_query_embedding(  # type: ignore[attr-defined]
            text
        )
        return np.asarray(vec, dtype=np.float32)

    def _encode_sparse(self, text: str) -> qmodels.SparseVector | None:
        try:
            return _encode_sparse_query(text)
        except (ValueError, TypeError, AttributeError):  # pragma: no cover - defensive
            return None

    def _fusion(self) -> qmodels.FusionQuery:
        mode = (self.params.fusion_mode or "rrf").lower()
        if mode == "dbsf":
            return qmodels.FusionQuery(fusion=qmodels.Fusion.DBSF)
        return qmodels.FusionQuery(fusion=qmodels.Fusion.RRF)

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:  # pylint: disable=too-many-locals
        """Execute hybrid retrieval and return deduplicated, ordered results."""
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)

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
        try:
            result = self._client.query_points(
                collection_name=self.params.collection,
                prefetch=prefetch,
                query=self._fusion(),
                limit=self.params.fused_top_k,
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

        return nodes


__all__ = ["ServerHybridRetriever", "_HybridParams"]
