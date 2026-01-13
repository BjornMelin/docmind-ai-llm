"""Multimodal retrieval fusion (text + PDF page images).

Design:
- Text retrieval remains server-side hybrid via Qdrant Query API (RRF/DBSF).
- Visual retrieval uses SigLIP text->image embedding against a dedicated Qdrant
  image collection (named vector: ``siglip``).
- Results are fused at the application level via rank-based RRF and then passed
  through the existing modality-aware reranking pipeline.

Qdrant payload for image points must be thin and must not include raw filesystem
paths or base64 blobs. We resolve content-addressed artifacts locally at
render/rerank time.
"""

from __future__ import annotations

import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any

from llama_index.core.schema import NodeWithScore, QueryBundle
from loguru import logger
from qdrant_client import QdrantClient

from src.config import settings
from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams
from src.retrieval.rrf import rrf_merge
from src.utils.qdrant_utils import nodes_from_query_result
from src.utils.siglip_adapter import SiglipEmbedding
from src.utils.storage import get_client_config
from src.utils.telemetry import log_jsonl


@dataclass(frozen=True)
class ImageSearchParams:
    """Configuration for SigLIP image retrieval."""

    collection: str
    top_k: int = 30
    using: str = "siglip"
    with_payload: tuple[str, ...] = (
        "doc_id",
        "page_id",
        "page_no",
        "modality",
        "image_artifact_id",
        "image_artifact_suffix",
        "thumbnail_artifact_id",
        "thumbnail_artifact_suffix",
        "phash",
        "bbox",
        "text",
    )


class ImageSiglipRetriever:
    """Retrieve PDF page images using SigLIP text->image embeddings (Qdrant)."""

    def __init__(
        self,
        params: ImageSearchParams,
        *,
        client: QdrantClient | None = None,
        embedder: SiglipEmbedding | None = None,
    ) -> None:
        """Create a SigLIP image retriever."""
        self.params = params
        self._client = client or QdrantClient(**get_client_config())
        self._embedder = embedder or SiglipEmbedding()
        self._collection_checked = False

    def close(self) -> None:
        """Close underlying client (best-effort)."""
        try:
            self._client.close()
        except Exception:  # pragma: no cover - defensive
            return

    def _query_vec(self, vec: Any, *, top_k: int | None = None) -> list[NodeWithScore]:
        limit = int(top_k or self.params.top_k)
        if hasattr(vec, "tolist"):
            vec_list = vec.tolist()
        else:
            vec_list = list(vec) if not isinstance(vec, list) else vec
        if not self._collection_checked:
            try:
                from src.retrieval.image_index import ensure_siglip_image_collection

                ensure_siglip_image_collection(
                    self._client,
                    self.params.collection,
                    dim=len(vec_list),
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("SigLIP collection check failed: {}", exc)
            self._collection_checked = True

        try:
            result = self._client.query_points(
                collection_name=self.params.collection,
                query=vec_list,
                using=self.params.using,
                limit=limit,
                with_payload=list(self.params.with_payload),
            )
        except Exception as exc:  # pragma: no cover - fail open
            logger.debug("Image query_points failed: {}", exc)
            return []

        nodes = nodes_from_query_result(
            result,
            top_k=limit,
            id_keys=("page_id",),
            prefer_point_id=False,
        )

        return nodes

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Retrieve image nodes for a text query."""
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)
        try:
            vec = self._embedder.get_text_embedding(qtext)
        except Exception as exc:  # pragma: no cover - fail open
            logger.debug("SigLIP text embedding failed: {}", exc)
            return []
        return self._query_vec(vec)

    def retrieve_by_image(
        self, image: Any, *, top_k: int | None = None
    ) -> list[NodeWithScore]:
        """Retrieve image nodes for an image query."""
        try:
            vec = self._embedder.get_image_embedding(image)
        except Exception as exc:  # pragma: no cover - fail open
            logger.debug("SigLIP image embedding failed: {}", exc)
            return []
        return self._query_vec(vec, top_k=top_k)


class MultimodalFusionRetriever:
    """Fuse text hybrid retrieval with SigLIP image retrieval (rank-based RRF)."""

    def __init__(
        self,
        *,
        text_retriever: ServerHybridRetriever | None = None,
        image_retriever: ImageSiglipRetriever | None = None,
        fused_top_k: int | None = None,
        dedup_key: str | None = None,
    ) -> None:
        """Create a fusion retriever from text and image components."""
        self._text = text_retriever or ServerHybridRetriever(
            _HybridParams(
                collection=settings.database.qdrant_collection,
                fused_top_k=int(getattr(settings.retrieval, "fused_top_k", 60)),
                prefetch_sparse=int(
                    getattr(settings.retrieval, "prefetch_sparse_limit", 400)
                ),
                prefetch_dense=int(
                    getattr(settings.retrieval, "prefetch_dense_limit", 200)
                ),
                fusion_mode=str(getattr(settings.retrieval, "fusion_mode", "rrf")),
                dedup_key=str(getattr(settings.retrieval, "dedup_key", "page_id")),
            )
        )
        self._image = image_retriever or ImageSiglipRetriever(
            ImageSearchParams(collection=settings.database.qdrant_image_collection)
        )
        self._fused_top_k = int(
            fused_top_k or getattr(settings.retrieval, "fused_top_k", 60)
        )
        self._dedup_key = str(
            dedup_key or getattr(settings.retrieval, "dedup_key", "page_id")
        )

    def close(self) -> None:
        """Close underlying retrievers (best-effort)."""
        with contextlib.suppress(Exception):
            self._text.close()
        with contextlib.suppress(Exception):
            self._image.close()

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Retrieve fused multimodal results."""
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as executor:
            text_future = executor.submit(self._text.retrieve, qtext)
            image_future = executor.submit(self._image.retrieve, qtext)
            text_timeout_s = (
                float(getattr(settings.retrieval, "text_timeout_ms", 5000)) / 1000.0
            )
            image_timeout_s = (
                float(getattr(settings.retrieval, "image_timeout_ms", 5000)) / 1000.0
            )
            try:
                text_nodes = text_future.result(timeout=text_timeout_s)
            except FuturesTimeoutError:
                text_future.cancel()
                logger.warning("Text retrieval timed out")
                text_nodes = []
            try:
                image_nodes = image_future.result(timeout=image_timeout_s)
            except FuturesTimeoutError:
                image_future.cancel()
                logger.warning("Image retrieval timed out")
                image_nodes = []

        k_constant = int(getattr(settings.retrieval, "rrf_k", 60))
        fused = rrf_merge([text_nodes, image_nodes], k_constant=k_constant)

        # Deduplicate by configured key (default: page_id).
        key_name = (self._dedup_key or "page_id").strip()
        best: dict[str, tuple[float, NodeWithScore]] = {}
        for nws in fused:
            score = float(getattr(nws, "score", 0.0) or 0.0)
            meta = getattr(nws.node, "metadata", {}) or {}
            key = str(meta.get(key_name) or nws.node.node_id)
            cur = best.get(key)
            if cur is None or score > cur[0]:
                best[key] = (score, nws)

        dedup_sorted = sorted(best.values(), key=lambda x: (-x[0], x[1].node.node_id))
        out = [nws for _score, nws in dedup_sorted[: self._fused_top_k]]

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # PII-safe telemetry-like metadata (no query text).
        with contextlib.suppress(Exception):
            logger.debug(
                "MultimodalFusionRetriever done (text=%d, image=%d, out=%d, ms=%d)",
                len(text_nodes),
                len(image_nodes),
                len(out),
                latency_ms,
            )
        with contextlib.suppress(Exception):
            log_jsonl({
                "retrieval.multimodal": True,
                "retrieval.text_count": len(text_nodes),
                "retrieval.image_count": len(image_nodes),
                "retrieval.fused_count": len(out),
                "retrieval.rrf_k": int(k_constant),
                "dedup.key": key_name,
                "retrieval.latency_ms": latency_ms,
            })

        return out


__all__ = ["ImageSearchParams", "ImageSiglipRetriever", "MultimodalFusionRetriever"]
