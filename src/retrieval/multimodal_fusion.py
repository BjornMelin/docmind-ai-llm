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

import asyncio
import contextlib
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from loguru import logger
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.config import settings
from src.retrieval.async_work import AsyncWorkExecutor
from src.retrieval.hybrid import HybridParams, ServerHybridRetriever
from src.retrieval.image_index import check_siglip_image_collection
from src.retrieval.rrf import rrf_merge
from src.utils.log_safety import build_pii_log_entry
from src.utils.qdrant_utils import nodes_from_query_result
from src.utils.siglip_adapter import SiglipEmbedding
from src.utils.storage import get_client_config, sparse_retrieval_enabled
from src.utils.telemetry import log_jsonl

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image

    type ImageInput = Image.Image | np.ndarray | bytes | str
else:
    type ImageInput = Any


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
        client_factory: Callable[[], QdrantClient] | None = None,
        async_client: AsyncQdrantClient | None = None,
        async_client_factory: Callable[[], AsyncQdrantClient] | None = None,
        embedder: SiglipEmbedding | None = None,
    ) -> None:
        """Create a SigLIP image retriever."""
        self.params = params
        client_config = get_client_config()
        self._client = client
        self._client_factory = client_factory or (lambda: QdrantClient(**client_config))
        self._async_client = async_client
        self._async_client_factory = async_client_factory or (
            lambda: AsyncQdrantClient(**client_config)
        )
        self._embedder = embedder or SiglipEmbedding()
        self._cpu_work = AsyncWorkExecutor(name="docmind-siglip-cpu")
        try:
            check_siglip_image_collection(
                self._get_client(),
                self.params.collection,
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close the synchronous client (best-effort)."""
        self._cpu_work.close()
        if self._client is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                self._client.close()
            self._client = None

    async def aclose(self) -> None:
        """Close both clients after asynchronous use (best-effort)."""
        self._cpu_work.close()
        async_client = self._async_client
        self._async_client = None
        if async_client is not None:
            with contextlib.suppress(Exception):  # pragma: no cover - defensive
                await async_client.close()
        await self._cpu_work.aclose()
        self.close()

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def _get_async_client(self) -> AsyncQdrantClient:
        if self._async_client is None:
            self._async_client = self._async_client_factory()
        return self._async_client

    def _query_vec(self, vec: Any, *, top_k: int | None = None) -> list[NodeWithScore]:
        limit = int(top_k or self.params.top_k)
        if hasattr(vec, "tolist"):
            vec_list = vec.tolist()
        else:
            vec_list = list(vec) if not isinstance(vec, list) else vec
        try:
            timeout_ms = settings.retrieval.siglip_timeout_ms
            timeout_s = max(1, math.ceil(timeout_ms / 1000))
            result = self._get_client().query_points(
                collection_name=self.params.collection,
                query=vec_list,
                using=self.params.using,
                limit=limit,
                with_payload=list(self.params.with_payload),
                timeout=timeout_s,
            )
        except Exception as exc:  # pragma: no cover - fail open
            redaction = build_pii_log_entry(
                str(exc), key_id="multimodal_fusion.query_points"
            )
            logger.debug(
                "Image query_points failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return []

        nodes = nodes_from_query_result(
            result,
            top_k=limit,
            id_keys=("page_id",),
            prefer_point_id=False,
        )

        return nodes

    async def _aquery_vec(
        self, vec: Any, *, top_k: int | None = None
    ) -> list[NodeWithScore]:
        """Query image vectors through Qdrant's native async client."""
        limit = int(top_k or self.params.top_k)
        vec_list = vec.tolist() if hasattr(vec, "tolist") else list(vec)
        try:
            timeout_ms = settings.retrieval.siglip_timeout_ms
            result = await self._get_async_client().query_points(
                collection_name=self.params.collection,
                query=vec_list,
                using=self.params.using,
                limit=limit,
                with_payload=list(self.params.with_payload),
                timeout=max(1, math.ceil(timeout_ms / 1000)),
            )
        except Exception as exc:  # pragma: no cover - fail open
            redaction = build_pii_log_entry(
                str(exc), key_id="multimodal_fusion.query_points"
            )
            logger.debug(
                "Image query_points failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return []
        return nodes_from_query_result(
            result,
            top_k=limit,
            id_keys=("page_id",),
            prefer_point_id=False,
        )

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Retrieve image nodes for a text query."""
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)
        try:
            vec = self._embedder.get_text_embedding(qtext)
        except Exception as exc:  # pragma: no cover - fail open
            redaction = build_pii_log_entry(
                str(exc), key_id="multimodal_fusion.text_embedding"
            )
            logger.debug(
                "SigLIP text embedding failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return []
        return self._query_vec(vec)

    async def aretrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Retrieve image nodes without blocking the event loop."""
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)
        try:
            vec = await self._cpu_work.run(self._embedder.get_text_embedding, qtext)
        except Exception as exc:  # pragma: no cover - fail open
            redaction = build_pii_log_entry(
                str(exc), key_id="multimodal_fusion.text_embedding"
            )
            logger.debug(
                "SigLIP text embedding failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return []
        return await self._aquery_vec(vec)

    def retrieve_by_image(
        self, image: ImageInput, *, top_k: int | None = None
    ) -> list[NodeWithScore]:
        """Retrieve image nodes for an image query."""
        try:
            vec = self._embedder.get_image_embedding(image)
        except Exception as exc:  # pragma: no cover - fail open
            redaction = build_pii_log_entry(
                str(exc), key_id="multimodal_fusion.image_embedding"
            )
            logger.debug(
                "SigLIP image embedding failed (error_type={}, error={})",
                type(exc).__name__,
                redaction.redacted,
            )
            return []
        return self._query_vec(vec, top_k=top_k)


class MultimodalFusionRetriever(BaseRetriever):
    """Fuse text hybrid retrieval with SigLIP image retrieval (rank-based RRF)."""

    def __init__(
        self,
        *,
        text_retriever: Any | None = None,
        image_retriever: ImageSiglipRetriever | None = None,
        text_collection: str | None = None,
        image_collection: str | None = None,
        fused_top_k: int | None = None,
        dedup_key: str | None = None,
    ) -> None:
        """Create a fusion retriever from text and image components."""
        super().__init__()
        with contextlib.ExitStack() as acquired:
            if text_retriever is None:
                if not sparse_retrieval_enabled():
                    raise ValueError(
                        "Dense multimodal fusion requires an injected text retriever"
                    )
                text_retriever = ServerHybridRetriever(
                    HybridParams(
                        collection=(
                            text_collection or settings.database.qdrant_collection
                        ),
                        fused_top_k=settings.retrieval.fused_top_k,
                        prefetch_sparse=settings.retrieval.prefetch_sparse_limit,
                        prefetch_dense=settings.retrieval.prefetch_dense_limit,
                        fusion_mode=settings.retrieval.fusion_mode,
                        rrf_k=settings.retrieval.rrf_k,
                        dedup_key=settings.retrieval.dedup_key,
                    )
                )
                acquired.callback(text_retriever.close)
            if image_retriever is None:
                image_retriever = ImageSiglipRetriever(
                    ImageSearchParams(
                        collection=(
                            image_collection
                            or settings.database.qdrant_image_collection
                        )
                    )
                )
                acquired.callback(image_retriever.close)
            self._text = text_retriever
            self._image = image_retriever
            self._fused_top_k = fused_top_k or settings.retrieval.fused_top_k
            self._dedup_key = dedup_key or settings.retrieval.dedup_key
            acquired.pop_all()

    def close(self) -> None:
        """Close underlying retrievers (best-effort)."""
        close_text = getattr(self._text, "close", None)
        if callable(close_text):
            with contextlib.suppress(Exception):
                close_text()
        with contextlib.suppress(Exception):
            self._image.close()

    async def aclose(self) -> None:
        """Close both child retrievers after asynchronous use."""
        close_calls: list[Awaitable[Any]] = [self._image.aclose()]
        close_text = getattr(self._text, "aclose", None)
        if callable(close_text):
            close_calls.append(cast(Callable[[], Awaitable[Any]], close_text)())
        await asyncio.gather(*close_calls, return_exceptions=True)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve fused results using each child's configured client timeout."""
        qtext = query_bundle.query_str
        t0 = time.perf_counter()
        text_nodes = self._text.retrieve(qtext)
        image_nodes = self._image.retrieve(qtext)
        return self._fuse(text_nodes, image_nodes, t0=t0)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve both branches under the router's total request deadline."""
        qtext = query_bundle.query_str
        t0 = time.perf_counter()

        text_nodes, image_nodes = await asyncio.gather(
            self._text.aretrieve(qtext),
            self._image.aretrieve(qtext),
        )
        return self._fuse(text_nodes, image_nodes, t0=t0)

    def _fuse(
        self,
        text_nodes: list[NodeWithScore],
        image_nodes: list[NodeWithScore],
        *,
        t0: float,
    ) -> list[NodeWithScore]:
        """Fuse, deduplicate, and record retrieval results."""
        try:
            k_constant = int(settings.retrieval.rrf_k)
        except (AttributeError, TypeError, ValueError):
            k_constant = 60
        fused = rrf_merge([text_nodes, image_nodes], k_constant=k_constant)

        # Deduplicate by configured key (default: page_id).
        key_name = (self._dedup_key or "page_id").strip()
        best: dict[str, tuple[float, NodeWithScore]] = {}
        for nws in fused:
            score = float(getattr(nws, "score", 0.0) or 0.0)
            meta = getattr(nws.node, "metadata", {}) or {}
            key = str(meta.get(key_name) or nws.node.node_id)
            best[key] = max(
                best.get(key, (score, nws)), (score, nws), key=lambda x: x[0]
            )

        dedup_sorted = sorted(best.values(), key=lambda x: (-x[0], x[1].node.node_id))
        out = [nws for _, nws in dedup_sorted[: self._fused_top_k]]

        latency_ms = int((time.perf_counter() - t0) * 1000)

        # PII-safe telemetry-like metadata (no query text).
        with contextlib.suppress(Exception):
            logger.debug(
                "MultimodalFusionRetriever done (text={}, image={}, out={}, ms={})",
                len(text_nodes),
                len(image_nodes),
                len(out),
                latency_ms,
            )
        with contextlib.suppress(Exception):
            log_jsonl(
                {
                    "retrieval.multimodal": True,
                    "retrieval.text_count": len(text_nodes),
                    "retrieval.image_count": len(image_nodes),
                    "retrieval.fused_count": len(out),
                    "retrieval.rrf_k": int(k_constant),
                    "dedup.key": key_name,
                    "dedup.before": len(fused),
                    "dedup.after": len(dedup_sorted),
                    "dedup.dropped": len(fused) - len(dedup_sorted),
                    "retrieval.latency_ms": latency_ms,
                }
            )

        return out


__all__ = ["ImageSearchParams", "ImageSiglipRetriever", "MultimodalFusionRetriever"]
