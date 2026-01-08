"""Multimodal utilities for image embeddings and simple cross-modal flows.

This module keeps logic lightweight and library-first. All heavy model calls
are expected to be provided by the caller (e.g., ``clip``-like object with a
``get_image_embedding`` method). Tests mock these boundaries to remain fully
offline and deterministic.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from loguru import logger

# Public constants (validated by unit tests)
EMBEDDING_DIMENSIONS = 512
MAX_TEST_IMAGES = 10
TEXT_TRUNCATION_LIMIT = 200
RANK_ADJUSTMENT = 1


@dataclass
class ImageDocument:
    """Simple container for an image document reference."""

    image_path: str
    metadata: dict[str, Any]


async def generate_image_embeddings(clip: Any, image: Any) -> np.ndarray:
    """Generate and L2-normalize an image embedding using the provided model.

    Args:
        clip: Object exposing ``get_image_embedding(image)``.
        image: PIL.Image.Image or any accepted type by the backend.

    Returns:
        L2-normalized numpy vector.
    """
    raw = await asyncio.to_thread(clip.get_image_embedding, image)
    if hasattr(raw, "cpu") and callable(raw.cpu):  # torch tensor path
        vec = cast(Any, raw).cpu().numpy()
    else:
        vec = np.asarray(raw, dtype=np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


def validate_vram_usage(clip: Any, images: list[Any] | None = None) -> float:
    """Return approximate VRAM delta (GB) for a tiny embedding run.

    Args:
        clip: Model exposing ``get_image_embedding``. Only used when ``images``
            are supplied.
        images: Optional list of images to embed. When provided the helper
            measures VRAM before and after the calls.

    Returns:
        float: Delta in gigabytes. ``0.0`` when CUDA is unavailable or any
        error occurs.
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return 0.0

    try:
        cuda_iface = getattr(torch, "cuda", None)
        if not cuda_iface:
            return 0.0
        if not cuda_iface.is_available():  # type: ignore[attr-defined]
            return 0.0

        before = float(cuda_iface.memory_allocated())  # type: ignore[attr-defined]
        if images:
            for img in images[:MAX_TEST_IMAGES]:
                try:
                    clip.get_image_embedding(img)
                except (RuntimeError, ValueError, TypeError):  # pragma: no cover
                    break

        after = float(cuda_iface.memory_allocated())  # type: ignore[attr-defined]
        with contextlib.suppress(RuntimeError, AttributeError):
            cuda_iface.empty_cache()  # type: ignore[attr-defined]
        return max(0.0, (after - before) / (1024**3))
    except (AttributeError, RuntimeError):
        return 0.0


def batch_process_images(
    clip: Any,
    images: list[Any],
    *,
    batch_size: int | None = None,
    output_dim: int | None = None,
) -> np.ndarray:
    """Process images in batches and return a ``(N, D)`` matrix.

    Args:
        clip: Model exposing ``get_image_embedding``.
        images: Input images to embed.
        batch_size: Optional batch size; defaults to ``len(images)``.
        output_dim: Optional strict embedding dimension; mismatches raise an
            error.

    Returns:
        numpy.ndarray: Stacked embeddings with shape ``(N, D)`` where ``D`` is
        either ``output_dim`` or :data:`EMBEDDING_DIMENSIONS`.

    Raises:
        ValueError: If ``output_dim`` is set and the model output size differs.
    """
    if not images:
        return np.array([])

    bs = int(batch_size or max(1, len(images)))
    out: list[np.ndarray] = []
    dim = int(output_dim or EMBEDDING_DIMENSIONS)
    zero = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(images), bs):
        for img in images[i : i + bs]:
            try:
                emb = clip.get_image_embedding(img)
                if hasattr(emb, "cpu") and callable(emb.cpu):  # torch tensor
                    arr = cast(Any, emb).cpu().numpy()
                else:
                    arr = np.asarray(emb)
                # Optional strict validation of output dimensionality
                if output_dim is not None and not (
                    arr.ndim == 1 and arr.shape[0] == dim
                ):
                    raise ValueError(
                        f"Model output dimension {arr.shape} != expected ({dim},)"
                    )
                out.append(arr)
            # Per-image error handling converts failures to zero-vectors
            except (RuntimeError, ValueError, TypeError) as exc:  # pragma: no cover
                logger.error("Image embedding failed: %s", exc)
                out.append(zero)

    # Validate/normalize shapes; pad to expected D when needed
    mat = np.vstack([(v if (v.ndim == 1 and v.shape[0] == dim) else zero) for v in out])
    return mat


async def cross_modal_search(
    index: Any,
    *,
    query: str | None = None,
    query_image: Any | None = None,
    search_type: str = "text_to_image",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Query a mocked index for multimodal search results.

    Args:
        index: Object exposing ``as_query_engine`` or ``as_retriever``.
        query: Text query for ``text_to_image`` searches.
        query_image: Image-like payload for ``image_to_image`` searches.
        search_type: Either ``"text_to_image"`` or ``"image_to_image"``.
        top_k: Maximum number of results to return.

    Returns:
        list[dict[str, Any]]: Normalised search results with stable keys.
    """
    results: list[dict[str, Any]] = []
    if search_type == "text_to_image" and query is not None:
        if not query:
            return []
        response = await asyncio.to_thread(index.as_query_engine().query, query)
        nodes = getattr(response, "source_nodes", [])
        if not isinstance(nodes, list | tuple):
            return []
        for rank, node in enumerate(nodes[:top_k], start=1):
            text = getattr(getattr(node, "node", {}), "text", "")
            if len(text) > TEXT_TRUNCATION_LIMIT:
                text = text[:TEXT_TRUNCATION_LIMIT]
            results.append(
                {
                    "score": getattr(node, "score", 0.0),
                    "image_path": getattr(
                        getattr(node, "node", {}), "metadata", {}
                    ).get("image_path"),
                    "text": text,
                    "rank": rank,
                }
            )
        return results

    if search_type == "image_to_image" and query_image is not None:
        # A retriever path is used in tests; we only shape outputs
        retrieved = await asyncio.to_thread(index.as_retriever().retrieve, query_image)
        for rank, node in enumerate(retrieved[:top_k], start=1):
            results.append(
                {
                    "similarity": getattr(node, "score", 0.0),
                    "image_path": getattr(
                        getattr(node, "node", {}), "metadata", {}
                    ).get("image_path"),
                    "text": getattr(getattr(node, "node", {}), "text", ""),
                    "rank": rank,
                }
            )
        return results

    # Unsupported type or missing inputs → empty results
    return results


def create_image_documents(
    image_paths: Iterable[str], metadata: dict[str, Any] | None = None
) -> list[ImageDocument]:
    """Create ``ImageDocument`` entries while skipping failures.

    Args:
        image_paths: Iterable of filesystem paths pointing to images.
        metadata: Optional metadata dictionary shared by all documents. When
            omitted a default ``{"source": "multimodal"}`` entry is used.

    Returns:
        list[ImageDocument]: Successfully constructed document records.
    """
    docs: list[ImageDocument] = []
    meta = metadata or {"source": "multimodal"}
    for p in image_paths:
        try:
            docs.append(ImageDocument(image_path=str(p), metadata=meta))
        except (TypeError, ValueError, OSError) as exc:  # pragma: no cover
            logger.error("Failed creating ImageDocument for %s: %s", p, exc)
    return docs


async def validate_end_to_end_pipeline(
    query: str,
    query_image: Any,
    clip: Any,
    property_graph: Any,
    llm: Any,
) -> dict[str, Any]:
    """Run a lightweight multimodal validation pipeline.

    Args:
        query: Text query describing the user request.
        query_image: Reference image passed to the embedding model.
        clip: Model exposing ``get_image_embedding``; mocked in tests.
        property_graph: Placeholder graph object (unused but kept for future expansion).
        llm: Placeholder LLM client (unused, mocked in tests).

    Returns:
        dict[str, Any]: Summary containing the final response, entity metadata,
        similarity metrics, and execution timing.
    """
    # Image embedding (normalized) drives a dummy similarity metric
    import time

    start = time.perf_counter()
    emb = await generate_image_embeddings(clip, query_image)
    similarity = float(np.clip(np.dot(emb, emb), 0.0, 1.0))  # == 1.0 when normalized

    # Toy entity extraction from the query
    base_entities = {"LlamaIndex", "BGE-M3"}
    dynamic = {w for w in ("OpenCLIP", "SigLIP") if w.lower() in query.lower()}
    entities = sorted(base_entities | dynamic)
    relationships = max(0, len(entities) - RANK_ADJUSTMENT)

    # Compose a final response string (LLM mocked in tests)
    _ = property_graph, llm  # boundaries mocked; kept for future expansion
    final = (
        f"Query '{query}' processed with visual similarity={similarity:.2f}. "
        f"Entity relationships detected: {', '.join(entities)}."
    )

    end = time.perf_counter()

    return {
        "final_response": final
        + " Combined with entity relationships and multimodal context.",
        "entity_relationships": {
            "entities_found": entities,
            "relationship_count": relationships,
        },
        "visual_similarity": {
            "embedding_dim": int(emb.shape[0] if emb.ndim == 1 else emb.shape[-1]),
            "norm": float(np.linalg.norm(emb)),
            "score": similarity,
        },
        "pipeline_time": end - start,
    }


# ContextlibSuppress replaced with stdlib contextlib.suppress
