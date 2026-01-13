"""Image indexing helpers for multimodal retrieval.

Indexes PDF page images into a dedicated Qdrant collection using SigLIP
cross-modal embeddings (text<->image).

Constraint: Qdrant payload must be **thin** and must not contain
base64 blobs or raw filesystem paths. We store content-addressed artifact
references (sha256 + suffix) and resolve to local paths at runtime.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.models import Distance

from src.persistence.artifacts import ArtifactRef
from src.utils.images import open_image_encrypted
from src.utils.qdrant_utils import get_collection_params

_SIGLIP_VECTOR_NAME = "siglip"
_DEFAULT_SIGLIP_DIM = 768
_UUID_NAMESPACE = uuid.UUID("d3b17330-1e80-4c4f-9f5d-9f2a1432f6cf")


@dataclass(frozen=True, slots=True)
class PageImageRecord:
    """Record describing a single PDF page image to index in Qdrant."""

    doc_id: str
    page_no: int
    image: ArtifactRef
    image_path: Path
    thumbnail: ArtifactRef | None = None
    thumbnail_path: Path | None = None
    phash: str | None = None
    page_text: str | None = None
    bbox: list[float] | None = None

    def point_id(self) -> uuid.UUID:
        """Return a deterministic point ID for (doc_id, page_no)."""
        return uuid.uuid5(_UUID_NAMESPACE, f"{self.doc_id}::page::{self.page_no}")


def ensure_siglip_image_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    vector_name: str = _SIGLIP_VECTOR_NAME,
    dim: int = _DEFAULT_SIGLIP_DIM,
) -> None:
    """Ensure the image collection exists with the expected SigLIP vector head."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                vector_name: qmodels.VectorParams(
                    size=int(dim), distance=Distance.COSINE
                ),
            },
        )
        logger.info("Created image collection '%s' (dim=%d)", collection_name, dim)
        return

    # Patch missing vector head if needed (idempotent).
    try:
        params = get_collection_params(client, collection_name)
        vec_cfg = getattr(params, "vectors", None) or getattr(
            params, "vectors_config", None
        )
        if not isinstance(vec_cfg, dict) or vector_name not in vec_cfg:
            # NOTE: qdrant-client's `update_collection(vectors_config=...)` only
            # supports VectorParamsDiff (no size/distance), so it cannot add new
            # named vector heads. If the collection was created without the
            # expected vector, the safest path is to ask the operator to
            # recreate it.
            logger.warning(
                "Image collection '%s' is missing vector head '%s'; "
                "recreate the collection to enable image indexing",
                collection_name,
                vector_name,
            )
            return

        vec_params = vec_cfg.get(vector_name)
        existing_dim = getattr(vec_params, "size", None)
        if existing_dim is None and isinstance(vec_params, dict):
            existing_dim = vec_params.get("size")
        if existing_dim is not None and int(existing_dim) != int(dim):
            logger.error(
                "Image collection '%s' vector '%s' has dim=%d but expected dim=%d; "
                "recreate the collection to enable image indexing",
                collection_name,
                vector_name,
                int(existing_dim),
                int(dim),
            )
            raise ValueError(
                "Image collection dimension mismatch; recreate the collection "
                f"(collection={collection_name}, vector={vector_name}, "
                f"expected_dim={int(dim)}, actual_dim={int(existing_dim)})"
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("ensure_siglip_image_collection skipped: %s", exc)


def index_page_images_siglip(
    client: QdrantClient,
    collection_name: str,
    records: list[PageImageRecord],
    *,
    embedder: Any,
    batch_size: int = 8,
) -> int:
    """Upsert SigLIP embeddings for page images; returns number indexed."""
    if not records:
        return 0

    expected_dim = _DEFAULT_SIGLIP_DIM
    ensure_loaded = getattr(embedder, "_ensure_loaded", None)
    if callable(ensure_loaded):
        with contextlib.suppress(Exception):
            ensure_loaded()
    with contextlib.suppress(Exception):
        expected_dim = int(
            getattr(embedder, "_dim", None)
            or getattr(embedder, "dim", None)
            or _DEFAULT_SIGLIP_DIM
        )

    ensure_siglip_image_collection(client, collection_name, dim=expected_dim)

    # Best-effort short-circuit when existing phash matches (avoid re-embed).
    existing_phash: dict[str, str] = {}
    try:
        pts = client.retrieve(
            collection_name=collection_name,
            ids=[r.point_id() for r in records],
            with_payload=["phash"],
        )
        for p in pts or []:
            payload = getattr(p, "payload", {}) or {}
            ph = payload.get("phash")
            if ph:
                existing_phash[str(getattr(p, "id", ""))] = str(ph)
    except Exception:
        existing_phash = {}

    to_embed: list[PageImageRecord] = []
    to_payload_only: list[tuple[uuid.UUID, dict[str, Any]]] = []

    for r in records:
        pid = r.point_id()
        payload = _build_payload(r)
        cur = existing_phash.get(str(pid))
        if cur and r.phash and str(r.phash) == str(cur):
            to_payload_only.append((pid, payload))
        else:
            to_embed.append(r)

    if to_payload_only:
        for pid, payload in to_payload_only:
            with contextlib.suppress(Exception):
                client.set_payload(
                    collection_name=collection_name,
                    points=[pid],
                    payload=payload,
                )

    indexed = 0
    for i in range(0, len(to_embed), max(1, int(batch_size))):
        batch = to_embed[i : i + max(1, int(batch_size))]
        imgs = [_load_rgb_image(r.image_path) for r in batch]
        vecs = embedder.get_image_embeddings(imgs, batch_size=len(imgs))
        if not isinstance(vecs, np.ndarray):
            vecs = np.asarray(vecs, dtype=np.float32)
        points: list[qmodels.PointStruct] = []
        if len(vecs) != len(batch):
            logger.warning(
                "Embedding count mismatch: expected %d, got %d",
                len(batch),
                len(vecs),
            )
            continue
        for r, vec in zip(batch, vecs, strict=False):
            pid = r.point_id()
            points.append(
                qmodels.PointStruct(
                    id=pid,
                    vector={_SIGLIP_VECTOR_NAME: vec.tolist()},
                    payload=_build_payload(r),
                )
            )
        client.upsert(collection_name=collection_name, points=points)
        indexed += len(points)

    return indexed


def count_page_images_for_doc_id(
    client: QdrantClient, collection_name: str, *, doc_id: str
) -> int:
    """Return number of indexed page images for a given document id (best-effort)."""
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=str(doc_id))
            )
        ]
    )
    try:
        result = client.count(
            collection_name=collection_name, count_filter=flt, exact=True
        )
        return int(getattr(result, "count", 0) or 0)
    except Exception:  # pragma: no cover - defensive
        return 0


def collect_artifact_refs_for_doc_id(
    client: QdrantClient, collection_name: str, *, doc_id: str
) -> set[ArtifactRef]:
    """Collect artifact refs referenced by page-image points for a doc (best-effort)."""
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=str(doc_id))
            )
        ]
    )
    refs: set[ArtifactRef] = set()
    try:
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=collection_name,
                scroll_filter=flt,
                limit=256,
                offset=offset,
                with_payload=[
                    "image_artifact_id",
                    "image_artifact_suffix",
                    "thumbnail_artifact_id",
                    "thumbnail_artifact_suffix",
                ],
                with_vectors=False,
            )
            for p in points or []:
                payload = getattr(p, "payload", {}) or {}
                img_id = payload.get("image_artifact_id")
                img_sfx = payload.get("image_artifact_suffix") or ""
                if img_id:
                    refs.add(ArtifactRef(sha256=str(img_id), suffix=str(img_sfx)))
                th_id = payload.get("thumbnail_artifact_id")
                th_sfx = payload.get("thumbnail_artifact_suffix") or ""
                if th_id:
                    refs.add(ArtifactRef(sha256=str(th_id), suffix=str(th_sfx)))
            if offset is None:
                break
    except Exception:  # pragma: no cover - defensive
        return refs
    return refs


def delete_page_images_for_doc_id(
    client: QdrantClient, collection_name: str, *, doc_id: str
) -> int:
    """Delete page-image points for a doc_id. Returns best-effort prior count."""
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=str(doc_id))
            )
        ]
    )
    prior = 0
    try:
        prior = count_page_images_for_doc_id(client, collection_name, doc_id=doc_id)
    except Exception:
        prior = 0
    try:
        client.delete(collection_name=collection_name, points_selector=flt, wait=True)
    except Exception:  # pragma: no cover - defensive
        return prior
    return prior


def count_artifact_references_in_image_collection(
    client: QdrantClient, collection_name: str, *, artifact_id: str
) -> int:
    """Count how many points reference the given artifact id (image or thumbnail)."""
    flt = qmodels.Filter(
        should=[
            qmodels.FieldCondition(
                key="image_artifact_id",
                match=qmodels.MatchValue(value=str(artifact_id)),
            ),
            qmodels.FieldCondition(
                key="thumbnail_artifact_id",
                match=qmodels.MatchValue(value=str(artifact_id)),
            ),
        ]
    )
    try:
        result = client.count(
            collection_name=collection_name, count_filter=flt, exact=True
        )
        return int(getattr(result, "count", 0) or 0)
    except Exception:  # pragma: no cover - defensive
        return 0


def _build_payload(r: PageImageRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "doc_id": r.doc_id,
        "page_no": int(r.page_no),
        "page_id": f"{r.doc_id}::page::{int(r.page_no)}",
        "modality": "pdf_page_image",
        "image_artifact_id": r.image.sha256,
        "image_artifact_suffix": r.image.suffix,
    }
    if r.thumbnail is not None:
        payload["thumbnail_artifact_id"] = r.thumbnail.sha256
        payload["thumbnail_artifact_suffix"] = r.thumbnail.suffix
    if r.phash:
        payload["phash"] = str(r.phash)
    if r.page_text:
        payload["text"] = str(r.page_text)
    if r.bbox is not None:
        payload["bbox"] = list(r.bbox)
    return payload


def _load_rgb_image(path: Path) -> Any:
    with open_image_encrypted(str(path)) as im:
        if im is None:  # pragma: no cover - defensive
            raise FileNotFoundError(str(path))
        return im.convert("RGB").copy()


__all__ = [
    "PageImageRecord",
    "collect_artifact_refs_for_doc_id",
    "count_artifact_references_in_image_collection",
    "count_page_images_for_doc_id",
    "delete_page_images_for_doc_id",
    "ensure_siglip_image_collection",
    "index_page_images_siglip",
]
