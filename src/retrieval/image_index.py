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
from typing import Any, cast

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from qdrant_client.http.models import Distance

from src.config import settings
from src.config.embedding_defaults import (
    DEFAULT_SIGLIP_MODEL_ID,
    DEFAULT_SIGLIP_MODEL_REVISION,
)
from src.persistence.artifacts import ArtifactRef
from src.persistence.deployment_identity import (
    get_or_create_deployment_id,
    read_deployment_id,
)
from src.utils.images import open_image_encrypted
from src.utils.qdrant_utils import get_collection_params

_SIGLIP_VECTOR_NAME = "siglip"
_DEFAULT_SIGLIP_DIM = 768
_IMAGE_COLLECTION_SCHEMA_VERSION = "2"
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


class ImageCollectionIncompatibleError(RuntimeError):
    """Raised when an image collection does not match its model identity."""


def canonical_image_collection_metadata(*, dim: int) -> dict[str, Any]:
    """Return immutable semantic identity for a staged image collection."""
    return _image_collection_metadata(
        dim=dim,
        deployment_id=get_or_create_deployment_id(settings.data_dir),
    )


def _expected_image_collection_metadata(*, dim: int) -> dict[str, Any]:
    """Return read-only expected metadata for compatibility checks."""
    return _image_collection_metadata(
        dim=dim,
        deployment_id=read_deployment_id(settings.data_dir),
    )


def _image_collection_metadata(*, dim: int, deployment_id: str) -> dict[str, Any]:
    """Build image collection metadata from an explicit deployment identity."""
    model_id = str(settings.embedding.siglip_model_id)
    revision = settings.embedding.siglip_model_revision
    if revision is None and model_id == DEFAULT_SIGLIP_MODEL_ID:
        revision = DEFAULT_SIGLIP_MODEL_REVISION
    return {
        "docmind_deployment_id": deployment_id,
        "docmind_owner": "image",
        "docmind_schema_version": _IMAGE_COLLECTION_SCHEMA_VERSION,
        "siglip_model": model_id,
        "siglip_revision": str(revision or "unpinned"),
        "siglip_dimension": int(dim),
        "normalize_image": bool(settings.embedding.normalize_image),
    }


def check_siglip_image_collection(
    client: QdrantClient,
    collection_name: str,
    *,
    vector_name: str = _SIGLIP_VECTOR_NAME,
    dim: int = _DEFAULT_SIGLIP_DIM,
) -> None:
    """Validate an existing image collection without mutating it."""
    if not client.collection_exists(collection_name):
        raise ImageCollectionIncompatibleError("image_collection_missing")
    info = client.get_collection(collection_name)
    params = get_collection_params(client, collection_name)
    vectors = getattr(params, "vectors", None) or getattr(
        params, "vectors_config", None
    )
    if not isinstance(vectors, dict) or vector_name not in vectors:
        raise ImageCollectionIncompatibleError("siglip_vector_missing")
    vector = vectors[vector_name]
    if int(getattr(vector, "size", 0)) != int(dim):
        raise ImageCollectionIncompatibleError("siglip_dimension_mismatch")
    if getattr(vector, "distance", None) != Distance.COSINE:
        raise ImageCollectionIncompatibleError("siglip_distance_mismatch")
    actual_metadata = getattr(getattr(info, "config", None), "metadata", None)
    expected_metadata = _expected_image_collection_metadata(dim=dim)
    if not isinstance(actual_metadata, dict):
        raise ImageCollectionIncompatibleError("image_collection_metadata_missing")
    if any(
        actual_metadata.get(key) != value for key, value in expected_metadata.items()
    ):
        raise ImageCollectionIncompatibleError("image_collection_metadata_mismatch")


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
            metadata=canonical_image_collection_metadata(dim=dim),
        )
        logger.info("Created image collection '{}' (dim={})", collection_name, int(dim))
        return

    check_siglip_image_collection(
        client,
        collection_name,
        vector_name=vector_name,
        dim=dim,
    )


def _siglip_expected_dim(embedder: Any) -> int:
    expected = _DEFAULT_SIGLIP_DIM
    ensure_loaded = getattr(embedder, "_ensure_loaded", None)
    if callable(ensure_loaded):
        with contextlib.suppress(Exception):
            ensure_loaded()
    with contextlib.suppress(Exception):
        expected = int(
            getattr(embedder, "_dim", None)
            or getattr(embedder, "dim", None)
            or _DEFAULT_SIGLIP_DIM
        )
    return expected


def _page_image_record_identifier(rec: PageImageRecord) -> str:
    for attr in ("id", "page_id"):
        val = getattr(rec, attr, None)
        if val:
            return str(val)
    image_path = getattr(rec, "image_path", None)
    if image_path:
        return str(Path(image_path).name)
    doc_id = getattr(rec, "doc_id", None)
    if doc_id:
        return str(doc_id)
    return "<unknown>"


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

    expected_dim = _siglip_expected_dim(embedder)
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
            client.set_payload(
                collection_name=collection_name,
                points=[pid],
                payload=payload,
                wait=True,
            )

    indexed = len(to_payload_only)
    batch_size_int = max(1, int(batch_size))
    for i in range(0, len(to_embed), batch_size_int):
        batch = to_embed[i : i + batch_size_int]
        imgs = [_load_rgb_image(r.image_path) for r in batch]
        vecs = embedder.get_image_embeddings(imgs, batch_size=len(imgs))
        if not isinstance(vecs, np.ndarray):
            vecs = np.asarray(vecs, dtype=np.float32)

        expected = len(batch)
        got = len(vecs)
        matched = min(expected, got)
        if expected != got:
            skipped_ids = [_page_image_record_identifier(r) for r in batch[matched:]]
            logger.warning(
                "Embedding count mismatch; expected len(batch)={} got len(vecs)={}; "
                "indexing {} and skipping {} records: {}",
                expected,
                got,
                matched,
                len(skipped_ids),
                skipped_ids,
            )

        points: list[qmodels.PointStruct] = []
        for r, vec in zip(batch[:matched], vecs[:matched], strict=False):
            pid = r.point_id()
            points.append(
                qmodels.PointStruct(
                    id=pid,
                    vector={_SIGLIP_VECTOR_NAME: vec.tolist()},
                    payload=_build_payload(r),
                )
            )

        if points:
            client.upsert(collection_name=collection_name, points=points, wait=True)
            indexed += len(points)

    return indexed


def collect_page_image_point_ids_for_doc_id(
    client: QdrantClient, collection_name: str, *, doc_id: str
) -> set[str]:
    """Return every page-image point ID owned by a document.

    Raises:
        RuntimeError: If Qdrant returns a point without an ID or stalls while
            paginating the document's points.
    """
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_id", match=qmodels.MatchValue(value=str(doc_id))
            )
        ]
    )
    point_ids: set[str] = set()
    offset: Any | None = None
    seen_offsets: set[str] = set()
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        for point in points or []:
            point_id = getattr(point, "id", None)
            if point_id is None:
                raise RuntimeError("Qdrant returned a page-image point without an ID")
            point_ids.add(str(point_id))
        if next_offset is None:
            return point_ids
        offset_key = str(next_offset)
        if offset_key in seen_offsets:
            raise RuntimeError("Qdrant page-image pagination stalled")
        seen_offsets.add(offset_key)
        offset = next_offset


def delete_page_image_points_by_id(
    client: QdrantClient,
    collection_name: str,
    *,
    point_ids: set[str],
) -> int:
    """Delete an exact set of stale page-image points and verify removal."""
    if not point_ids:
        return 0
    ordered_ids = sorted(point_ids)
    client.delete(
        collection_name=collection_name,
        points_selector=qmodels.PointIdsList(points=cast(Any, ordered_ids)),
        wait=True,
    )
    remaining = client.retrieve(
        collection_name=collection_name,
        ids=ordered_ids,
        with_payload=False,
        with_vectors=False,
    )
    if remaining:
        raise RuntimeError("Stale page-image deletion left indexed points behind")
    return len(ordered_ids)


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
    "ImageCollectionIncompatibleError",
    "PageImageRecord",
    "canonical_image_collection_metadata",
    "check_siglip_image_collection",
    "collect_artifact_refs_for_doc_id",
    "collect_page_image_point_ids_for_doc_id",
    "count_artifact_references_in_image_collection",
    "count_page_images_for_doc_id",
    "delete_page_image_points_by_id",
    "ensure_siglip_image_collection",
    "index_page_images_siglip",
]
