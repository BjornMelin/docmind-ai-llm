"""PDF page image emission utilities.

Generates page-image artifacts for PDFs with stable filenames and bounding
boxes, and provides a convenience converter to LlamaIndex ImageDocument nodes
for downstream pipeline usage.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, cast

import fitz  # PyMuPDF
import numpy as np
from llama_index.core.schema import ImageDocument
from loguru import logger
from PIL import Image  # type: ignore
from PIL.Image import Resampling  # type: ignore

from src.config import settings
from src.persistence.artifacts import ArtifactStore
from src.utils.security import encrypt_file, get_image_kid

_PAGE_TEXT_MAX_CHARS = 8000


def _phash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash (average hash) for deduplication.

    Args:
        img: Image to hash.
        hash_size: Width/height of the hash grid; defaults to 8.

    Returns:
        str: Hex-encoded average hash suitable for duplicate detection hints.
    """
    gray = img.convert("L").resize((hash_size, hash_size), Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    avg = arr.mean()
    bits = (arr > avg).astype(np.uint8).flatten()
    # Pack bits into hex string
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def _save_with_format(
    pix: fitz.Pixmap, target_stem: Path, *, encrypt: bool | None = None
) -> tuple[Path, str]:
    """Persist a rendered page as WebP (preferred) or JPEG fallback.

    Encryption is applied when ``encrypt`` is true (defaulting to
    ``settings.processing.encrypt_page_images``), yielding ``*.enc`` outputs.

    Args:
        pix: PyMuPDF pixmap for the rendered page.
        target_stem: Path stem used to derive the output filename.
        encrypt: Whether to encrypt the rendered output; ``None`` uses settings.

    Returns:
        tuple[Path, str]: Output path (possibly encrypted) and perceptual hash.
    """
    if encrypt is None:
        encrypt = getattr(settings.processing, "encrypt_page_images", False)

    # Convert to PIL Image from raw samples
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    # Strip EXIF/metadata explicitly
    if hasattr(img, "info"):
        img.info.pop("exif", None)
    if mode == "RGBA":
        img = img.convert("RGB")

    # Resize to long-edge ~2000 px
    long_edge = max(img.width, img.height)
    if long_edge > 2000:
        img.thumbnail((2000, 2000), Resampling.LANCZOS)

    # Try WebP first
    webp_path = target_stem.with_suffix(".webp")
    try:
        img.save(webp_path, format="WEBP", quality=70, method=6)
        out = str(webp_path)
        if encrypt:
            out = encrypt_file(out)
        return Path(out), _phash(img)
    except (OSError, ValueError):
        # Fallback to JPEG
        jpg_path = target_stem.with_suffix(".jpg")
        img.save(jpg_path, format="JPEG", quality=75)
        out = str(jpg_path)
        if encrypt:
            out = encrypt_file(out)
        return Path(out), _phash(img)


def _render_pdf_pages(
    pdf_path: Path, out_dir: Path, dpi: int = 200, *, encrypt: bool | None = None
) -> list[tuple[int, Path, fitz.Rect, str, str]]:
    """Render PDF pages to image files while preserving deterministic names.

    Output filenames follow the ``<stem>__page-<n>`` convention and reuse
    existing images unless the source PDF has changed. When encryption is
    enabled, rendered files are suffixed with ``.enc``.

    Args:
        pdf_path: Source PDF path.
        out_dir: Directory to store rendered images.
        dpi: Render resolution in dots per inch; defaults to 200.
        encrypt: Whether to encrypt rendered outputs; ``None`` defers to settings.

    Returns:
        list[tuple[int, Path, fitz.Rect, str, str]]: One entry per page containing
        the 1-based page number, output image path, page rectangle, phash, and
        extracted page text (best-effort).
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, Path, fitz.Rect, str, str]] = []
    pdf_mtime = pdf_path.stat().st_mtime if pdf_path.exists() else 0.0

    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for idx in range(doc.page_count):
            page_num = idx + 1
            page = doc.load_page(idx)
            # Best-effort page text extraction for downstream grounding.
            try:
                page_text = (cast(Any, page).get_text("text") or "").strip()
            except Exception:  # pragma: no cover - PyMuPDF quirks
                page_text = ""
            if len(page_text) > _PAGE_TEXT_MAX_CHARS:
                page_text = page_text[:_PAGE_TEXT_MAX_CHARS]
            img_stem = f"{pdf_path.stem}__page-{page_num}"
            # Paths depend on chosen format/encryption; select an existing file
            # if present to avoid re-render loops for encrypted outputs.
            base = out_dir / img_stem
            candidates = [
                base.with_suffix(".webp.enc"),
                base.with_suffix(".webp"),
                base.with_suffix(".jpg.enc"),
                base.with_suffix(".jpg"),
                base.with_suffix(".jpeg.enc"),
                base.with_suffix(".jpeg"),
            ]
            existing = next((p for p in candidates if p.exists()), None)
            img_path = existing or base.with_suffix(".webp")

            # Refresh if missing, source PDF is newer, or encryption flag changed.
            # Otherwise we can incorrectly reuse plaintext when encryption is enabled
            # (or vice versa), breaking expectations and downstream consumers.
            wants_encrypt = (
                getattr(settings.processing, "encrypt_page_images", False)
                if encrypt is None
                else bool(encrypt)
            )
            needs_render = True
            if existing is not None:
                existing_is_enc = img_path.name.endswith(".enc")
                try:
                    needs_render = (img_path.stat().st_mtime < pdf_mtime) or (
                        existing_is_enc != wants_encrypt
                    )
                except OSError:
                    needs_render = True

            if needs_render:
                pix = cast(Any, page).get_pixmap(matrix=mat)
                # Save as WebP or JPEG fallback
                img_path, phash = _save_with_format(
                    pix, out_dir / img_stem, encrypt=encrypt
                )
                # Ensure deterministic mtime ordering for downstream caches/tests:
                # set the image mtime to at least the source PDF's mtime.
                with contextlib.suppress(OSError):
                    os.utime(img_path, (pdf_mtime, pdf_mtime))
                results.append((page_num, img_path, page.rect, phash, page_text))
            else:
                # If not re-rendered, recompute phash on the fly for metadata.
                # Handle encrypted images by decrypting to a temporary file.
                ph = ""
                try:
                    from src.utils.images import open_image_encrypted

                    with open_image_encrypted(str(img_path)) as im:
                        ph = _phash(im) if im is not None else ""
                except (OSError, ValueError, RuntimeError):
                    ph = ""
                results.append((page_num, img_path, page.rect, ph, page_text))

    return results


def pdf_pages_to_image_documents(
    pdf_path: Path,
    dpi: int = 200,
    output_dir: Path | None = None,
    document_id: str | None = None,
    *,
    encrypt: bool | None = None,
) -> tuple[list[ImageDocument], Path]:
    """Render PDF pages to images and return ImageDocument nodes.

    Args:
        pdf_path: Path to the PDF file
        dpi: Render resolution (dots per inch)
        output_dir: Directory to save images. Created if ``None``.
        document_id: Optional stable document identifier for metadata linkage.
        encrypt: Whether to encrypt rendered outputs (defaults to settings).

    Returns:
        tuple[list[ImageDocument], Path]: Generated image documents and the
        directory containing the rendered assets.
    """
    pdf_path = Path(pdf_path)
    if output_dir:
        out_dir = Path(output_dir)
    else:
        # Prefer a stable cache location under settings.cache_dir rather than a
        # temp directory that could leak in logs/persistence.
        out_dir = settings.cache_dir / "page_images" / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = _render_pdf_pages(pdf_path, out_dir, dpi, encrypt=encrypt)
    docs: list[ImageDocument] = []
    failed_pages: list[tuple[int, str]] = []
    # Store rendered images as content-addressed artifacts and reference jailed
    # artifact paths in ImageDocument nodes.
    store = ArtifactStore.from_settings(settings)
    doc_id = str(document_id or pdf_path.stem or "document")

    for i, path, _rect, phash, page_text in entries:
        try:
            ref = store.put_file(Path(path))
        except (OSError, ValueError) as exc:
            logger.exception(
                "ArtifactStore.put_file failed for PDF page image "
                "(page={}, path={}, phash={})",
                i,
                path,
                phash,
            )
            failed_pages.append((i, f"put_file: {exc}"))
            continue

        try:
            resolved_path = store.resolve_path(ref)
        except (OSError, ValueError) as exc:
            logger.exception(
                "ArtifactStore.resolve_path failed for PDF page image "
                "(page={}, path={}, phash={}, ref={})",
                i,
                path,
                phash,
                ref,
            )
            failed_pages.append((i, f"resolve_path: {exc}"))
            continue

        meta: dict[str, Any] = {
            "page": i,
            "modality": "pdf_page_image",
            # Avoid persisting raw filesystem paths in metadata.
            "source": pdf_path.name,
            "source_filename": pdf_path.name,
            "phash": phash,
            "page_text": page_text,
        }
        if doc_id:
            meta.update(
                {
                    "doc_id": doc_id,
                    "document_id": doc_id,
                    "page_id": f"{doc_id}::page::{i}",
                }
            )
        if str(path).endswith(".enc"):
            meta.update(
                {
                    "encrypted": True,
                    "kid": get_image_kid(),
                }
            )
        meta.update(
            {
                "image_artifact_id": ref.sha256,
                "image_artifact_suffix": ref.suffix,
            }
        )
        docs.append(ImageDocument(image_path=str(resolved_path), metadata=meta))

    if failed_pages:
        logger.warning(
            "Failed to store {} page(s): {}", len(failed_pages), failed_pages
        )
    return docs, out_dir


def save_pdf_page_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    *,
    encrypt: bool | None = None,
) -> list[dict]:
    """Render each PDF page to a stable image (WebP/JPEG) and return metadata.

    - Stable filename format: ``<stem>__page-<n>.<webp|jpg>``; encrypted outputs
      end with ``.enc`` when AES-GCM is enabled.
    - Returns one item per page with page number, image path, bbox, phash, and
      optional encryption metadata (kid).

    Args:
        pdf_path: Path to the source PDF
        out_dir: Directory to store generated images
        dpi: Render resolution (dots per inch)
        encrypt: Whether to encrypt rendered outputs; ``None`` defers to settings.

    Returns:
        list[dict]: Page metadata containing image path, bbox, phash, and flags.
    """
    entries = _render_pdf_pages(Path(pdf_path), Path(out_dir), dpi, encrypt=encrypt)

    return [
        {
            "page_no": idx,
            "image_path": str(path),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "phash": phash,
            "page_text": page_text,
            **(
                {"encrypted": True, "kid": get_image_kid()}
                if str(path).endswith(".enc")
                else {}
            ),
        }
        for idx, path, rect, phash, page_text in entries
    ]


__all__ = ["pdf_pages_to_image_documents", "save_pdf_page_images"]
