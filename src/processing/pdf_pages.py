"""PDF page image emission utilities.

Generates page-image artifacts for PDFs with stable filenames and bounding
boxes, and provides a convenience converter to LlamaIndex ImageDocument nodes
for downstream pipeline usage.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import numpy as np
from llama_index.core.schema import ImageDocument
from PIL import Image  # type: ignore


def _phash(img: Image.Image, hash_size: int = 8) -> str:
    """Compute a simple perceptual hash (average hash) for dedup hints.

    This avoids external deps; not a full DCT pHash but sufficient for duplicates.
    """
    gray = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    avg = arr.mean()
    bits = (arr > avg).astype(np.uint8).flatten()
    # Pack bits into hex string
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def _save_with_format(pix: fitz.Pixmap, target_stem: Path) -> tuple[Path, str]:
    """Save pixmap as WebP (preferred) or JPEG fallback. Returns (path, phash)."""
    # Convert to PIL Image from raw samples
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")

    # Resize to long-edge ~2000 px
    long_edge = max(img.width, img.height)
    if long_edge > 2000:
        img.thumbnail((2000, 2000), Image.LANCZOS)

    # Try WebP first
    webp_path = target_stem.with_suffix(".webp")
    try:
        img.save(webp_path, format="WEBP", quality=70, method=6)
        return webp_path, _phash(img)
    except Exception:
        # Fallback to JPEG
        jpg_path = target_stem.with_suffix(".jpg")
        img.save(jpg_path, format="JPEG", quality=75)
        return jpg_path, _phash(img)


def _render_pdf_pages(
    pdf_path: Path, out_dir: Path, dpi: int = 200
) -> list[tuple[int, Path, fitz.Rect, str]]:
    """Render each page to a stable PNG file, refreshing as needed.

    - Stable filename: ``<stem>__page-<n>.png`` (1-based)
    - Idempotent but refreshes images if the source PDF is newer than existing PNGs.

    Returns list of tuples ``(page_no, image_path, page_rect)``.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, Path, fitz.Rect, str]] = []
    pdf_mtime = pdf_path.stat().st_mtime if pdf_path.exists() else 0.0

    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for idx, page in enumerate(doc, start=1):
            img_stem = f"{pdf_path.stem}__page-{idx}"
            # Paths depend on chosen format; compute stem only here
            img_path = out_dir / (img_stem + ".webp")

            # Refresh if missing or source PDF is newer
            needs_render = True
            if img_path.exists():
                try:
                    needs_render = img_path.stat().st_mtime < pdf_mtime
                except OSError:
                    needs_render = True

            if needs_render:
                pix = page.get_pixmap(matrix=mat)
                # Save as WebP or JPEG fallback
                img_path, phash = _save_with_format(pix, out_dir / img_stem)
                # Ensure deterministic mtime ordering for downstream caches/tests:
                # set the image mtime to at least the source PDF's mtime.
                with contextlib.suppress(OSError):
                    os.utime(img_path, (pdf_mtime, pdf_mtime))
                results.append((idx, img_path, page.rect, phash))
            else:
                # If not re-rendered, recompute phash on the fly for metadata
                try:
                    with Image.open(img_path) as im:
                        ph = _phash(im)
                except Exception:
                    ph = ""
                results.append((idx, img_path, page.rect, ph))

    return results


def pdf_pages_to_image_documents(
    pdf_path: Path, dpi: int = 200, output_dir: Path | None = None
) -> tuple[list[ImageDocument], Path]:
    """Render PDF pages to images and return ImageDocument nodes.

    Args:
        pdf_path: Path to the PDF file
        dpi: Render resolution (dots per inch)
        output_dir: Directory to save images. Created if ``None``.

    Returns:
        Tuple of ImageDocuments and the directory containing the images.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp())
    entries = _render_pdf_pages(pdf_path, out_dir, dpi)
    docs: list[ImageDocument] = []

    for i, path, _rect, phash in entries:
        meta: dict[str, Any] = {
            "page": i,
            "modality": "pdf_page_image",
            "source": str(pdf_path),
            "phash": phash,
        }
        docs.append(ImageDocument(image_path=str(path), metadata=meta))

    return docs, out_dir


def save_pdf_page_images(pdf_path: Path, out_dir: Path, dpi: int = 200) -> list[dict]:
    """Render each PDF page to a PNG with a stable filename and return metadata.

    - Stable filename format: ``<stem>__page-<n>.png`` (1-based page numbering)
    - Returns one item per page with page number, image path, and page bbox
    - Idempotent: existing PNG files are not re-written

    Args:
        pdf_path: Path to the source PDF
        out_dir: Directory to store generated images
        dpi: Render resolution (dots per inch)

    Returns:
        A list of dicts: {"page_no": int, "image_path": str, "bbox": [x0,y0,x1,y1]}
    """
    entries = _render_pdf_pages(Path(pdf_path), Path(out_dir), dpi)

    return [
        {
            "page_no": idx,
            "image_path": str(path),
            "bbox": [float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1)],
            "phash": phash,
        }
        for idx, path, rect, phash in entries
    ]


__all__ = ["pdf_pages_to_image_documents", "save_pdf_page_images"]
