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
from llama_index.core.schema import ImageDocument


def _render_pdf_pages(
    pdf_path: Path, out_dir: Path, dpi: int = 180
) -> list[tuple[int, Path, fitz.Rect]]:
    """Render each page to a stable PNG file, refreshing as needed.

    - Stable filename: ``<stem>__page-<n>.png`` (1-based)
    - Idempotent but refreshes images if the source PDF is newer than existing PNGs.

    Returns list of tuples ``(page_no, image_path, page_rect)``.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[int, Path, fitz.Rect]] = []
    pdf_mtime = pdf_path.stat().st_mtime if pdf_path.exists() else 0.0

    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for idx, page in enumerate(doc, start=1):
            img_name = f"{pdf_path.stem}__page-{idx}.png"
            img_path = out_dir / img_name

            # Refresh if missing or source PDF is newer
            needs_render = True
            if img_path.exists():
                try:
                    needs_render = img_path.stat().st_mtime < pdf_mtime
                except OSError:
                    needs_render = True

            if needs_render:
                page.get_pixmap(matrix=mat).save(str(img_path))
                # Ensure deterministic mtime ordering for downstream caches/tests:
                # set the image mtime to at least the source PDF's mtime.
                with contextlib.suppress(OSError):
                    os.utime(img_path, (pdf_mtime, pdf_mtime))

            results.append((idx, img_path, page.rect))

    return results


def pdf_pages_to_image_documents(
    pdf_path: Path, dpi: int = 180, output_dir: Path | None = None
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

    for i, path, _rect in entries:
        meta: dict[str, Any] = {
            "page": i,
            "modality": "pdf_page_image",
            "source": str(pdf_path),
        }
        docs.append(ImageDocument(image_path=str(path), metadata=meta))

    return docs, out_dir


def save_pdf_page_images(pdf_path: Path, out_dir: Path, dpi: int = 180) -> list[dict]:
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
        }
        for idx, path, rect in entries
    ]


__all__ = ["pdf_pages_to_image_documents", "save_pdf_page_images"]
