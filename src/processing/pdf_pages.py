"""PDF page image emission utilities.

Generates page-image artifacts for PDFs with stable filenames and bounding
boxes, and provides a convenience converter to LlamaIndex ImageDocument nodes
for downstream pipeline usage.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from llama_index.core.schema import ImageDocument


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
    docs: list[ImageDocument] = []

    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat)
            unique_id = uuid.uuid4().hex
            img_path = out_dir / f"{pdf_path.stem}_page_{i}_{unique_id}.png"
            pix.save(str(img_path))

            meta: dict[str, Any] = {
                "page": i,
                "modality": "pdf_page_image",
                "source": str(pdf_path),
            }
            docs.append(ImageDocument(image_path=str(img_path), metadata=meta))

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
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for idx, page in enumerate(doc, start=1):
            # Stable filename, idempotent write behavior
            img_name = f"{pdf_path.stem}__page-{idx}.png"
            img_path = out_dir / img_name

            if not img_path.exists():
                pix = page.get_pixmap(matrix=mat)
                pix.save(str(img_path))

            rect = page.rect  # has x0, y0, x1, y1 floats
            results.append(
                {
                    "page_no": idx,
                    "image_path": str(img_path),
                    "bbox": [
                        float(rect.x0),
                        float(rect.y0),
                        float(rect.x1),
                        float(rect.y1),
                    ],
                }
            )

    return results


__all__ = ["pdf_pages_to_image_documents", "save_pdf_page_images"]
