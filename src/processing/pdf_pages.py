"""PDF page image emission utilities (ADR-009).

Generates page-image documents for PDFs and tags modality metadata to enable
visual reranking via ColPali. Uses PyMuPDF (pymupdf) for fast rendering.
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


__all__ = ["pdf_pages_to_image_documents"]
