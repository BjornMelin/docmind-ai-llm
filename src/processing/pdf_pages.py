"""PDF page image emission utilities (ADR-009).

Generates page-image documents for PDFs and tags modality metadata to enable
visual reranking via ColPali. Uses PyMuPDF (pymupdf) for fast rendering.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from llama_index.core.schema import ImageDocument


def pdf_pages_to_image_documents(pdf_path: Path, dpi: int = 180) -> list[ImageDocument]:
    """Render PDF pages to images and return ImageDocument nodes.

    Args:
        pdf_path: Path to the PDF file
        dpi: Render resolution (dots per inch)

    Returns:
        List of ImageDocument with modality metadata for visual reranking
    """
    pdf_path = Path(pdf_path)
    docs: list[ImageDocument] = []

    with fitz.open(pdf_path) as doc:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=mat)
            # Save to a temporary PNG in the same directory for indexing
            out_path = pdf_path.with_suffix("")
            img_path = out_path.parent / f"{out_path.name}_page_{i}.png"
            pix.save(str(img_path))

            meta: dict[str, Any] = {
                "page": i,
                "modality": "pdf_page_image",
                "source": str(pdf_path),
            }
            docs.append(ImageDocument(image_path=str(img_path), metadata=meta))

    return docs
