"""Unit tests for PDF page image emission utilities.

Offline and deterministic: PyMuPDF is patched with a lightweight stub.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.unit
def test_save_pdf_page_images_stable_names_and_bbox(tmp_path: Path) -> None:
    """save_pdf_page_images writes stable filenames and returns bbox metadata."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("dummy")

    # Build a tiny fitz stub: document with one page
    class _Page:
        def __init__(self) -> None:
            self.rect = SimpleNamespace(x0=0.0, y0=0.0, x1=612.0, y1=792.0)

        def get_pixmap(self, matrix=None):
            class _Pix:
                def save(self, path: str) -> None:
                    with open(path, "wb") as f:
                        f.write(b"PNG")

            return _Pix()

    class _Doc:
        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            return None

        def __iter__(self):
            return iter([_Page()])

    with patch("src.processing.pdf_pages.fitz.open", return_value=_Doc()):
        from src.processing.pdf_pages import save_pdf_page_images

        out_dir = tmp_path / "images"
        items = save_pdf_page_images(pdf_path, out_dir, dpi=144)

        assert len(items) == 1
        it = items[0]
        # Stable name format
        assert Path(it["image_path"]).name == f"{pdf_path.stem}__page-1.png"
        # BBox present
        assert it["bbox"] == [0.0, 0.0, 612.0, 792.0]
