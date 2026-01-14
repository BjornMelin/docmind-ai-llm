import base64
import os
import tempfile

import pytest


def _skip_if_no_pymupdf():
    try:
        import fitz  # noqa: F401
    except Exception:
        pytest.skip("PyMuPDF not available")


def _make_pdf(path: str):
    import fitz  # type: ignore

    doc = fitz.open()
    page = doc.new_page(width=200, height=100)
    page.insert_text((20, 50), "Hello")
    doc.save(path)
    doc.close()


def test_pdf_page_images_encrypted(monkeypatch):
    _skip_if_no_pymupdf()
    from src.config import settings
    from src.processing.pdf_pages import save_pdf_page_images

    # Enable encryption
    key = os.urandom(32)
    monkeypatch.setattr(
        settings.image, "img_aes_key_base64", base64.b64encode(key).decode("ascii")
    )
    settings.processing.encrypt_page_images = True

    with tempfile.TemporaryDirectory() as td:
        pdf = os.path.join(td, "t.pdf")
        out = os.path.join(td, "imgs")
        _make_pdf(pdf)
        items = save_pdf_page_images(pdf, out_dir=out, dpi=200)
        assert items
        # Ensure encrypted extension
        assert items[0]["image_path"].endswith(".enc")
