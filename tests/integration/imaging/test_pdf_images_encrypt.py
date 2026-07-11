import base64
import os
import tempfile
from pathlib import Path

from pydantic import SecretStr


def _make_pdf(path: Path) -> None:
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=100)
    with path.open("wb") as handle:
        writer.write(handle)


def test_pdf_page_images_encrypted(monkeypatch):
    from src.config import settings
    from src.processing.pdf_pages import save_pdf_page_images

    # Enable encryption
    key = os.urandom(32)
    monkeypatch.setattr(
        settings.image_encryption,
        "aes_key_base64",
        SecretStr(base64.b64encode(key).decode("ascii")),
    )
    settings.processing.encrypt_page_images = True

    with tempfile.TemporaryDirectory() as td:
        pdf = Path(td) / "t.pdf"
        out = Path(td) / "imgs"
        _make_pdf(pdf)
        items = save_pdf_page_images(pdf, out_dir=out, dpi=200)
        assert items
        # Ensure encrypted extension
        assert items[0]["image_path"].endswith(".enc")
