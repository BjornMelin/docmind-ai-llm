from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def test_pdf_pages_to_image_documents_builds_metadata_and_artifact_refs(
    monkeypatch, tmp_path: Path
) -> None:
    from PIL import Image

    from src.processing import pdf_pages as pp

    entries = [
        (1, tmp_path / "p1.webp", SimpleNamespace(), "ph1", "t1"),
        (2, tmp_path / "p2.webp.enc", SimpleNamespace(), "ph2", "t2"),
    ]
    monkeypatch.setattr(pp, "_render_pdf_pages", lambda *_a, **_k: list(entries))

    refs = [
        SimpleNamespace(sha256="a", suffix=".png"),
        SimpleNamespace(sha256="b", suffix=".png"),
    ]

    class _Store:
        def __init__(self) -> None:
            self.put_calls: list[Path] = []

        def put_file(self, path: Path):  # type: ignore[no-untyped-def]
            self.put_calls.append(Path(path))
            return refs[len(self.put_calls) - 1]

        def resolve_path(self, ref):  # type: ignore[no-untyped-def]
            return tmp_path / f"{ref.sha256}{ref.suffix}"

    monkeypatch.setattr(pp.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(pp, "get_image_kid", lambda: "kid1")

    # Create real image files at the resolved artifact paths because
    # llama_index.core.schema.ImageDocument validates accessibility.
    Image.new("RGB", (2, 2), color="white").save(tmp_path / "a.png")
    Image.new("RGB", (2, 2), color="black").save(tmp_path / "b.png")

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    docs, out_dir = pp.pdf_pages_to_image_documents(
        pdf_path=pdf,
        output_dir=tmp_path / "out",
        encrypt=True,
    )
    assert out_dir.exists()
    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1
    assert docs[0].metadata["phash"] == "ph1"
    assert docs[0].metadata["source_filename"] == "doc.pdf"
    assert docs[0].metadata["image_artifact_id"] == "a"

    assert docs[1].metadata["page"] == 2
    assert docs[1].metadata["encrypted"] is True
    assert docs[1].metadata["kid"] == "kid1"
    assert docs[1].metadata["image_artifact_id"] == "b"


def test_save_pdf_page_images_returns_bbox_and_encryption_metadata(
    monkeypatch, tmp_path: Path
) -> None:
    from src.processing import pdf_pages as pp

    rect = SimpleNamespace(x0=1, y0=2, x1=3, y1=4)
    entries = [
        (1, tmp_path / "p1.webp", rect, "ph1", ""),
        (2, tmp_path / "p2.webp.enc", rect, "ph2", ""),
    ]
    monkeypatch.setattr(pp, "_render_pdf_pages", lambda *_a, **_k: list(entries))
    monkeypatch.setattr(pp, "get_image_kid", lambda: "kid1")

    items = pp.save_pdf_page_images(tmp_path / "doc.pdf", out_dir=tmp_path / "out")
    assert items[0]["page_no"] == 1
    assert items[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert "encrypted" not in items[0]
    assert items[1]["encrypted"] is True
    assert items[1]["kid"] == "kid1"


def test_pdf_pages_to_image_documents_uses_default_cache_dir(
    monkeypatch, tmp_path: Path
) -> None:
    from src.processing import pdf_pages as pp

    monkeypatch.setattr(pp, "_render_pdf_pages", lambda *_a, **_k: [])
    monkeypatch.setattr(pp.ArtifactStore, "from_settings", lambda _s: object())
    monkeypatch.setattr(
        pp,
        "settings",
        SimpleNamespace(
            cache_dir=tmp_path, processing=SimpleNamespace(encrypt_page_images=False)
        ),
    )

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    docs, out_dir = pp.pdf_pages_to_image_documents(pdf_path=pdf, output_dir=None)
    assert docs == []
    assert out_dir == tmp_path / "page_images" / "doc"


def test_pdf_pages_to_image_documents_records_put_file_failures(
    monkeypatch, tmp_path: Path
) -> None:
    from src.processing import pdf_pages as pp

    img = tmp_path / "p1.webp"
    img.write_bytes(b"x")
    monkeypatch.setattr(
        pp,
        "_render_pdf_pages",
        lambda *_a, **_k: [(1, img, SimpleNamespace(), "ph", "")],
    )

    class _Store:
        def put_file(self, _path: Path):  # type: ignore[no-untyped-def]
            raise OSError("boom")

    warnings: list[tuple] = []
    monkeypatch.setattr(pp.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(pp.logger, "warning", lambda *a, **k: warnings.append((a, k)))

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    docs, _out_dir = pp.pdf_pages_to_image_documents(
        pdf_path=pdf, output_dir=tmp_path / "out"
    )
    assert docs == []
    assert warnings


def test_pdf_pages_to_image_documents_records_resolve_path_failures(
    monkeypatch, tmp_path: Path
) -> None:
    from src.processing import pdf_pages as pp

    img = tmp_path / "p1.webp"
    img.write_bytes(b"x")
    monkeypatch.setattr(
        pp,
        "_render_pdf_pages",
        lambda *_a, **_k: [(1, img, SimpleNamespace(), "ph", "")],
    )

    class _Store:
        def put_file(self, _path: Path):  # type: ignore[no-untyped-def]
            return SimpleNamespace(sha256="a", suffix=".png")

        def resolve_path(self, _ref):  # type: ignore[no-untyped-def]
            raise ValueError("bad")

    warnings: list[tuple] = []
    monkeypatch.setattr(pp.ArtifactStore, "from_settings", lambda _s: _Store())
    monkeypatch.setattr(pp.logger, "warning", lambda *a, **k: warnings.append((a, k)))

    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%EOF\n")
    docs, _out_dir = pp.pdf_pages_to_image_documents(
        pdf_path=pdf, output_dir=tmp_path / "out"
    )
    assert docs == []
    assert warnings
