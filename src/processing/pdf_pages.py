"""PDF page image emission utilities.

Generates page-image artifacts for PDFs with stable filenames and bounding
boxes.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image  # type: ignore
from PIL.Image import Resampling  # type: ignore

from src.config import settings
from src.utils.security import encrypt_file, get_image_kid

_PAGE_TEXT_MAX_CHARS = 8000


@dataclass(frozen=True, slots=True)
class PageRect:
    """PDF page bounding box in PDF canvas units."""

    x0: float
    y0: float
    x1: float
    y1: float


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
    pix: Any, target_stem: Path, *, encrypt: bool | None = None
) -> tuple[Path, str]:
    """Persist a rendered page as WebP (preferred) or JPEG fallback.

    Encryption is applied when ``encrypt`` is true (defaulting to
    ``settings.processing.encrypt_page_images``), yielding ``*.enc`` outputs.

    Args:
        pix: Rendered page bitmap. pypdfium2 bitmaps and legacy pixmap-like
            test doubles are supported.
        target_stem: Path stem used to derive the output filename.
        encrypt: Whether to encrypt the rendered output; ``None`` uses settings.

    Returns:
        tuple[Path, str]: Output path (possibly encrypted) and perceptual hash.
    """
    if encrypt is None:
        encrypt = getattr(settings.processing, "encrypt_page_images", False)

    img = _rendered_to_pil(pix)
    # Strip EXIF/metadata explicitly
    if hasattr(img, "info"):
        img.info.pop("exif", None)
    if img.mode == "RGBA":
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
) -> list[tuple[int, Path, PageRect, str, str]]:
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
        list[tuple[int, Path, PageRect, str, str]]: One entry per page containing
        the 1-based page number, output image path, page rectangle, phash, and
        extracted page text (best-effort).
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("pypdfium2 is required for PDF page rendering") from exc

    results: list[tuple[int, Path, PageRect, str, str]] = []
    pdf_mtime = pdf_path.stat().st_mtime if pdf_path.exists() else 0.0

    with pdfium.PdfDocument(pdf_path) as doc:
        if len(doc) > settings.parsing.max_pages:
            raise ValueError("PDF exceeds configured page limit")
        scale = dpi / 72.0
        for idx in range(len(doc)):
            page = doc[idx]
            try:
                results.append(
                    _render_or_reuse_page(
                        page=page,
                        page_num=idx + 1,
                        pdf_path=pdf_path,
                        out_dir=out_dir,
                        pdf_mtime=pdf_mtime,
                        scale=scale,
                        encrypt=encrypt,
                    )
                )
            finally:
                page.close()

    return results


def _render_or_reuse_page(
    *,
    page: Any,
    page_num: int,
    pdf_path: Path,
    out_dir: Path,
    pdf_mtime: float,
    scale: float,
    encrypt: bool | None,
) -> tuple[int, Path, PageRect, str, str]:
    page_text = _extract_page_text(page)
    rect = _page_rect(page)
    render_pixels = int((rect.x1 - rect.x0) * scale) * int((rect.y1 - rect.y0) * scale)
    if render_pixels > settings.parsing.max_render_pixels:
        raise ValueError("PDF page exceeds configured render pixel limit")
    img_stem = f"{pdf_path.stem}__page-{page_num}"
    base = out_dir / img_stem
    existing = next((p for p in _page_image_candidates(base) if p.exists()), None)
    img_path = existing or base.with_suffix(".webp")

    if _needs_render(img_path, existing=existing, pdf_mtime=pdf_mtime, encrypt=encrypt):
        bitmap = page.render(scale=scale)
        try:
            img_path, phash = _save_with_format(
                bitmap, out_dir / img_stem, encrypt=encrypt
            )
        finally:
            bitmap.close()
        with contextlib.suppress(OSError):
            os.utime(img_path, (pdf_mtime, pdf_mtime))
        return page_num, img_path, rect, phash, page_text

    return page_num, img_path, rect, _phash_existing(img_path), page_text


def _page_image_candidates(base: Path) -> list[Path]:
    return [
        base.with_suffix(".webp.enc"),
        base.with_suffix(".webp"),
        base.with_suffix(".jpg.enc"),
        base.with_suffix(".jpg"),
        base.with_suffix(".jpeg.enc"),
        base.with_suffix(".jpeg"),
    ]


def _needs_render(
    img_path: Path,
    *,
    existing: Path | None,
    pdf_mtime: float,
    encrypt: bool | None,
) -> bool:
    if existing is None:
        return True
    wants_encrypt = (
        getattr(settings.processing, "encrypt_page_images", False)
        if encrypt is None
        else bool(encrypt)
    )
    try:
        return (img_path.stat().st_mtime < pdf_mtime) or (
            (img_path.suffix == ".enc") != wants_encrypt
        )
    except OSError:
        return True


def _phash_existing(img_path: Path) -> str:
    try:
        from src.utils.images import open_image_encrypted

        with open_image_encrypted(str(img_path)) as im:
            return _phash(im) if im is not None else ""
    except (OSError, ValueError, RuntimeError):
        return ""


def _rendered_to_pil(rendered: Any) -> Image.Image:
    """Convert a pypdfium2 bitmap or pixmap-like object to a PIL image."""
    to_pil = getattr(rendered, "to_pil", None)
    if callable(to_pil):
        img = to_pil()
        if isinstance(img, Image.Image):
            return img
    mode = "RGB" if int(getattr(rendered, "n", 3)) < 4 else "RGBA"
    return Image.frombytes(
        mode,
        (int(rendered.width), int(rendered.height)),
        rendered.samples,
    )


def _extract_page_text(page: Any) -> str:
    text = ""
    try:
        textpage = page.get_textpage()
        try:
            text = str(textpage.get_text_range() or "").strip()
        finally:
            textpage.close()
    except Exception:  # pragma: no cover - PDFium quirks
        text = ""
    if len(text) > _PAGE_TEXT_MAX_CHARS:
        return text[:_PAGE_TEXT_MAX_CHARS]
    return text


def _page_rect(page: Any) -> PageRect:
    try:
        bbox = page.get_bbox()
        return PageRect(
            x0=float(bbox[0]),
            y0=float(bbox[1]),
            x1=float(bbox[2]),
            y1=float(bbox[3]),
        )
    except (AttributeError, TypeError, ValueError, IndexError):
        width, height = page.get_size()
        return PageRect(x0=0.0, y0=0.0, x1=float(width), y1=float(height))


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
                if path.suffix == ".enc"
                else {}
            ),
        }
        for idx, path, rect, phash, page_text in entries
    ]


__all__ = ["save_pdf_page_images"]
