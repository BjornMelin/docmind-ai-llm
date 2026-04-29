"""Image I/O helpers (encrypted-safe open).

Provides a context manager to open images securely, handling optional
`*.enc` files using the local decryptor and ensuring temporary files are
cleaned up regardless of success or failure.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
from collections.abc import Callable
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

MAX_UNTRUSTED_IMAGE_PIXELS = 50_000_000
MAX_IMAGE_BYTES = 25 * 1024 * 1024
IMAGE_READ_CHUNK_BYTES = 1024 * 1024
_PIL_MAX_PIXELS_LOCK = threading.Lock()


def _configure_pillow_limits(image_module: Any) -> None:
    """Apply deliberate Pillow limits for untrusted image inputs."""
    current_limit = image_module.MAX_IMAGE_PIXELS
    if current_limit is None or current_limit > MAX_UNTRUSTED_IMAGE_PIXELS:
        image_module.MAX_IMAGE_PIXELS = MAX_UNTRUSTED_IMAGE_PIXELS


def _buffer_upload(upload: Any) -> BytesIO:
    """Read an upload into a bounded buffer.

    Args:
        upload: Binary file-like object.

    Returns:
        BytesIO: In-memory copy of the uploaded bytes.

    Raises:
        ValueError: If the upload exceeds `MAX_IMAGE_BYTES`.
        TypeError: If `upload` does not expose a binary `read` method.
    """
    read = getattr(upload, "read", None)
    if not callable(read):
        raise TypeError("Upload object must expose a binary read() method")
    read_bytes = cast(Callable[[int], bytes], read)
    if hasattr(upload, "seek"):
        upload.seek(0)

    total = 0
    with tempfile.SpooledTemporaryFile(max_size=IMAGE_READ_CHUNK_BYTES) as buffer:
        while True:
            chunk = read_bytes(IMAGE_READ_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_IMAGE_BYTES:
                raise ValueError("Uploaded image exceeds the maximum allowed size")
            buffer.write(chunk)
        buffer.seek(0)
        return BytesIO(buffer.read())


def open_untrusted_image(upload: Any) -> PILImage:
    """Open and verify an untrusted uploaded image.

    Args:
        upload: Streamlit uploaded file or another binary file-like object.

    Returns:
        PIL.Image.Image: Loaded image detached from the upload stream.

    Raises:
        ImportError: If Pillow is not installed.
        PIL.UnidentifiedImageError: If Pillow cannot identify the image.
        PIL.Image.DecompressionBombError: If the image exceeds configured
            pixel limits.
        ValueError: If the upload exceeds `MAX_IMAGE_BYTES`.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("Pillow is required for image operations") from exc

    with _PIL_MAX_PIXELS_LOCK:
        previous_limit = Image.MAX_IMAGE_PIXELS
        _configure_pillow_limits(Image)
        try:
            data = _buffer_upload(upload)
            with Image.open(data) as probe:
                probe.verify()
            data.seek(0)
            with Image.open(data) as image:
                image.load()
                return image.copy()
        finally:
            Image.MAX_IMAGE_PIXELS = previous_limit
            if hasattr(upload, "seek"):
                upload.seek(0)


@contextmanager
def open_image_encrypted(path: str):
    """Yield an image handle for plaintext or ``*.enc`` inputs.

    Args:
        path: Filesystem path to the image. ``*.enc`` inputs are decrypted via
            :func:`src.utils.security.decrypt_file` when available.

    Yields:
        PIL.Image.Image: Image opened in read-only mode.

    Raises:
        ImportError: If Pillow is not installed.
    """
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("Pillow is required for image operations") from exc

    tmp: str | None = None
    try:
        to_open = path
        if path.endswith(".enc"):
            try:
                from src.utils.security import decrypt_file  # type: ignore
            except (
                ImportError,
                AttributeError,
                RuntimeError,
            ):  # pragma: no cover - fallback
                # If decryptor is missing, treat input as plaintext for best-effort open
                def decrypt_file(pth: str) -> str:  # type: ignore
                    return pth

            dec = decrypt_file(path)
            # Avoid deleting original if decryptor returned the same path
            if dec != path:
                tmp = dec
            to_open = dec
        with Image.open(to_open) as im:
            yield im
    finally:
        if tmp:
            with contextlib.suppress(Exception):
                os.remove(tmp)


def ensure_thumbnail(
    image_path: str | Path,
    *,
    max_side: int = 384,
    thumb_dir: Path | None = None,
    encrypt: bool | None = None,
) -> Path:
    """Create (or reuse) a small WebP thumbnail for fast UI rendering.

    Thumbnails are stored alongside the original by default (or under ``thumb_dir``)
    and are encrypted when ``encrypt`` is true or the source image is encrypted.
    Plaintext deletion after encryption follows DOCMIND_IMG_DELETE_PLAINTEXT.
    """
    try:
        from PIL.Image import Resampling  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("Pillow is required for image operations") from exc

    src = Path(image_path)
    thumb_root = Path(thumb_dir) if thumb_dir is not None else src.parent
    thumb_root.mkdir(parents=True, exist_ok=True)

    # Determine underlying extension and encryption state.
    # src.stem strips .enc → "image.png", then Path().stem strips image ext → "image"
    base_stem = Path(src.stem).stem if src.suffix == ".enc" else src.stem
    is_enc = src.suffix == ".enc"

    should_encrypt = bool(encrypt) if encrypt is not None else is_enc
    thumb_plain = thumb_root / f"{base_stem}__thumb.webp"
    thumb_target = (
        thumb_plain.with_suffix(thumb_plain.suffix + ".enc")
        if should_encrypt
        else thumb_plain
    )

    # Reuse existing thumbnail when up-to-date.
    try:
        if (
            thumb_target.exists()
            and src.exists()
            and thumb_target.stat().st_mtime >= src.stat().st_mtime
        ):
            return thumb_target
    except OSError:
        # Best-effort; fall through to recreate.
        pass

    with open_image_encrypted(str(src)) as im:
        img = im.convert("RGB")
        img.thumbnail((int(max_side), int(max_side)), Resampling.LANCZOS)
        img.save(thumb_plain, format="WEBP", quality=60, method=6)

    if should_encrypt:
        try:
            from src.utils.security import encrypt_file

            # encrypt_file decides whether to delete plaintext (env-driven).
            enc_path = encrypt_file(str(thumb_plain))
            return Path(enc_path)
        except (OSError, RuntimeError, ValueError, ImportError) as exc:
            logger.warning(
                "Thumbnail encryption failed; returning plaintext: {} ({})",
                thumb_plain,
                exc,
            )
            return thumb_plain

    return thumb_plain


__all__ = ["ensure_thumbnail", "open_image_encrypted", "open_untrusted_image"]
