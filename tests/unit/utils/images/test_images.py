"""Tests for image helpers."""

from __future__ import annotations

import shutil
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, UnidentifiedImageError

from src.utils.images import (
    IMAGE_READ_CHUNK_BYTES,
    MAX_IMAGE_BYTES,
    MAX_UNTRUSTED_IMAGE_PIXELS,
    open_image_encrypted,
    open_untrusted_image,
)


def _write_sample_png(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (8, 8), color=color)
    image.save(path)


def test_open_image_plaintext(tmp_path: Path) -> None:
    """Plain images should open without altering the file."""
    image_path = tmp_path / "sample.png"
    _write_sample_png(image_path, (255, 0, 0))

    with open_image_encrypted(str(image_path)) as handle:
        assert handle.size == (8, 8)


def test_open_image_encrypted_invokes_decryptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Encrypted images should be decrypted and temporary files removed."""
    source_image = tmp_path / "plain.png"
    _write_sample_png(source_image, (0, 255, 0))

    encrypted = tmp_path / "cipher.png.enc"
    encrypted.write_bytes(b"placeholder")

    decrypted = tmp_path / "cipher.png"

    def fake_decrypt(path: str) -> str:
        assert path == str(encrypted)
        shutil.copy2(source_image, decrypted)
        return str(decrypted)

    monkeypatch.setattr("src.utils.security.decrypt_file", fake_decrypt)

    with open_image_encrypted(str(encrypted)) as handle:
        assert handle.size == (8, 8)

    assert not decrypted.exists()


@pytest.mark.unit
def test_open_untrusted_image_verifies_uploaded_bytes() -> None:
    """Uploaded images should be verified and detached."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(buffer, format="PNG")
    buffer.seek(0)
    previous_limit = Image.MAX_IMAGE_PIXELS

    image = open_untrusted_image(buffer)

    assert image.size == (8, 8)
    assert image.mode == "RGB"
    assert previous_limit == Image.MAX_IMAGE_PIXELS


@pytest.mark.unit
def test_open_untrusted_image_restores_pillow_pixel_limit() -> None:
    """Untrusted validation should not leak Pillow pixel limits globally."""
    buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(0, 0, 255)).save(buffer, format="PNG")
    buffer.seek(0)
    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = MAX_UNTRUSTED_IMAGE_PIXELS * 2
    try:
        open_untrusted_image(buffer)

        assert Image.MAX_IMAGE_PIXELS == MAX_UNTRUSTED_IMAGE_PIXELS * 2
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit


@pytest.mark.unit
def test_open_untrusted_image_rejects_malformed_upload() -> None:
    """Malformed upload bytes should be rejected by Pillow verification."""
    with pytest.raises(UnidentifiedImageError):
        open_untrusted_image(BytesIO(b"not an image"))


@pytest.mark.unit
def test_open_untrusted_image_rejects_oversized_upload() -> None:
    """Oversized upload bytes should be rejected before Pillow opens them."""

    class StreamingOversizedUpload:
        """Stream bytes beyond the configured image upload limit.

        Attributes:
            _position: Current read cursor position.
            _remaining: Number of unread bytes left in the stream.
            _chunk: Reusable byte chunk returned by read.
        """

        def __init__(self) -> None:
            """Initialize StreamingOversizedUpload.

            Args:
                None.

            Returns:
                None.
            """
            self._position = 0
            self._remaining = MAX_IMAGE_BYTES + 1
            self._chunk = b"0" * IMAGE_READ_CHUNK_BYTES

        def read(self, size: int = -1) -> bytes:
            """Read bytes from the oversized stream.

            Args:
                size: Maximum number of bytes to read, or -1 for a full chunk.

            Returns:
                The next byte chunk, or empty bytes at EOF.
            """
            if self._remaining <= 0:
                return b""
            limit = (
                len(self._chunk) if size < 0 else min(size, len(self._chunk))
            )
            chunk_size = min(limit, self._remaining)
            self._remaining -= chunk_size
            self._position += chunk_size
            return self._chunk[:chunk_size]

        def seek(self, offset: int, whence: int = 0) -> int:
            """Seek within the oversized stream.

            Args:
                offset: Target stream offset.
                whence: Offset reference point.

            Returns:
                The new stream position.

            Raises:
                OSError: Raised when the seek is not a rewind to the start.
            """
            if offset == 0 and whence == 0:
                self._position = 0
                self._remaining = MAX_IMAGE_BYTES + 1
                return 0
            raise OSError("seek only supports rewinding to the start")

        def tell(self) -> int:
            """Return the current stream position.

            Args:
                None.

            Returns:
                The current byte offset.
            """
            return self._position

    upload = StreamingOversizedUpload()

    with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
        open_untrusted_image(upload)

    assert upload.tell() == 0


@pytest.mark.unit
def test_open_untrusted_image_ignores_non_callable_seek() -> None:
    """Uploads with a non-callable seek attribute should still open safely."""

    class UploadWithBadSeek:
        """Expose a valid image stream with a non-callable seek attribute.

        Attributes:
            _buffer: In-memory PNG bytes used by read.
            seek: Non-callable attribute that shadows normal seek behavior.
        """

        def __init__(self) -> None:
            """Initialize UploadWithBadSeek.

            Args:
                None.

            Returns:
                None.
            """
            self._buffer = BytesIO()
            Image.new("RGB", (8, 8), color=(255, 255, 0)).save(
                self._buffer,
                format="PNG",
            )
            self._buffer.seek(0)
            self.seek = 42

        def read(self, size: int = -1) -> bytes:
            """Read bytes from the backing image buffer.

            Args:
                size: Maximum number of bytes to read, or -1 for all bytes.

            Returns:
                Bytes read from the image buffer.
            """
            return self._buffer.read(size)

    image = open_untrusted_image(UploadWithBadSeek())

    assert image.size == (8, 8)
    assert image.mode == "RGB"
