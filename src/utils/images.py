"""Image I/O helpers (encrypted-safe open).

Provides a context manager to open images securely, handling optional
`*.enc` files using the local decryptor and ensuring temporary files are
cleaned up regardless of success or failure.
"""

from __future__ import annotations

import contextlib
import os
from contextlib import contextmanager


@contextmanager
def open_image_encrypted(path: str):
    """Open an image path that may be encrypted ("*.enc").

    Yields a PIL.Image.Image opened in read-only mode. Any temporary file
    created by decryption will be removed when the context exits.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dep
        raise ImportError("Pillow is required for image operations") from exc

    tmp: str | None = None
    try:
        to_open = path
        if str(path).endswith(".enc"):
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
