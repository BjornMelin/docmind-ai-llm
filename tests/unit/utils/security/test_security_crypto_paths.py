"""Security crypto roundtrip tests (skipped if cryptography missing)."""

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("cryptography") is None,
    reason="cryptography not available",
)
def test_encrypt_decrypt_roundtrip(tmp_path, monkeypatch):
    from src.config.settings import settings
    from src.utils.security import decrypt_file, encrypt_file

    # Setup key (32 bytes base64) and kid
    key = os.urandom(32)
    import base64

    monkeypatch.setattr(
        settings.image_encryption, "aes_key_base64", base64.b64encode(key).decode()
    )
    monkeypatch.setattr(settings.image_encryption, "kid", "kid-1")

    p = Path(tmp_path) / "x.bin"
    src = os.urandom(128)
    p.write_bytes(src)

    enc = encrypt_file(str(p))
    assert enc.endswith(".enc")

    dec = decrypt_file(enc)
    assert Path(dec).read_bytes() == src
