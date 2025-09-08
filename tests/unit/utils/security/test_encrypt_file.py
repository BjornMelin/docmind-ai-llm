"""Tests for file encryption and decryption utilities."""

import tempfile

import pytest

from src.utils.security import decrypt_file, encrypt_file


def test_encrypt_file_passthrough_without_key(monkeypatch):
    """Test that encrypt_file returns original path when no encryption key is set.

    Verifies passthrough behavior when DOCMIND_IMG_AES_KEY_BASE64 environment
    variable is not set, ensuring files remain unencrypted.

    Args:
        monkeypatch: Pytest fixture for manipulating environment variables.
    """
    monkeypatch.delenv("DOCMIND_IMG_AES_KEY_BASE64", raising=False)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(b"abc")
        path = f.name
    out = encrypt_file(path)
    assert out == path


@pytest.mark.parametrize("keylen", [16, 24, 32])
def test_encrypt_file_with_key_round_trip(monkeypatch, keylen):
    """Test encrypt/decrypt round trip with AES keys of different lengths.

    Tests AES-GCM encryption and decryption using different key sizes
    (128, 192, 256 bits). Verifies that encrypted files can be successfully
    decrypted back to original content.

    Args:
        monkeypatch: Pytest fixture for manipulating environment variables.
        keylen: AES key length in bytes (16, 24, or 32).
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
    except Exception:
        pytest.skip("cryptography not available")

    import base64
    import os as _os

    key = _os.urandom(keylen)
    monkeypatch.setenv(
        "DOCMIND_IMG_AES_KEY_BASE64", base64.b64encode(key).decode("ascii")
    )

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(b"hello world")
        path = f.name
    enc = encrypt_file(path)
    assert enc.endswith(".enc")
    dec = decrypt_file(enc)
    with open(dec, "rb") as g:
        assert g.read() == b"hello world"


def test_encrypt_file_with_kid_aad(monkeypatch, tmp_path):
    """Test that KID is used as AAD and influences decrypt.

    With correct KID, decrypt succeeds; with wrong KID, decrypt_file returns
    the original .enc path (graceful failure).
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
    except Exception:
        pytest.skip("cryptography not available")

    import base64

    key = b"x" * 32
    monkeypatch.setenv("DOCMIND_IMG_AES_KEY_BASE64", base64.b64encode(key).decode())
    monkeypatch.setenv("DOCMIND_IMG_KID", "kid-123")

    p = tmp_path / "x.bin"
    p.write_bytes(b"abc")
    enc = encrypt_file(str(p))
    assert enc.endswith(".enc")

    # Decrypt with correct kid
    dec1 = decrypt_file(enc)
    with open(dec1, "rb") as f:
        assert f.read() == b"abc"

    # Change KID to simulate AAD mismatch
    monkeypatch.setenv("DOCMIND_IMG_KID", "other")
    dec2 = decrypt_file(enc)
    assert dec2 == enc


def test_encrypt_file_delete_plaintext(monkeypatch, tmp_path):
    """When DOCMIND_IMG_DELETE_PLAINTEXT=1, original file is removed on success."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: F401
    except Exception:
        pytest.skip("cryptography not available")

    import base64
    import os as _os

    key = _os.urandom(32)
    monkeypatch.setenv(
        "DOCMIND_IMG_AES_KEY_BASE64", base64.b64encode(key).decode("ascii")
    )
    monkeypatch.setenv("DOCMIND_IMG_DELETE_PLAINTEXT", "1")

    p = tmp_path / "img.webp"
    p.write_bytes(b"imgdata")
    enc = encrypt_file(str(p))
    assert enc.endswith(".enc")
    # Plaintext should be gone
    assert not p.exists()
