"""Tests for encrypted image helper."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

from src.utils.images import open_image_encrypted


def _install_pil(monkeypatch: pytest.MonkeyPatch, opener):
    pil_pkg = ModuleType("PIL")
    image_mod = SimpleNamespace(open=opener)
    pil_pkg.Image = image_mod
    monkeypatch.setitem(sys.modules, "PIL", pil_pkg)
    monkeypatch.setitem(sys.modules, "PIL.Image", image_mod)


def test_open_image_encrypted_plaintext(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = {}

    @contextmanager
    def fake_open(path: str):
        opened["path"] = path
        yield SimpleNamespace(path=path)

    _install_pil(monkeypatch, fake_open)

    with open_image_encrypted("sample.png") as handle:
        assert handle.path == "sample.png"
    assert opened["path"] == "sample.png"


def test_open_image_encrypted_with_decrypt(monkeypatch: pytest.MonkeyPatch) -> None:
    removed = []

    @contextmanager
    def fake_open(path: str):
        yield SimpleNamespace(path=path)

    _install_pil(monkeypatch, fake_open)

    sec_mod = ModuleType("src.utils.security")
    sec_mod.decrypt_file = lambda path: "decrypted.png"
    monkeypatch.setitem(sys.modules, "src.utils.security", sec_mod)
    monkeypatch.setattr("src.utils.images.os.remove", lambda path: removed.append(path))

    with open_image_encrypted("sample.png.enc") as handle:
        assert handle.path == "decrypted.png"
    assert removed == ["decrypted.png"]


def test_open_image_encrypted_missing_decryptor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def fake_open(path: str):
        yield SimpleNamespace(path=path)

    _install_pil(monkeypatch, fake_open)
    monkeypatch.setitem(
        sys.modules,
        "src.utils.security",
        ModuleType("src.utils.security"),
    )

    with open_image_encrypted("sample.png.enc") as handle:
        assert handle.path == "sample.png.enc"
