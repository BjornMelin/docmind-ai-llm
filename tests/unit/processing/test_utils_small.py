"""Unit tests for processing.utils helpers."""

from __future__ import annotations

import importlib

from src.processing.utils import _normalize_text, is_unstructured_like, sha256_id


def test_sha256_id_and_normalization():  # type: ignore[no-untyped-def]
    umod = importlib.import_module("src.processing.utils")
    # Same strings with irregular spacing normalize to the same digest
    a = umod.sha256_id(" hello  world ")
    b = umod.sha256_id("hello   world")
    assert a == b


def test_is_unstructured_like():  # type: ignore[no-untyped-def]
    umod = importlib.import_module("src.processing.utils")

    class _E:
        def __init__(self):
            self.text = "t"
            self.category = "c"
            self.metadata = type("M", (), {"a": 1})()

    assert umod.is_unstructured_like(_E()) is True
    assert umod.is_unstructured_like(object()) is False


def test_normalize_text_handles_whitespace():  # type: ignore[no-untyped-def]
    assert _normalize_text("  Hello\tworld\n") == "Hello world"
    assert _normalize_text("") == ""
    assert _normalize_text("   ") == ""


def test_sha256_id_mixes_bytes_and_strings():  # type: ignore[no-untyped-def]
    digest_a = sha256_id("prefix", b"data", "suffix")
    digest_b = sha256_id("prefix", b"data", "suffix")
    digest_c = sha256_id("prefix", "DATA", "suffix")
    assert digest_a == digest_b
    assert digest_a != digest_c


def test_is_unstructured_like_rejects_mock_metadata():  # type: ignore[no-untyped-def]
    from unittest.mock import Mock

    class _Fake:
        text = "t"
        category = "c"
        metadata = Mock()

    assert is_unstructured_like(_Fake()) is False
