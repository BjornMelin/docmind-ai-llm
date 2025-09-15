"""Unit tests for processing.utils helpers."""

from __future__ import annotations

import importlib


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
