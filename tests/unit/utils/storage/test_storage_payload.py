"""Tests for persist_image_metadata helper."""

from __future__ import annotations

import importlib


def test_persist_image_metadata_success():
    mod = importlib.import_module("src.utils.storage")

    class _C:
        def __init__(self):
            self.updated = []

        def set_payload(self, collection_name, points, payload):
            self.updated.append((collection_name, points, payload))

    c = _C()
    assert mod.persist_image_metadata(c, "col", 1, {"a": 1}) is True
    assert c.updated
    assert c.updated[0][0] == "col"


def test_persist_image_metadata_failure():
    mod = importlib.import_module("src.utils.storage")

    class _C:
        def set_payload(self, *_a, **_k):
            raise OSError("fail")

    assert mod.persist_image_metadata(_C(), "col", 1, {"a": 1}) is False
