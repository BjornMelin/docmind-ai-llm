"""Test resource cleanup in SigLIP reranking path.

Ensures that image resources are closed even on error/exception paths.
"""

from __future__ import annotations

import io
import importlib
from pathlib import Path

from PIL import Image


def test_siglip_cleanup_on_error(tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    rmod = importlib.import_module("src.retrieval.reranking")

    # Create a small PNG image on disk
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    img_path = tmp_path / "tiny.png"
    img.save(img_path)

    # Nodes with image metadata
    class _Node:  # minimal shim for NodeWithScore.node
        def __init__(self, path: Path):
            self.node_id = str(path)
            self.metadata = {"path": str(path)}

    class _NWS:  # NodeWithScore shim
        def __init__(self, p: Path):
            self.node = _Node(p)
            self.score = 0.0

    nodes = [_NWS(img_path)]

    # Force model loader to fail to trigger exception path after images are opened
    monkeypatch.setattr(rmod, "_load_siglip", lambda: (_ for _ in ()).throw(RuntimeError("fail")))

    # Track calls to Image.Image.close
    close_count = {"n": 0}
    orig_close = Image.Image.close

    def _counting_close(self):  # type: ignore[no-untyped-def]
        close_count["n"] += 1
        return orig_close(self)

    monkeypatch.setattr(Image.Image, "close", _counting_close, raising=False)

    out = rmod._siglip_rescore("q", nodes, budget_ms=50)
    assert isinstance(out, list)
    # We expect at least one close call (converted image closed in finally)
    assert close_count["n"] >= 1

