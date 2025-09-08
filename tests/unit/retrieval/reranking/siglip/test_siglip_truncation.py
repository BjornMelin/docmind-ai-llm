"""SigLIP truncation path: long text gets truncated even when images fail.

We avoid heavy deps by:
- Patching PIL.Image.open to raise (images become None)
- Patching transformers+torch via _load_siglip and sys.modules to light stubs
"""

from __future__ import annotations

import types

from llama_index.core.schema import NodeWithScore, TextNode

from src.retrieval import reranking as rr


def test_siglip_truncates_long_text(monkeypatch):
    # Ensure small limit for brevity
    monkeypatch.setattr(rr, "TEXT_TRUNCATION_LIMIT", 8, raising=False)

    # Create a visual-text node with long text
    txt = "X" * 32
    node = TextNode(text=txt)
    nws = NodeWithScore(node=node, score=0.0)
    nws.node.metadata["modality"] = "pdf_page_image"
    nws.node.metadata["image_path"] = "/nope.webp"
    nodes = [nws]

    # Patch PIL.Image.open to raise so images list contains None
    from PIL import Image  # type: ignore

    def _raise_open(_path):
        raise OSError("no image")

    monkeypatch.setattr(Image, "open", staticmethod(_raise_open))

    # Patch torch as a light module with no_grad context manager
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    import sys

    fake_torch = types.SimpleNamespace(no_grad=lambda: _NoGrad())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    # Patch _load_siglip to return minimal stubs
    class _Model:
        def get_text_features(self, **_):
            class _T:
                def __truediv__(self, _other):
                    return self

                def norm(self, *_, **__):
                    return 1.0

            return _T()

    class _Proc:
        def __call__(self, *a, **k):
            return {}

    monkeypatch.setattr(rr, "_load_siglip", lambda: (_Model(), _Proc(), "cpu"))

    out = rr._siglip_rescore("q", nodes, budget_ms=9999)
    assert len(out) == 1
    assert len(out[0].node.text) == 8
