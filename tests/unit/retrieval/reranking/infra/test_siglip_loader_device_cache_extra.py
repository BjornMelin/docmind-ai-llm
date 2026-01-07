"""Additional tests for SigLIP loader device selection and cache reuse."""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def test_siglip_loader_device_and_cache(monkeypatch):
    rr = importlib.import_module("src.retrieval.reranking")
    vision = importlib.import_module("src.utils.vision_siglip")
    core = importlib.import_module("src.utils.core")

    # Fake transformers with stable class identities
    class _M:
        @staticmethod
        def from_pretrained(_id):
            return SimpleNamespace(to=lambda _d: SimpleNamespace())

    class _P:
        @staticmethod
        def from_pretrained(_id):
            return SimpleNamespace()

    fake_tf = SimpleNamespace(SiglipModel=_M, SiglipProcessor=_P)
    monkeypatch.setitem(importlib.sys.modules, "transformers", fake_tf)
    # Clear cache to avoid interference
    vision._cached.cache_clear()  # pylint: disable=protected-access

    # Patch torch cuda availability via core.TORCH (used by select_device())
    class _Cuda:
        @staticmethod
        def is_available():
            return False

    monkeypatch.setattr(core, "TORCH", SimpleNamespace(cuda=_Cuda), raising=False)
    # First load: CPU
    m1, p1, dev1 = rr._load_siglip()  # pylint: disable=protected-access
    assert dev1 == "cpu"

    # Make CUDA available
    class _Cuda2:
        @staticmethod
        def is_available():
            return True

    monkeypatch.setattr(core, "TORCH", SimpleNamespace(cuda=_Cuda2), raising=False)

    # Edge-case: verify cache key includes device when availability flips.
    m2, p2, dev2 = rr._load_siglip()  # pylint: disable=protected-access
    assert m1
    assert p1
    assert m2
    assert p2
    assert dev2 == "cuda"
    assert vision._cached.cache_info().currsize == 2  # pylint: disable=protected-access
