
"""Tests for src.utils.vision_siglip loader caching behavior."""

from __future__ import annotations

import sys
import types

import pytest

from src.utils import vision_siglip


@pytest.mark.unit
def test_load_siglip_uses_cached_loader(monkeypatch):
    call_count = {"model": 0, "processor": 0}

    def _select(device: str) -> str:
        return device

    monkeypatch.setattr(vision_siglip, "select_device", _select)

    def _model_loader(model_id: str):
        call_count["model"] += 1
        return types.SimpleNamespace(to=lambda device: None)

    def _proc_loader(model_id: str):
        call_count["processor"] += 1
        return object()

    transformers = types.SimpleNamespace(
        SiglipModel=types.SimpleNamespace(from_pretrained=_model_loader),
        SiglipProcessor=types.SimpleNamespace(from_pretrained=_proc_loader),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model1, proc1, device1 = vision_siglip.load_siglip("siglip-test", "cpu")
    model2, proc2, device2 = vision_siglip.load_siglip("siglip-test", "cpu")

    assert call_count == {"model": 1, "processor": 1}
    assert model1 is model2 and proc1 is proc2
    assert device1 == device2 == "cpu"
