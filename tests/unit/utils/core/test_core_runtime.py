"""Runtime error-path tests for the canonical device helpers."""

from __future__ import annotations

import types

import pytest


@pytest.mark.unit
def test_resolve_device_parses_indices(monkeypatch):
    import src.utils.core as core

    dummy_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(current_device=lambda: 1)
    )
    monkeypatch.setattr(core, "TORCH", dummy_torch, raising=True)
    monkeypatch.setattr(core, "select_device", lambda prefer: "cuda")

    assert core.resolve_device("cuda:5") == ("cuda:5", 5)
    assert core.resolve_device("cuda") == ("cuda:1", 1)


@pytest.mark.unit
def test_resolve_device_falls_back_on_errors(monkeypatch):
    import src.utils.core as core

    monkeypatch.setattr(core, "TORCH", types.SimpleNamespace(cuda=None))

    def _raise(_prefer: str) -> str:  # pragma: no cover - intentional
        raise RuntimeError("select failed")

    monkeypatch.setattr(core, "select_device", _raise)
    assert core.resolve_device("auto") == ("cpu", None)


@pytest.mark.unit
def test_has_cuda_vram_returns_false_without_cuda(monkeypatch):
    import src.utils.core as core

    monkeypatch.setattr(core, "is_cuda_available", lambda: False)
    assert core.has_cuda_vram(8.0) is False


@pytest.mark.unit
def test_select_device_prefers_mps_when_available(monkeypatch):
    import src.utils.core as core

    monkeypatch.setattr(core, "is_cuda_available", lambda: False)
    monkeypatch.setattr(core, "_is_mps_available", lambda: True)
    assert core.select_device("auto") == "mps"
    assert core.select_device("mps") == "mps"


@pytest.mark.unit
def test_select_device_handles_invalid_inputs(monkeypatch):
    import src.utils.core as core

    def _broken(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(core, "is_cuda_available", _broken)
    assert core.select_device("auto") == "cpu"
