"""Unit tests for src.retrieval.adapter_registry registry behaviors."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import adapter_registry as reg

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _isolate_registry() -> None:
    reg._REGISTRY.clear()
    yield
    reg._REGISTRY.clear()


def test_get_adapter_named_missing_raises() -> None:
    with pytest.raises(reg.MissingGraphAdapterError):
        reg.get_adapter("missing")


def test_get_adapter_returns_first_registered_when_no_default() -> None:
    a = SimpleNamespace(name="other", supports_graphrag=True)
    reg.register_adapter(a)

    resolved = reg.get_adapter()
    assert resolved is a


def test_list_adapters_is_sorted() -> None:
    reg.register_adapter(SimpleNamespace(name="b"))
    reg.register_adapter(SimpleNamespace(name="a"))
    assert reg.list_adapters() == ["a", "b"]


def test_unregister_adapter_is_idempotent() -> None:
    reg.register_adapter(SimpleNamespace(name="x"))
    reg.unregister_adapter("x")
    reg.unregister_adapter("x")
    assert reg.list_adapters() == []


def test_ensure_default_adapter_skips_when_llama_index_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise():  # pragma: no cover - simple stub
        raise reg.MissingLlamaIndexError("nope")

    monkeypatch.setattr(reg, "build_llama_index_factory", _raise, raising=True)

    reg.ensure_default_adapter()
    assert "llama_index" not in reg._REGISTRY


def test_get_default_adapter_health_returns_guidance_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise():  # pragma: no cover - simple stub
        raise reg.MissingLlamaIndexError("nope")

    monkeypatch.setattr(reg, "build_llama_index_factory", _raise, raising=True)

    supported, name, hint = reg.get_default_adapter_health()
    assert supported is False
    assert name == "unavailable"
    assert hint == reg.GRAPH_DEPENDENCY_HINT
