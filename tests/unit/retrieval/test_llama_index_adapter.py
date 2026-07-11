"""Unit tests for lazy LlamaIndex imports and GraphRAG health."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from unittest import mock

import pytest

from src.retrieval import llama_index_adapter as lia
from src.retrieval.llama_index_adapter import (
    MissingLlamaIndexError,
    get_graphrag_health,
    get_llama_index_adapter,
    llama_index_available,
    set_llama_index_adapter,
)


@pytest.fixture(autouse=True)
def _clear_graphrag_health_cache() -> Iterator[None]:
    lia._cached_graphrag_health.cache_clear()  # type: ignore[attr-defined]
    yield
    lia._cached_graphrag_health.cache_clear()  # type: ignore[attr-defined]


def test_llama_index_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns False when find_spec cannot locate llama_index.core."""
    monkeypatch.setattr(
        "importlib.util.find_spec",
        lambda name: None if name == "llama_index.core" else mock.DEFAULT,
    )
    assert llama_index_available() is False


def test_get_adapter_missing_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface the required-core repair hint when the dependency is absent."""
    set_llama_index_adapter(None)
    monkeypatch.setattr(
        "src.retrieval.llama_index_adapter.llama_index_available",
        lambda: False,
    )
    with pytest.raises(MissingLlamaIndexError) as exc_info:
        get_llama_index_adapter(force_reload=True)
    assert "llama-index-core is a required DocMind dependency" in str(exc_info.value)


def test_graphrag_health_reports_required_core_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lia,
        "import_module",
        lambda _name: SimpleNamespace(PropertyGraphIndex=object()),
    )

    assert get_graphrag_health(force_refresh=True) == (
        True,
        "llama_index",
        "LlamaIndex core PropertyGraphIndex is available.",
    )


def test_graphrag_health_reports_broken_required_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(_name: str) -> None:
        raise ImportError

    monkeypatch.setattr(lia, "import_module", _raise)

    supported, name, hint = get_graphrag_health(force_refresh=True)
    assert supported is False
    assert name == "unavailable"
    assert "required DocMind dependency" in hint


def test_build_real_adapter_pydantic_selector_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lia, "llama_index_available", lambda: True)

    def _fake_import_module(name: str):
        if name == "llama_index.core.query_engine":
            return SimpleNamespace(
                RouterQueryEngine=object(), RetrieverQueryEngine=object()
            )
        if name == "llama_index.core.selectors":
            return SimpleNamespace(LLMSingleSelector=object())
        if name == "llama_index.core.tools":
            return SimpleNamespace(QueryEngineTool=object(), ToolMetadata=object())
        raise ImportError(name)

    monkeypatch.setattr(lia, "import_module", _fake_import_module)
    adapter = lia._build_real_adapter()
    assert adapter.get_pydantic_selector(llm=None) is None

    class _Pydantic:
        @staticmethod
        def from_defaults(llm=None):  # type: ignore[no-untyped-def]
            return "SEL"

    def _fake_import_with_pydantic(name: str):
        if name == "llama_index.core.query_engine":
            return SimpleNamespace(
                RouterQueryEngine=object(), RetrieverQueryEngine=object()
            )
        if name == "llama_index.core.selectors":
            return SimpleNamespace(
                LLMSingleSelector=object(), PydanticSingleSelector=_Pydantic
            )
        if name == "llama_index.core.tools":
            return SimpleNamespace(QueryEngineTool=object(), ToolMetadata=object())
        raise ImportError(name)

    monkeypatch.setattr(lia, "import_module", _fake_import_with_pydantic)
    adapter = lia._build_real_adapter()
    llm = SimpleNamespace(metadata=SimpleNamespace(is_function_calling_model=False))
    assert adapter.get_pydantic_selector(llm=llm) is None
    llm.metadata.is_function_calling_model = True
    assert adapter.get_pydantic_selector(llm=llm) == "SEL"
