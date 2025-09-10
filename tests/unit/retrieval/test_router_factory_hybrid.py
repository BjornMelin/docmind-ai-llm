"""Unit tests for router_factory hybrid tool composition."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _stub_li_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub llama_index modules used by router_factory
    qe_mod = ModuleType("llama_index.core.query_engine")
    sel_mod = ModuleType("llama_index.core.selectors")
    tools_mod = ModuleType("llama_index.core.tools")

    class _RetrieverQueryEngine:
        @classmethod
        def from_args(cls, retriever, llm=None, response_mode=None, verbose=False):  # type: ignore[override]
            return SimpleNamespace(kind="retriever_engine", retriever=retriever)

    class _RouterQueryEngine:
        def __init__(self, selector, query_engine_tools, verbose=False):  # type: ignore[override]
            self.selector = selector
            self.tools = query_engine_tools

    class _LLMSingleSelector:
        @classmethod
        def from_defaults(cls, llm=None):  # type: ignore[override]
            return SimpleNamespace(kind="llm_selector")

    class _QueryEngineTool:
        def __init__(self, query_engine, metadata):  # type: ignore[override]
            self.query_engine = query_engine
            self.metadata = metadata

    class _ToolMetadata:
        def __init__(self, name, description):  # type: ignore[override]
            self.name = name
            self.description = description

    qe_mod.RetrieverQueryEngine = _RetrieverQueryEngine
    qe_mod.RouterQueryEngine = _RouterQueryEngine
    sel_mod.LLMSingleSelector = _LLMSingleSelector
    tools_mod.QueryEngineTool = _QueryEngineTool
    tools_mod.ToolMetadata = _ToolMetadata

    monkeypatch.setitem(sys.modules, "llama_index.core.query_engine", qe_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.selectors", sel_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", tools_mod)


def test_router_includes_hybrid_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub ServerHybridRetriever to avoid dependencies
    hyr_mod = ModuleType("src.retrieval.hybrid")

    class _DummyHybridRetriever:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class _Params:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    hyr_mod.ServerHybridRetriever = _DummyHybridRetriever
    hyr_mod._HybridParams = _Params  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.retrieval.hybrid", hyr_mod)

    from src.config.settings import settings as _settings
    from src.retrieval.router_factory import build_router_engine

    _settings.retrieval.top_k = 5

    class _VecIdx:
        def as_query_engine(self, *_, **__):
            return SimpleNamespace(kind="vector_engine")

    class _PgIdx:
        def as_retriever(self, include_text=False, path_depth=1):  # type: ignore[override]
            return SimpleNamespace(kind="kg_retriever")

        @property
        def property_graph_store(self):
            return object()

    router = build_router_engine(_VecIdx(), _PgIdx(), _settings)
    # Ensure three tools present: semantic_search, hybrid_search, knowledge_graph
    names = [t.metadata.name for t in router.tools]
    assert "semantic_search" in names
    assert "hybrid_search" in names
    assert "knowledge_graph" in names
