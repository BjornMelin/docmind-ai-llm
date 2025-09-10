"""Test selector fallback when PydanticSingleSelector is unavailable."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


def test_selector_fallback_to_llm(monkeypatch):  # type: ignore[no-untyped-def]
    # Force import error for PydanticSingleSelector
    fake_sel = ModuleType("llama_index.core.selectors")

    class _LLMSingleSelector:
        @classmethod
        def from_defaults(cls, llm=None):  # type: ignore[override]
            return SimpleNamespace(kind="llm_selector")

    fake_sel.LLMSingleSelector = _LLMSingleSelector
    monkeypatch.setitem(sys.modules, "llama_index.core.selectors", fake_sel)

    # Stub query engines and tools
    qe_mod = ModuleType("llama_index.core.query_engine")
    tools_mod = ModuleType("llama_index.core.tools")

    class _RetrieverQueryEngine:
        @classmethod
        def from_args(cls, retriever, llm=None, response_mode=None, verbose=False):  # type: ignore[override]
            return SimpleNamespace(kind="retriever_engine", retriever=retriever)

    class _RouterQueryEngine:
        def __init__(self, selector, query_engine_tools, verbose=False):  # type: ignore[override]
            self.selector = selector
            self.tools = query_engine_tools

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
    tools_mod.QueryEngineTool = _QueryEngineTool
    tools_mod.ToolMetadata = _ToolMetadata

    monkeypatch.setitem(sys.modules, "llama_index.core.query_engine", qe_mod)
    monkeypatch.setitem(sys.modules, "llama_index.core.tools", tools_mod)

    # Stub hybrid retriever import
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

    class _VecIdx:
        def as_query_engine(self, *_, **__):
            return SimpleNamespace(kind="vector_engine")

    router = build_router_engine(_VecIdx(), None, _settings)
    assert getattr(router.selector, "kind", "") == "llm_selector"
