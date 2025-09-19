"""RouterFactory KG fallbacks when node_postprocessors unsupported.

Asserts that when PG index methods reject node_postprocessors (TypeError),
router_factory falls back to calls without that argument and still registers
KG tool without raising exceptions.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace


def _count_tools(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def test_router_factory_kg_as_query_engine_fallback(monkeypatch):  # type: ignore[no-untyped-def]
    rf = importlib.import_module("src.retrieval.router_factory")

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self):
            self.property_graph_store = object()

        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            if "node_postprocessors" in kwargs:
                raise TypeError("node_postprocessors unsupported")
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _QET:  # capture tools
        def __init__(self, query_engine, metadata):
            self.query_engine = query_engine
            self.metadata = metadata

    class _RQE:
        def __init__(self, selector=None, query_engine_tools=None, verbose=False):
            self.selector = selector
            self.query_engine_tools = query_engine_tools or []
            self.verbose = verbose

    monkeypatch.setattr(rf, "QueryEngineTool", _QET)
    monkeypatch.setattr(rf, "RouterQueryEngine", _RQE)

    class _Cfg:
        class retrieval:  # noqa: N801
            top_k = 5
            use_reranking = True
            reranking_top_k = 3

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

    router = rf.build_router_engine(
        _Vec(), _Pg(), settings=_Cfg, llm=SimpleNamespace(), enable_hybrid=False
    )
    assert _count_tools(router) == 2  # vector + kg


def test_router_factory_kg_retriever_fallback(monkeypatch):  # type: ignore[no-untyped-def]
    rf = importlib.import_module("src.retrieval.router_factory")

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self):
            self.property_graph_store = object()

        def as_retriever(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(retriever=True)

    class _QET:
        def __init__(self, query_engine, metadata):
            self.query_engine = query_engine
            self.metadata = metadata

    class _RQE:
        @classmethod
        def from_args(cls, **kwargs):  # type: ignore[no-untyped-def]
            if "node_postprocessors" in kwargs:
                raise TypeError("node_postprocessors unsupported")
            return SimpleNamespace(qe=True, kwargs=kwargs)

        def __init__(self, selector=None, query_engine_tools=None, verbose=False):
            self.selector = selector
            self.query_engine_tools = query_engine_tools or []
            self.verbose = verbose

    monkeypatch.setattr(rf, "QueryEngineTool", _QET)
    monkeypatch.setattr(rf, "RouterQueryEngine", _RQE)
    monkeypatch.setattr(rf, "RetrieverQueryEngine", _RQE)

    class _Cfg:
        class retrieval:  # noqa: N801
            top_k = 5
            use_reranking = True
            reranking_top_k = 3

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

    router = rf.build_router_engine(
        _Vec(), _Pg(), settings=_Cfg, llm=SimpleNamespace(), enable_hybrid=False
    )
    assert _count_tools(router) == 2
