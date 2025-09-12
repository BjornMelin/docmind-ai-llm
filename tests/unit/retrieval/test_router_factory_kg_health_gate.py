"""Test KG tool health gate via shallow probe.

Ensures KG tool is only registered when a shallow probe returns results.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def _count_tools(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def _has_tool(router, name: str) -> bool:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is None:
            continue
        for t in tools:
            if getattr(getattr(t, "metadata", SimpleNamespace()), "name", "") == name:
                return True
    return False


def test_kg_tool_absent_when_probe_empty(monkeypatch):  # type: ignore[no-untyped-def]
    rf = importlib.import_module("src.retrieval.router_factory")

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _ProbeRetr:
        def retrieve(self, _q):  # type: ignore[no-untyped-def]
            return []  # empty probe

    class _Pg:
        def __init__(self):
            self.property_graph_store = object()

        def as_retriever(self, **_):  # type: ignore[no-untyped-def]
            return _ProbeRetr()

    class _QET:
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
    # Only vector tool should be present when probe is empty
    assert _count_tools(router) == 1
    assert _has_tool(router, "knowledge_graph") is False


def test_kg_tool_present_when_probe_ok(monkeypatch):  # type: ignore[no-untyped-def]
    rf = importlib.import_module("src.retrieval.router_factory")

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _ProbeRetr:
        def retrieve(self, _q):  # type: ignore[no-untyped-def]
            return [1]  # non-empty probe

    class _Pg:
        def __init__(self):
            self.property_graph_store = object()

        def as_retriever(self, **_):  # type: ignore[no-untyped-def]
            return _ProbeRetr()

    class _QET:
        def __init__(self, query_engine, metadata):
            self.query_engine = query_engine
            self.metadata = metadata

    class _RQE:
        @classmethod
        def from_args(cls, **kwargs):  # type: ignore[no-untyped-def]
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
    # vector + KG
    assert _count_tools(router) == 2
    assert _has_tool(router, "knowledge_graph") is True

