"""RouterFactory rerank injection toggle tests.

Asserts that when DOCMIND_RETRIEVAL__USE_RERANKING (via settings) is True,
router_factory injects node_postprocessors for vector and KG tools; and omits
them when False. We stub RouterQueryEngine and QueryEngineTool to avoid LLM
resolution and capture constructed engines.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


class _FakeVector:
    def __init__(self) -> None:
        self.kwargs = None

    def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return SimpleNamespace(qe=True, kwargs=kwargs)


class _FakeKG:
    def __init__(self) -> None:
        self.property_graph_store = object()
        self.kwargs = None

    def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return SimpleNamespace(qe=True, kwargs=kwargs)


def _count_tools(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def test_router_factory_injects_postprocessors_toggle(monkeypatch):  # type: ignore[no-untyped-def]
    rf = importlib.import_module("src.retrieval.router_factory")

    captured = []

    class _QET:  # minimal QueryEngineTool stub
        def __init__(self, query_engine, metadata):
            captured.append(query_engine)
            self.query_engine = query_engine
            self.metadata = metadata

    class _RQE:  # minimal RouterQueryEngine stub
        def __init__(self, selector=None, query_engine_tools=None, verbose=False):
            self.selector = selector
            self.query_engine_tools = query_engine_tools or []
            self.verbose = verbose

    monkeypatch.setattr(rf, "QueryEngineTool", _QET)
    monkeypatch.setattr(rf, "RouterQueryEngine", _RQE)

    # Build with reranking enabled
    class _Cfg:
        class retrieval:  # noqa: N801
            top_k = 5
            use_reranking = True
            reranking_top_k = 3

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

    vec = _FakeVector()
    kg = _FakeKG()
    router = rf.build_router_engine(
        vec, kg, settings=_Cfg, llm=SimpleNamespace(), enable_hybrid=False
    )
    assert _count_tools(router) == 2
    # Ensure vector and KG engines include node_postprocessors
    assert vec.kwargs is not None
    assert vec.kwargs.get("node_postprocessors") is not None
    assert kg.kwargs is not None
    assert kg.kwargs.get("node_postprocessors") is not None

    # Now disable reranking and rebuild
    _Cfg.retrieval.use_reranking = False
    vec2 = _FakeVector()
    kg2 = _FakeKG()
    router2 = rf.build_router_engine(
        vec2, kg2, settings=_Cfg, llm=SimpleNamespace(), enable_hybrid=False
    )
    assert _count_tools(router2) == 2
    assert vec2.kwargs is not None
    assert vec2.kwargs.get("node_postprocessors") is None
    assert kg2.kwargs is not None
    assert kg2.kwargs.get("node_postprocessors") is None
