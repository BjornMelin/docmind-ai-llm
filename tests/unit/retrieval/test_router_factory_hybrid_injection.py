"""RouterFactory hybrid rerank injection tests (enable_hybrid=True).

Verifies that when reranking is enabled, the hybrid tool receives
node_postprocessors via RetrieverQueryEngine.from_args; and when disabled,
no node_postprocessors are passed.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace


def _tool_count(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def test_hybrid_rerank_injection_toggle(monkeypatch):  # type: ignore[no-untyped-def]
    rf = importlib.import_module("src.retrieval.router_factory")

    # Patch ServerHybridRetriever to avoid external deps
    class _DummyHybrid:
        def __init__(self, *_a, **_k):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr(rf, "ServerHybridRetriever", _DummyHybrid)

    # Capture from_args kwargs
    class _RQE:
        last_kwargs = None  # type: ignore[var-annotated]

        @classmethod
        def from_args(cls, **kwargs):  # type: ignore[no-untyped-def]
            cls.last_kwargs = kwargs
            return SimpleNamespace(qe=True, kwargs=kwargs)

        def __init__(self, selector=None, query_engine_tools=None, verbose=False):
            self.selector = selector
            self.query_engine_tools = query_engine_tools or []
            self.verbose = verbose

    monkeypatch.setattr(rf, "RetrieverQueryEngine", _RQE)
    monkeypatch.setattr(rf, "RouterQueryEngine", _RQE)

    # Patch QueryEngineTool to avoid library internals
    class _QET:  # minimal tool wrapper
        def __init__(self, query_engine, metadata):
            self.query_engine = query_engine
            self.metadata = metadata

    monkeypatch.setattr(rf, "QueryEngineTool", _QET)

    # Stub vector and KG to keep router building
    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self):
            self.property_graph_store = object()

        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    # Settings stub with reranking enabled
    class _Cfg:
        class retrieval:  # noqa: N801
            top_k = 5
            use_reranking = True
            reranking_top_k = 3
            enable_server_hybrid = False  # we will use explicit enable_hybrid=True

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

        class database:  # noqa: N801
            qdrant_collection = "c"

    vec = _Vec()
    pg = _Pg()

    # Build with hybrid enabled and reranking True
    router = rf.build_router_engine(
        vec, pg, settings=_Cfg, llm=SimpleNamespace(), enable_hybrid=True
    )
    count = _tool_count(router)
    assert count == 3
    assert _RQE.last_kwargs is not None
    assert "node_postprocessors" in _RQE.last_kwargs

    # Disable reranking and rebuild
    _Cfg.retrieval.use_reranking = False
    _RQE.last_kwargs = None
    router2 = rf.build_router_engine(
        vec, pg, settings=_Cfg, llm=SimpleNamespace(), enable_hybrid=True
    )
    count2 = _tool_count(router2)
    assert count2 == 3
    assert _RQE.last_kwargs is not None
    assert _RQE.last_kwargs.get("node_postprocessors") is None
