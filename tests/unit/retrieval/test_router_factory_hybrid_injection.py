"""RouterFactory hybrid rerank injection tests (enable_hybrid=True).

Verifies that when reranking is enabled, the hybrid tool receives
node_postprocessors via ``RetrieverQueryEngine.from_args``; and when disabled,
no node_postprocessors are passed. Uses a lightweight adapter to avoid
importing llama_index during unit tests.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.retrieval import router_factory as rf


def _tool_count(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def _make_adapter():
    class _RQE:
        last_kwargs = None  # type: ignore[var-annotated]

        @classmethod
        def from_args(cls, **kwargs):  # type: ignore[no-untyped-def]
            cls.last_kwargs = kwargs
            return SimpleNamespace(qe=True, kwargs=kwargs)

        def __init__(
            self,
            *,
            selector=None,
            query_engine_tools=None,
            verbose=False,
            llm=None,
            **kwargs,
        ) -> None:
            self.selector = selector
            self.query_engine_tools = list(query_engine_tools or [])
            self.verbose = verbose
            self.llm = llm
            self.kwargs = kwargs

    class _QET:
        def __init__(self, query_engine, metadata):
            self.query_engine = query_engine
            self.metadata = metadata

    class _ToolMetadata:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _LLMSingleSelector:
        @classmethod
        def from_defaults(cls, llm=None):
            return SimpleNamespace(llm=llm)

    return SimpleNamespace(
        RouterQueryEngine=_RQE,
        RetrieverQueryEngine=_RQE,
        QueryEngineTool=_QET,
        ToolMetadata=_ToolMetadata,
        LLMSingleSelector=_LLMSingleSelector,
        get_pydantic_selector=lambda _llm: None,
        __is_stub__=False,
        supports_graphrag=True,
        graphrag_disabled_reason="",
    )


def test_hybrid_rerank_injection_toggle(monkeypatch):  # type: ignore[no-untyped-def]
    adapter = _make_adapter()

    class _DummyHybrid:
        def __init__(self, *_args, **_kwargs):  # type: ignore[no-untyped-def]
            pass

    monkeypatch.setattr("src.retrieval.hybrid.ServerHybridRetriever", _DummyHybrid)

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self):
            self.property_graph_store = object()

        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Cfg:
        class retrieval:  # noqa: N801
            top_k = 5
            use_reranking = True
            reranking_top_k = 3
            enable_server_hybrid = False

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

        class database:  # noqa: N801
            qdrant_collection = "c"

    vec = _Vec()
    pg = _Pg()

    router = rf.build_router_engine(
        vec,
        pg,
        settings=_Cfg,
        llm=SimpleNamespace(),
        enable_hybrid=True,
        adapter=adapter,
    )
    assert _tool_count(router) == 3
    assert adapter.RouterQueryEngine.last_kwargs is not None
    assert adapter.RouterQueryEngine.last_kwargs.get("node_postprocessors") is not None

    _Cfg.retrieval.use_reranking = False
    adapter.RouterQueryEngine.last_kwargs = None
    vec2 = _Vec()
    pg2 = _Pg()
    router2 = rf.build_router_engine(
        vec2,
        pg2,
        settings=_Cfg,
        llm=SimpleNamespace(),
        enable_hybrid=True,
        adapter=adapter,
    )
    assert _tool_count(router2) == 3
    assert adapter.RouterQueryEngine.last_kwargs is not None
    assert adapter.RouterQueryEngine.last_kwargs.get("node_postprocessors") is None
