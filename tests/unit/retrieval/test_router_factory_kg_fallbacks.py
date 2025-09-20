"""RouterFactory KG fallbacks when node_postprocessors unsupported."""

from __future__ import annotations

from types import SimpleNamespace

from src.retrieval import router_factory as rf


def _count_tools(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def _make_adapter(fail_on_post: bool):
    class _QueryEngineTool:
        def __init__(self, query_engine, metadata):
            self.query_engine = query_engine
            self.metadata = metadata

    class _RouterQueryEngine:
        @classmethod
        def from_args(cls, **kwargs):
            if fail_on_post and "node_postprocessors" in kwargs:
                raise TypeError("node_postprocessors unsupported")
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
            self.query_engine_tools = query_engine_tools or []
            self.verbose = verbose
            self.llm = llm
            self.kwargs = kwargs

    class _ToolMetadata:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _LLMSingleSelector:
        @classmethod
        def from_defaults(cls, llm=None):
            return SimpleNamespace(llm=llm)

    return SimpleNamespace(
        RouterQueryEngine=_RouterQueryEngine,
        RetrieverQueryEngine=_RouterQueryEngine,
        QueryEngineTool=_QueryEngineTool,
        ToolMetadata=_ToolMetadata,
        LLMSingleSelector=_LLMSingleSelector,
        get_pydantic_selector=lambda _llm: None,
        __is_stub__=False,
        supports_graphrag=True,
        graphrag_disabled_reason="",
    )


def test_router_factory_kg_as_query_engine_fallback():  # type: ignore[no-untyped-def]
    adapter = _make_adapter(fail_on_post=True)

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

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

    router = rf.build_router_engine(
        _Vec(),
        _Pg(),
        settings=_Cfg,
        llm=SimpleNamespace(),
        enable_hybrid=False,
        adapter=adapter,
    )
    assert _count_tools(router) == 2  # vector + kg


def test_router_factory_kg_retriever_fallback():  # type: ignore[no-untyped-def]
    adapter = _make_adapter(fail_on_post=True)

    class _Vec:
        def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(qe=True, kwargs=kwargs)

    class _Pg:
        def __init__(self):
            self.property_graph_store = object()

        def as_retriever(self, **kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(retriever=True)

    class _Cfg:
        class retrieval:  # noqa: N801
            top_k = 5
            use_reranking = True
            reranking_top_k = 3

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

    router = rf.build_router_engine(
        _Vec(),
        _Pg(),
        settings=_Cfg,
        llm=SimpleNamespace(),
        enable_hybrid=False,
        adapter=adapter,
    )
    assert _count_tools(router) == 2
