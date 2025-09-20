"""RouterFactory rerank injection toggle tests.

Validates that node_postprocessors are injected when reranking is enabled and
omitted otherwise, using a lightweight adapter to avoid importing llama_index.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.retrieval import router_factory as rf


class _FakeVector:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return SimpleNamespace(qe=True, kwargs=kwargs)


class _FakeKG:
    def __init__(self) -> None:
        self.property_graph_store = object()
        self.kwargs: dict[str, object] | None = None

    def as_query_engine(self, **kwargs):  # type: ignore[no-untyped-def]
        self.kwargs = kwargs
        return SimpleNamespace(qe=True, kwargs=kwargs)


def _count_tools(router) -> int:  # type: ignore[no-untyped-def]
    for attr in ("query_engine_tools", "_query_engine_tools"):
        tools = getattr(router, attr, None)
        if tools is not None:
            return len(list(tools))
    return 0


def _make_adapter(captured: list[object]):
    class _QueryEngineTool:
        def __init__(self, query_engine, metadata):
            captured.append(query_engine)
            self.query_engine = query_engine
            self.metadata = metadata

    class _RouterQueryEngine:
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

        @classmethod
        def from_args(cls, **kwargs):  # type: ignore[no-untyped-def]
            return cls(**kwargs)

    class _ToolMetadata:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _LLMSingleSelector:
        @classmethod
        def from_defaults(cls, llm=None):  # pragma: no cover - simple stub
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


def test_router_factory_injects_postprocessors_toggle():  # type: ignore[no-untyped-def]
    captured: list[object] = []
    adapter = _make_adapter(captured)

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
        vec,
        kg,
        settings=_Cfg,
        llm=SimpleNamespace(),
        enable_hybrid=False,
        adapter=adapter,
    )
    assert _count_tools(router) == 2
    assert vec.kwargs is not None
    assert vec.kwargs.get("node_postprocessors") is not None
    assert kg.kwargs is not None
    assert kg.kwargs.get("node_postprocessors") is not None

    _Cfg.retrieval.use_reranking = False
    vec2 = _FakeVector()
    kg2 = _FakeKG()
    router2 = rf.build_router_engine(
        vec2,
        kg2,
        settings=_Cfg,
        llm=SimpleNamespace(),
        enable_hybrid=False,
        adapter=adapter,
    )
    assert _count_tools(router2) == 2
    assert vec2.kwargs is not None
    assert vec2.kwargs.get("node_postprocessors") is None
    assert kg2.kwargs is not None
    assert kg2.kwargs.get("node_postprocessors") is None

    # Ensure captured engines were built both times
    assert len(captured) >= 2
