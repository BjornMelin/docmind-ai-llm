"""Test KG tool health gate via shallow probe using lightweight adapter."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.retrieval import router_factory as rf

pytest.importorskip(
    "llama_index.program.openai",
    reason="requires llama_index.program.openai",
)


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
            metadata = getattr(t, "metadata", SimpleNamespace())
            if getattr(metadata, "name", "") == name:
                return True
    return False


def _make_adapter():
    class _ToolMetadata:
        def __init__(self, name: str, description: str) -> None:
            self.name = name
            self.description = description

    class _QueryEngineTool:
        def __init__(self, query_engine, metadata):
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


def test_kg_tool_absent_when_probe_empty(monkeypatch):  # type: ignore[no-untyped-def]
    adapter = _make_adapter()

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
    assert _count_tools(router) == 1
    assert _has_tool(router, "knowledge_graph") is False


def test_kg_tool_present_when_probe_ok(monkeypatch):  # type: ignore[no-untyped-def]
    adapter = _make_adapter()

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

    class _Cfg:
        class retrieval:  # noqa: N801
            top_k = 5
            use_reranking = True
            reranking_top_k = 3

        class graphrag_cfg:  # noqa: N801
            default_path_depth = 1

    class _Artifacts:
        def __init__(self):
            self.query_engine = SimpleNamespace(kind="kg")
            self.retriever = SimpleNamespace()

    monkeypatch.setattr(
        rf,
        "build_graph_query_engine",
        lambda *_a, **_k: _Artifacts(),
    )

    router = rf.build_router_engine(
        _Vec(),
        _Pg(),
        settings=_Cfg,
        llm=SimpleNamespace(),
        enable_hybrid=False,
        adapter=adapter,
    )
    assert _count_tools(router) == 2
    assert _has_tool(router, "knowledge_graph") is True
