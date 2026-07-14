"""Async contract coverage for every native router tool."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Coroutine, Sequence
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import (
    BaseSelector,
    SelectorResult,
    SingleSelection,
)
from llama_index.core.llms import MockLLM
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from qdrant_client import models as qmodels

from src.retrieval import hybrid as hybrid_module
from src.retrieval import keyword as keyword_module
from src.retrieval import multimodal_fusion as multimodal_module
from src.retrieval import router_factory
from src.retrieval.hybrid import HybridParams, ServerHybridRetriever
from src.retrieval.keyword import KeywordParams, KeywordSparseRetriever
from src.retrieval.multimodal_fusion import MultimodalFusionRetriever
from src.retrieval.router_factory import DocMindRouterQueryEngine
from src.ui.router_session import replace_session_router

pytestmark = pytest.mark.unit


def _node(node_id: str) -> NodeWithScore:
    return NodeWithScore(node=TextNode(text=node_id, id_=node_id), score=1.0)


class _StaticRetriever(BaseRetriever):
    def __init__(self, node_id: str) -> None:
        super().__init__()
        self._node = _node(node_id)
        self.closed = False

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        del query_bundle
        return [self._node]

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        del query_bundle
        return [self._node]

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


class _FixedSelector(BaseSelector):
    def __init__(self, index: int) -> None:
        self._index = index

    def _result(self) -> SelectorResult:
        return SelectorResult(
            selections=[SingleSelection(index=self._index, reason="test")]
        )

    def _get_prompts(self) -> PromptDictType:
        return {}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        del prompts_dict

    def _select(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> SelectorResult:
        del choices, query
        return self._result()

    async def _aselect(
        self,
        choices: Sequence[ToolMetadata],
        query: QueryBundle,
    ) -> SelectorResult:
        del choices, query
        return self._result()


class _Client:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Point:
    def __init__(self) -> None:
        self.id = "point"
        self.score = 1.0
        self.payload = {"page_id": "page", "text": "result"}


class _AsyncHybridClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def query_points_groups(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(groups=[SimpleNamespace(hits=[_Point()])])

    async def close(self) -> None:
        self.close_calls += 1


class _AsyncKeywordClient:
    def __init__(self) -> None:
        self.close_calls = 0

    async def query_points(self, **_kwargs: Any) -> Any:
        return SimpleNamespace(points=[_Point()])

    async def close(self) -> None:
        self.close_calls += 1


class _Embedding:
    def get_query_embedding(self, _text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _VectorIndex:
    def __init__(self, llm: MockLLM) -> None:
        self._llm = llm

    def as_query_engine(self, **_kwargs: Any) -> RetrieverQueryEngine:
        return RetrieverQueryEngine.from_args(
            retriever=_StaticRetriever("semantic"),
            llm=self._llm,
        )


class _LoopThread:
    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.started = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self.started.wait(timeout=1.0)

    def _run(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.started.set()
        self.loop.run_forever()
        self.loop.close()

    def run(self, coroutine: Coroutine[Any, Any, Any]) -> Any:
        future: Future[Any] = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        return future.result(timeout=1.0)

    def close(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=1.0)


class _ManagedClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.close_calls = 0
        self.close_thread: int | None = None
        self.fail = fail

    def close(self) -> None:
        raise AssertionError("router lifecycle must use async cleanup")

    async def aclose(self) -> None:
        self.close_calls += 1
        self.close_thread = threading.get_ident()
        if self.fail:
            raise RuntimeError("client cleanup failed")


class _ManagedPostprocessor:
    def __init__(self) -> None:
        self.close_calls = 0

    def postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        **_kwargs: Any,
    ) -> list[NodeWithScore]:
        return nodes

    async def apostprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        **_kwargs: Any,
    ) -> list[NodeWithScore]:
        return nodes

    def close(self) -> None:
        raise AssertionError("router lifecycle must use async cleanup")

    async def aclose(self) -> None:
        self.close_calls += 1


class _SlowAsyncManagedClient:
    def __init__(self) -> None:
        self.close_calls = 0
        self.aclose_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    async def aclose(self) -> None:
        self.aclose_calls += 1
        await asyncio.sleep(0.1)


async def test_vector_embedding_runs_on_owned_executor_before_native_async_query() -> (
    None
):
    """The semantic route never enters an embedding model's async thread pool."""

    class _VectorEmbedding:
        def __init__(self) -> None:
            self.thread_name: str | None = None

        def get_agg_embedding_from_queries(
            self,
            _queries: list[str],
        ) -> list[float]:
            self.thread_name = threading.current_thread().name
            return [0.1, 0.2, 0.3]

        async def aget_agg_embedding_from_queries(
            self,
            _queries: list[str],
        ) -> list[float]:
            raise AssertionError("global async embedding path must not be used")

    class _NativeVectorRetriever(BaseRetriever):
        def __init__(self, embed_model: _VectorEmbedding) -> None:
            super().__init__()
            self._embed_model = embed_model

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            assert query_bundle.embedding == [0.1, 0.2, 0.3]
            return [_node("semantic")]

        async def _aretrieve(
            self,
            query_bundle: QueryBundle,
        ) -> list[NodeWithScore]:
            if query_bundle.embedding is None:
                await self._embed_model.aget_agg_embedding_from_queries(
                    query_bundle.embedding_strs
                )
            return self._retrieve(query_bundle)

    embed_model = _VectorEmbedding()
    wrapped = router_factory._OwnedVectorEmbeddingRetriever(
        _NativeVectorRetriever(embed_model),
        embed_model,
    )

    assert [node.node.node_id for node in await wrapped.aretrieve("query")] == [
        "semantic"
    ]
    assert embed_model.thread_name is not None
    assert embed_model.thread_name.startswith("docmind-vector-embedding")

    await wrapped.aclose()


def _bound_router(
    runner: _LoopThread,
    *,
    first_fails: bool = False,
) -> tuple[DocMindRouterQueryEngine, tuple[_ManagedClient, _ManagedClient]]:
    llm = MockLLM(max_tokens=8)
    tool = QueryEngineTool(
        query_engine=RetrieverQueryEngine.from_args(
            retriever=_StaticRetriever("semantic"),
            llm=llm,
        ),
        metadata=ToolMetadata(name="semantic", description="test"),
    )
    router = DocMindRouterQueryEngine(
        _FixedSelector(0),
        [tool],
        llm=llm,
    )
    first = _ManagedClient(fail=first_fails)
    second = _ManagedClient()
    router._managed_retrievers = (second, first)
    runner.run(router.aquery("bind owner loop"))
    return router, (first, second)


async def test_router_aquery_supports_every_retriever_and_closes_owned_clients(
    monkeypatch: pytest.MonkeyPatch,
    router_settings: Any,
) -> None:
    sync_hybrid = _Client()
    async_hybrid = _AsyncHybridClient()
    sync_keyword = _Client()
    async_keyword = _AsyncKeywordClient()
    text = _StaticRetriever("text")
    image = _StaticRetriever("image")

    monkeypatch.setattr(
        hybrid_module,
        "check_hybrid_collection",
        lambda *_a, **_k: SimpleNamespace(compatible=True),
    )
    monkeypatch.setattr(hybrid_module, "get_settings_embed_model", _Embedding)
    monkeypatch.setattr(
        keyword_module,
        "encode_to_qdrant",
        lambda _text: qmodels.SparseVector(indices=[1], values=[1.0]),
    )

    hybrid = ServerHybridRetriever(
        HybridParams(collection="test"),
        client=sync_hybrid,  # type: ignore[arg-type]
        async_client=async_hybrid,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(hybrid, "_encode_sparse", lambda _text: None)
    keyword = KeywordSparseRetriever(
        KeywordParams(collection="test"),
        client=sync_keyword,  # type: ignore[arg-type]
        async_client=async_keyword,  # type: ignore[arg-type]
    )
    multimodal = MultimodalFusionRetriever(
        text_retriever=text,  # type: ignore[arg-type]
        image_retriever=image,  # type: ignore[arg-type]
    )

    monkeypatch.setattr(hybrid_module, "ServerHybridRetriever", lambda _p: hybrid)
    monkeypatch.setattr(keyword_module, "KeywordSparseRetriever", lambda _p: keyword)
    monkeypatch.setattr(
        multimodal_module,
        "MultimodalFusionRetriever",
        lambda **_kwargs: multimodal,
    )
    postprocessors: list[_ManagedPostprocessor] = []

    def _postprocessors(*_args: Any, **_kwargs: Any) -> list[_ManagedPostprocessor]:
        postprocessor = _ManagedPostprocessor()
        postprocessors.append(postprocessor)
        return [postprocessor]

    monkeypatch.setattr(router_factory, "get_postprocessors", _postprocessors)

    router_settings.retrieval.enable_image_retrieval = True
    router_settings.retrieval.enable_server_hybrid = True
    router_settings.retrieval.enable_keyword_tool = True
    llm = MockLLM(max_tokens=32)
    router = router_factory.build_router_engine(
        _VectorIndex(llm),  # type: ignore[arg-type]
        settings=router_settings,
        llm=llm,
    )

    assert isinstance(router, DocMindRouterQueryEngine)
    tool_names = [metadata.name for metadata in router._metadatas]
    assert tool_names == [
        "semantic_search",
        "multimodal_search",
        "hybrid_search",
        "keyword_search",
    ]
    for index, tool_name in enumerate(tool_names):
        router._selector = _FixedSelector(index)
        response = await router.aquery(tool_name)
        assert response is not None, tool_name

    await router.aclose()
    await router.aclose()
    assert sync_hybrid.closed is True
    assert async_hybrid.close_calls == 1
    assert sync_keyword.closed is True
    assert async_keyword.close_calls == 1
    assert text.closed is True
    assert image.closed is True
    assert len(postprocessors) == 3
    assert [postprocessor.close_calls for postprocessor in postprocessors] == [1, 1, 1]


def test_graph_reranker_is_registered_for_router_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    router_settings: Any,
) -> None:
    postprocessor = _ManagedPostprocessor()
    llm = MockLLM(max_tokens=8)
    query_engine = RetrieverQueryEngine.from_args(
        retriever=_StaticRetriever("graph"),
        llm=llm,
    )
    monkeypatch.setattr(
        router_factory,
        "get_postprocessors",
        lambda *_a, **_k: [postprocessor],
    )
    monkeypatch.setattr(
        router_factory,
        "build_graph_query_engine",
        lambda *_a, **_k: SimpleNamespace(query_engine=query_engine),
    )
    tools: list[QueryEngineTool] = []
    resources: list[router_factory._ManagedRetriever] = []

    router_factory._maybe_add_graph_tool(
        tools=tools,
        managed_retrievers=resources,
        settings=router_settings,
        pg_index=SimpleNamespace(property_graph_store=object()),  # type: ignore[arg-type]
        llm=llm,
    )

    assert len(tools) == 1
    assert resources == [postprocessor]


def test_sync_only_router_close_does_not_wait_for_async_cleanup() -> None:
    llm = MockLLM(max_tokens=8)
    tool = QueryEngineTool(
        query_engine=RetrieverQueryEngine.from_args(
            retriever=_StaticRetriever("semantic"),
            llm=llm,
        ),
        metadata=ToolMetadata(name="semantic", description="test"),
    )
    router = DocMindRouterQueryEngine(_FixedSelector(0), [tool], llm=llm)
    resource = _SlowAsyncManagedClient()
    router._managed_retrievers = (resource,)

    assert router.query("bind sync path") is not None
    started = time.perf_counter()
    router.close()
    elapsed = time.perf_counter() - started
    router.close()

    assert elapsed < 0.05
    assert resource.close_calls == 1
    assert resource.aclose_calls == 0


async def test_aclose_after_owner_loop_stops_uses_sync_cleanup() -> None:
    runner = _LoopThread()
    llm = MockLLM(max_tokens=8)
    tool = QueryEngineTool(
        query_engine=RetrieverQueryEngine.from_args(
            retriever=_StaticRetriever("semantic"),
            llm=llm,
        ),
        metadata=ToolMetadata(name="semantic", description="test"),
    )
    router = DocMindRouterQueryEngine(_FixedSelector(0), [tool], llm=llm)
    resource = _SlowAsyncManagedClient()
    router._managed_retrievers = (resource,)
    runner.run(router.aquery("bind owner loop"))
    runner.close()

    started = time.perf_counter()
    await router.aclose()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05
    assert resource.close_calls == 1
    assert resource.aclose_calls == 0


def test_session_replace_clear_and_cleanup_error_close_on_owner_loop() -> None:
    runner = _LoopThread()
    try:
        scenarios = (
            (object(), False),
            (None, False),
            (object(), True),
        )
        for replacement, first_fails in scenarios:
            router, clients = _bound_router(runner, first_fails=first_fails)
            state = {"router_engine": router}

            replace_session_router(
                state,
                replacement,
                runtime_generation=1,
            )  # type: ignore[arg-type]
            router.close()

            assert state["router_engine"] is replacement
            assert [client.close_calls for client in clients] == [1, 1]
            assert {client.close_thread for client in clients} == {runner.thread.ident}
    finally:
        runner.close()
