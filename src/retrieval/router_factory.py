"""Router factory orchestrating vector, hybrid, and GraphRAG tools.

Builds a RouterQueryEngine with:
- semantic_search (vector)
- optional hybrid_search (server-side hybrid retriever)
- optional keyword_search (sparse-only exact-term retriever)
- optional knowledge_graph (GraphRAG)

LlamaIndex core is a required dependency; the factory uses its native query
engine and tool APIs directly.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Protocol, cast

from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.base.response.schema import RESPONSE_TYPE
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from loguru import logger

from src.config.settings import DocMindSettings
from src.config.settings import settings as default_settings
from src.retrieval.async_work import AsyncWorkExecutor
from src.retrieval.graph_config import build_graph_query_engine
from src.retrieval.image_index import ImageCollectionIncompatibleError
from src.retrieval.reranking import get_postprocessors
from src.telemetry.opentelemetry import record_router_selection, router_build_span
from src.utils.log_safety import build_pii_log_entry
from src.utils.storage import sparse_retrieval_enabled

_WARNED_TOOLS: set[str] = set()
_ROUTER_CLOSE_GRACE_S = 5.0


class _ManagedRetriever(Protocol):
    """Closeable resources created and owned by the router factory."""

    def close(self) -> None: ...

    async def aclose(self) -> None: ...


class _AggregateEmbeddingModel(Protocol):
    """Minimal embedding surface required by the owned retriever."""

    def get_agg_embedding_from_queries(self, queries: list[str]) -> list[float]: ...


class _OwnedVectorEmbeddingRetriever(BaseRetriever):
    """Run synchronous vector embeddings outside asyncio's global executor.

    LlamaIndex's Hugging Face async adapter delegates synchronous model work to
    ``asyncio.to_thread``. This wrapper keeps that work in one owned,
    queue-free executor, then hands the populated query bundle back to the
    native vector retriever and its asynchronous vector-store client.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        embed_model: _AggregateEmbeddingModel,
    ) -> None:
        """Wrap one native vector retriever and its synchronous embed model."""
        super().__init__(callback_manager=retriever.callback_manager)
        self._retriever = retriever
        self._embed_model = embed_model
        self._cpu_work = AsyncWorkExecutor(name="docmind-vector-embedding")

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Delegate synchronous retrieval unchanged."""
        return self._retriever.retrieve(query_bundle)

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Populate the embedding before native asynchronous retrieval."""
        if query_bundle.embedding is None and query_bundle.embedding_strs:
            embed = self._embed_model.get_agg_embedding_from_queries
            embedding = await self._cpu_work.run(embed, query_bundle.embedding_strs)
            query_bundle = QueryBundle(
                query_str=query_bundle.query_str,
                embedding=embedding,
            )
        return await self._retriever.aretrieve(query_bundle)

    def close(self) -> None:
        """Reject new embedding work without blocking the caller."""
        self._cpu_work.close()

    async def aclose(self) -> None:
        """Wait for admitted embedding work to leave its native thread."""
        await self._cpu_work.aclose()


def _register_postprocessors(
    managed_resources: list[_ManagedRetriever],
    postprocessors: Sequence[object] | None,
) -> None:
    """Register postprocessors that expose the owned lifecycle contract."""
    for postprocessor in postprocessors or ():
        if callable(getattr(postprocessor, "close", None)) and callable(
            getattr(postprocessor, "aclose", None)
        ):
            managed_resources.append(cast(_ManagedRetriever, postprocessor))


class DocMindRouterQueryEngine(RouterQueryEngine):
    """Native router with explicit lifecycle for factory-owned retrievers."""

    def __init__(
        self,
        selector: BaseSelector,
        query_engine_tools: Sequence[QueryEngineTool],
        llm: LLM | None = None,
        summarizer: TreeSummarize | None = None,
        verbose: bool = False,
    ) -> None:
        """Create a router and initialize its lifecycle state.

        Args:
            selector: Tool selector used by the native router.
            query_engine_tools: Query tools available to the selector.
            llm: Optional router language model.
            summarizer: Optional multi-tool response summarizer.
            verbose: Whether the native router emits selection details.
        """
        super().__init__(
            selector,
            query_engine_tools,
            llm=llm,
            summarizer=summarizer,
            verbose=verbose,
        )
        self._managed_retrievers: tuple[_ManagedRetriever, ...] = ()
        self._owner_loop: asyncio.AbstractEventLoop | None = None
        self._lifecycle_lock = threading.Lock()
        self._closed = False

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Router query engine is closed")
        return super()._query(query_bundle)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        loop = asyncio.get_running_loop()
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("Router query engine is closed")
            if self._owner_loop is None:
                self._owner_loop = loop
            elif self._owner_loop is not loop:
                raise RuntimeError(
                    "Router query engine cannot move between event loops"
                )
        return await super()._aquery(query_bundle)

    def _take_retrievers(self) -> tuple[_ManagedRetriever, ...]:
        with self._lifecycle_lock:
            if self._closed:
                return ()
            self._closed = True
            retrievers = self._managed_retrievers
            self._managed_retrievers = ()
            return retrievers

    async def _close_retrievers(
        self,
        retrievers: tuple[_ManagedRetriever, ...],
    ) -> None:
        results = await asyncio.gather(
            *(retriever.aclose() for retriever in reversed(retrievers)),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, Exception):
                logger.debug(
                    "Router retriever cleanup failed (error_type={})",
                    type(result).__name__,
                )

    @staticmethod
    def _close_retrievers_sync(
        retrievers: tuple[_ManagedRetriever, ...],
    ) -> None:
        """Reject sync-only resources without waiting on native worker exit."""
        for retriever in reversed(retrievers):
            try:
                retriever.close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                logger.debug(
                    "Router retriever cleanup failed (error_type={})",
                    type(exc).__name__,
                )

    def close(self) -> None:
        """Close every owned retriever on the router's async owner loop."""
        retrievers = self._take_retrievers()
        if not retrievers:
            return

        owner_loop = self._owner_loop
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (owner_loop is None and current_loop is None) or (
            owner_loop is not None and not owner_loop.is_running()
        ):
            self._close_retrievers_sync(retrievers)
            return

        coroutine = self._close_retrievers(retrievers)
        if owner_loop is not None and owner_loop.is_running():
            if current_loop is owner_loop:
                owner_loop.create_task(coroutine)
                return
            try:
                close_future = asyncio.run_coroutine_threadsafe(coroutine, owner_loop)
            except RuntimeError as exc:  # pragma: no cover - loop shutdown race
                logger.debug(
                    "Router cleanup scheduling failed (error_type={})",
                    type(exc).__name__,
                )
            else:
                try:
                    close_future.result(timeout=_ROUTER_CLOSE_GRACE_S)
                except FuturesTimeoutError:
                    logger.warning("Router cleanup exceeded the close grace period")
                except Exception as exc:  # pragma: no cover - loop shutdown race
                    logger.debug(
                        "Router cleanup failed (error_type={})",
                        type(exc).__name__,
                    )
                return

        if current_loop is not None:
            current_loop.create_task(coroutine)
        else:
            asyncio.run(coroutine)

    async def aclose(self) -> None:
        """Close every owned retriever exactly once."""
        retrievers = self._take_retrievers()
        if not retrievers:
            return

        owner_loop = self._owner_loop
        current_loop = asyncio.get_running_loop()
        if (
            owner_loop is not None
            and owner_loop is not current_loop
            and not owner_loop.is_running()
        ):
            self._close_retrievers_sync(retrievers)
            return
        if (
            owner_loop is not None
            and owner_loop is not current_loop
            and owner_loop.is_running()
        ):
            coroutine = self._close_retrievers(retrievers)
            try:
                close_future = asyncio.run_coroutine_threadsafe(
                    coroutine,
                    owner_loop,
                )
            except RuntimeError:  # pragma: no cover - loop shutdown race
                coroutine.close()
            else:
                await asyncio.wrap_future(close_future)
                return
        await self._close_retrievers(retrievers)


def _build_vector_tool(
    *,
    vector_index: BaseIndex,
    managed_retrievers: list[_ManagedRetriever],
    settings: DocMindSettings,
) -> QueryEngineTool:
    """Build the mandatory semantic vector search tool."""
    vector_post = get_postprocessors(
        "vector",
        use_reranking=settings.retrieval.use_reranking,
        top_n=settings.retrieval.reranking_top_k,
    )
    _register_postprocessors(managed_retrievers, vector_post)
    vector_engine = vector_index.as_query_engine(
        similarity_top_k=settings.retrieval.top_k,
        node_postprocessors=vector_post,
        response_mode=ResponseMode.NO_TEXT,
    )
    if isinstance(vector_engine, RetrieverQueryEngine):
        native_retriever = vector_engine.retriever
        embed_model = getattr(native_retriever, "_embed_model", None)
        if embed_model is not None and callable(
            getattr(embed_model, "get_agg_embedding_from_queries", None)
        ):
            owned_retriever = _OwnedVectorEmbeddingRetriever(
                native_retriever,
                embed_model,
            )
            managed_retrievers.append(owned_retriever)
            vector_engine = vector_engine.with_retriever(owned_retriever)

    return QueryEngineTool(
        query_engine=vector_engine,
        metadata=ToolMetadata(
            name="semantic_search",
            description=(
                "Semantic vector search over the document corpus. Best for general "
                "questions and content lookup."
            ),
        ),
    )


def _maybe_add_multimodal_tool(
    *,
    tools: list[QueryEngineTool],
    managed_retrievers: list[_ManagedRetriever],
    vector_index: BaseIndex,
    text_collection: str,
    image_collection: str,
    settings: DocMindSettings,
    llm: LLM | None,
) -> None:
    """Add multimodal search tool when enabled and available."""
    if not settings.retrieval.enable_image_retrieval:
        return
    try:
        from src.retrieval.multimodal_fusion import MultimodalFusionRetriever

        text_retriever: BaseRetriever | None = None
        if not sparse_retrieval_enabled(settings):
            native_retriever = vector_index.as_retriever(
                similarity_top_k=settings.retrieval.top_k
            )
            embed_model = getattr(native_retriever, "_embed_model", None)
            if embed_model is not None and callable(
                getattr(embed_model, "get_agg_embedding_from_queries", None)
            ):
                text_retriever = _OwnedVectorEmbeddingRetriever(
                    native_retriever,
                    embed_model,
                )
                managed_retrievers.append(text_retriever)
            else:
                text_retriever = native_retriever
        mm_retriever = MultimodalFusionRetriever(
            text_retriever=text_retriever,
            text_collection=text_collection,
            image_collection=image_collection,
        )
        managed_retrievers.append(mm_retriever)
        mm_post = get_postprocessors(
            "hybrid",
            use_reranking=settings.retrieval.use_reranking,
            top_n=settings.retrieval.reranking_top_k,
        )
        _register_postprocessors(managed_retrievers, mm_post)
        mm_engine = RetrieverQueryEngine.from_args(
            retriever=mm_retriever,
            node_postprocessors=mm_post,
            llm=llm,
            response_mode=ResponseMode.NO_TEXT,
            verbose=False,
        )
        tools.append(
            QueryEngineTool(
                query_engine=mm_engine,
                metadata=ToolMetadata(
                    name="multimodal_search",
                    description=(
                        "Multimodal search: fuses text "
                        "hybrid retrieval with visual PDF page-image "
                        "retrieval (SigLIP). Best for visually rich PDFs "
                        "(tables/charts/scans) and when user asks "
                        '"what does that chart show?".'
                    ),
                ),
            )
        )
    except (
        ImageCollectionIncompatibleError,
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
    ) as exc:
        _warn_once(
            "multimodal",
            "Multimodal tool construction skipped",
            reason=str(exc),
        )


def _maybe_add_hybrid_tool(
    *,
    tools: list[QueryEngineTool],
    managed_retrievers: list[_ManagedRetriever],
    text_collection: str,
    settings: DocMindSettings,
    llm: LLM | None,
) -> None:
    """Add hybrid search tool when enabled."""
    try:
        from src.retrieval.hybrid import HybridParams, ServerHybridRetriever

        params = HybridParams(
            collection=text_collection,
            fused_top_k=settings.retrieval.fused_top_k,
            prefetch_sparse=settings.retrieval.prefetch_sparse_limit,
            prefetch_dense=settings.retrieval.prefetch_dense_limit,
            fusion_mode=settings.retrieval.fusion_mode,
            rrf_k=settings.retrieval.rrf_k,
            dedup_key=settings.retrieval.dedup_key,
        )
        retriever = ServerHybridRetriever(params)
        managed_retrievers.append(retriever)
        hybrid_post = get_postprocessors(
            "hybrid",
            use_reranking=settings.retrieval.use_reranking,
            top_n=settings.retrieval.reranking_top_k,
        )
        _register_postprocessors(managed_retrievers, hybrid_post)
        hybrid_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=hybrid_post,
            llm=llm,
            response_mode=ResponseMode.NO_TEXT,
            verbose=False,
        )
        tools.append(
            QueryEngineTool(
                query_engine=hybrid_engine,
                metadata=ToolMetadata(
                    name="hybrid_search",
                    description=(
                        "Hybrid search via Qdrant Query API (dense+sparse fusion)"
                    ),
                ),
            )
        )
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        _warn_once("hybrid", "Hybrid tool construction skipped", reason=str(exc))


def _maybe_add_keyword_tool(
    *,
    tools: list[QueryEngineTool],
    managed_retrievers: list[_ManagedRetriever],
    text_collection: str,
    settings: DocMindSettings,
    llm: LLM | None,
) -> None:
    """Add sparse-only keyword search when enabled."""
    if not settings.retrieval.enable_keyword_tool:
        return

    try:
        from src.retrieval.keyword import KeywordParams, KeywordSparseRetriever

        params = KeywordParams(
            collection=text_collection,
            top_k=settings.retrieval.top_k,
            using="text-sparse",
        )
        retriever = KeywordSparseRetriever(params)
        managed_retrievers.append(retriever)
        keyword_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=None,
            llm=llm,
            response_mode=ResponseMode.NO_TEXT,
            verbose=False,
        )
        tools.append(
            QueryEngineTool(
                query_engine=keyword_engine,
                metadata=ToolMetadata(
                    name="keyword_search",
                    description=(
                        "Exact keyword, identifier, and error-code lookup via "
                        "sparse matching."
                    ),
                ),
            )
        )
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        _warn_once("keyword", "Keyword tool construction skipped", reason=str(exc))


def _maybe_add_graph_tool(
    *,
    tools: list[QueryEngineTool],
    managed_retrievers: list[_ManagedRetriever],
    settings: DocMindSettings,
    pg_index: PropertyGraphIndex | None,
    llm: LLM | None,
) -> None:
    """Add the GraphRAG tool when a healthy index is available."""
    if pg_index is None or pg_index.property_graph_store is None:
        return
    try:
        graph_post = get_postprocessors(
            "kg",
            use_reranking=settings.retrieval.use_reranking,
            top_n=settings.retrieval.reranking_top_k,
        )
        _register_postprocessors(managed_retrievers, graph_post)
        artifacts = build_graph_query_engine(
            pg_index,
            llm=llm,
            include_text=True,
            path_depth=settings.graphrag_cfg.default_path_depth,
            similarity_top_k=settings.retrieval.top_k,
            node_postprocessors=graph_post,
        )
        tools.append(
            QueryEngineTool(
                query_engine=cast(BaseQueryEngine, artifacts.query_engine),
                metadata=ToolMetadata(
                    name="knowledge_graph",
                    description=(
                        "Knowledge graph traversal for relationship-centric queries"
                    ),
                ),
            )
        )
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="router_factory.graph_tool")
        logger.debug(
            "Graph tool construction skipped (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )


def _warn_once(key: str, message: str, *, reason: str) -> None:
    """Emit a warning once, downgrade to debug on subsequent occurrences."""
    reason_redacted = build_pii_log_entry(
        str(reason), key_id=f"router_factory:{key}:reason"
    ).redacted
    if key not in _WARNED_TOOLS:
        logger.warning("{} (reason={})", message, reason_redacted)
        _WARNED_TOOLS.add(key)
    else:
        logger.debug("{} (reason={})", message, reason_redacted)


def build_router_engine(
    vector_index: BaseIndex,
    pg_index: PropertyGraphIndex | None = None,
    settings: DocMindSettings = default_settings,
    llm: LLM | None = None,
    *,
    text_collection: str | None = None,
    image_collection: str | None = None,
) -> DocMindRouterQueryEngine:
    """Create a router engine with vector and optional KG/hybrid tools."""
    managed_retrievers: list[_ManagedRetriever] = []
    active_text_collection = text_collection or settings.database.qdrant_collection
    active_image_collection = (
        image_collection or settings.database.qdrant_image_collection
    )
    tools = [
        _build_vector_tool(
            vector_index=vector_index,
            managed_retrievers=managed_retrievers,
            settings=settings,
        )
    ]

    hybrid_requested = settings.retrieval.enable_server_hybrid
    kg_requested = pg_index is not None

    with router_build_span(
        adapter_name="llama_index",
        kg_requested=kg_requested,
        hybrid_requested=hybrid_requested,
    ) as span:
        _maybe_add_multimodal_tool(
            tools=tools,
            managed_retrievers=managed_retrievers,
            vector_index=vector_index,
            text_collection=active_text_collection,
            image_collection=active_image_collection,
            settings=settings,
            llm=llm,
        )

        if hybrid_requested:
            _maybe_add_hybrid_tool(
                tools=tools,
                managed_retrievers=managed_retrievers,
                text_collection=active_text_collection,
                settings=settings,
                llm=llm,
            )

        _maybe_add_keyword_tool(
            tools=tools,
            managed_retrievers=managed_retrievers,
            text_collection=active_text_collection,
            settings=settings,
            llm=llm,
        )

        _maybe_add_graph_tool(
            tools=tools,
            managed_retrievers=managed_retrievers,
            settings=settings,
            pg_index=pg_index,
            llm=llm,
        )

        try:
            router = cast(
                DocMindRouterQueryEngine,
                DocMindRouterQueryEngine.from_defaults(
                    query_engine_tools=tools,
                    llm=llm,
                    verbose=False,
                ),
            )
        except Exception:
            for retriever in reversed(managed_retrievers):
                retriever.close()
            raise
        router._managed_retrievers = tuple(managed_retrievers)

        tool_names = [
            tool.metadata.name for tool in tools if tool.metadata.name is not None
        ]
        kg_present = "knowledge_graph" in tool_names
        record_router_selection(span, tool_names=tool_names, kg_enabled=kg_present)

    logger.info("Router engine built (kg_present={})", kg_present)
    return router


__all__ = [
    "DocMindRouterQueryEngine",
    "build_router_engine",
]
