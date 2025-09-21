"""Router factory orchestrating vector, hybrid, and GraphRAG tools."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.config.settings import settings as default_settings
from src.retrieval.adapter_registry import (
    GRAPH_DEPENDENCY_HINT,
    MissingGraphAdapterError,
    ensure_default_adapter,
    get_adapter,
)
from src.retrieval.adapters.protocols import (
    AdapterFactoryProtocol,
    GraphQueryArtifacts,
)
from src.retrieval.graph_config import build_graph_query_engine
from src.retrieval.postprocessor_utils import (
    build_retriever_query_engine,
    build_vector_query_engine,
)
from src.telemetry.opentelemetry import (
    record_router_selection,
    router_build_span,
)

_WARNING_FLAGS: dict[str, bool] = {"hybrid": False, "graph": False}


def _warn_once(key: str, message: str, *, reason: str) -> None:
    """Emit a warning once, downgrade to debug on subsequent occurrences."""
    if not _WARNING_FLAGS.get(key, False):
        logger.warning("%s (reason=%s)", message, reason)
        _WARNING_FLAGS[key] = True
    else:
        logger.debug("%s (reason=%s)", message, reason)


@dataclass(frozen=True)
class _RouterComponents:
    router_query_engine_cls: Any
    retriever_query_engine_cls: Any | None
    query_engine_tool_cls: Any
    tool_metadata_cls: Any
    llm_single_selector_cls: Any
    get_pydantic_selector: Callable[[Any | None], Any | None]
    available: bool


def _load_router_components() -> _RouterComponents:
    try:
        query_engine = importlib.import_module("llama_index.core.query_engine")
        selectors = importlib.import_module("llama_index.core.selectors")
        tools = importlib.import_module("llama_index.core.tools")
    except ImportError:  # pragma: no cover - optional dependency path
        logger.warning(
            "llama-index is not installed; falling back to vector-only routing"
        )

        class _ToolMetadata:
            def __init__(self, name: str, description: str) -> None:
                self.name = name
                self.description = description

        class _QueryEngineTool:
            def __init__(self, query_engine: Any, metadata: _ToolMetadata) -> None:
                self.query_engine = query_engine
                self.metadata = metadata

        class _RouterQueryEngine:
            def __init__(
                self,
                *,
                selector: Any,
                query_engine_tools: Sequence[Any],
                verbose: bool = False,
                llm: Any | None = None,
            ) -> None:
                self.selector = selector
                self.query_engine_tools = list(query_engine_tools)
                self.verbose = verbose
                self.llm = llm

            @classmethod
            def from_args(cls, **kwargs: Any) -> _RouterQueryEngine:
                """Construct a fallback router query engine."""
                return cls(**kwargs)

        class _LLMSingleSelector:
            def __init__(self, llm: Any | None = None) -> None:
                self.llm = llm

            @classmethod
            def from_defaults(cls, llm: Any | None = None) -> _LLMSingleSelector:
                """Return a fallback selector storing the provided LLM."""
                return cls(llm=llm)

        def _no_pydantic_selector(_llm: Any | None) -> None:
            return None

        return _RouterComponents(
            router_query_engine_cls=_RouterQueryEngine,
            retriever_query_engine_cls=None,
            query_engine_tool_cls=_QueryEngineTool,
            tool_metadata_cls=_ToolMetadata,
            llm_single_selector_cls=_LLMSingleSelector,
            get_pydantic_selector=_no_pydantic_selector,
            available=False,
        )

    def _maybe_pydantic_selector(llm: Any | None) -> Any | None:
        factory = getattr(selectors, "PydanticSingleSelector", None)
        if factory is None:
            return None
        metadata_obj = getattr(llm, "metadata", None)
        if not getattr(metadata_obj, "is_function_calling_model", False):
            return None
        try:
            return factory.from_defaults(llm=llm)
        except (ImportError, ValueError):  # pragma: no cover - defensive
            return None

    return _RouterComponents(
        router_query_engine_cls=query_engine.RouterQueryEngine,
        retriever_query_engine_cls=query_engine.RetrieverQueryEngine,
        query_engine_tool_cls=tools.QueryEngineTool,
        tool_metadata_cls=tools.ToolMetadata,
        llm_single_selector_cls=selectors.LLMSingleSelector,
        get_pydantic_selector=_maybe_pydantic_selector,
        available=True,
    )


def _resolve_adapter(adapter: AdapterFactoryProtocol | None) -> AdapterFactoryProtocol:
    if adapter is not None:
        return adapter
    ensure_default_adapter()
    return get_adapter()


def build_router_engine(
    vector_index: Any,
    pg_index: Any | None = None,
    settings: Any | None = None,
    llm: Any | None = None,
    *,
    enable_hybrid: bool | None = None,
    adapter: AdapterFactoryProtocol | None = None,
) -> Any:
    """Construct a router combining vector, hybrid, and GraphRAG tools."""
    cfg = settings or default_settings
    components = _load_router_components()
    adapter = _resolve_adapter(adapter) if adapter is not None else None
    adapter_available = adapter is not None

    if adapter is None:
        try:
            adapter = _resolve_adapter(None)
            adapter_available = True
        except MissingGraphAdapterError:
            adapter_available = False

    # Build vector search tool
    query_engine_tool_cls = components.query_engine_tool_cls
    tool_metadata_cls = components.tool_metadata_cls

    raw_top_k = getattr(getattr(cfg, "retrieval", cfg), "reranking_top_k", None)

    def _coerce_top_k(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    normalized_top_k = _coerce_top_k(raw_top_k)
    try:
        from src.retrieval.reranking import get_postprocessors as _get_pp

        use_rerank_flag = bool(getattr(cfg.retrieval, "use_reranking", True))
        vector_post = _get_pp(
            "vector", use_reranking=use_rerank_flag, top_n=normalized_top_k
        )
    except Exception:  # pragma: no cover - reranker optional
        use_rerank_flag = True
        vector_post = None

    try:
        top_k = int(getattr(cfg.retrieval, "top_k", 10))
    except Exception:
        top_k = 10

    vector_engine = build_vector_query_engine(
        vector_index,
        vector_post,
        similarity_top_k=top_k,
    )

    vector_tool = query_engine_tool_cls(
        query_engine=vector_engine,
        metadata=tool_metadata_cls(
            name="semantic_search",
            description=(
                "Semantic vector search over the document corpus. Best for general "
                "questions and content lookup."
            ),
        ),
    )
    tools = [vector_tool]

    hybrid_requested = (
        bool(enable_hybrid)
        if enable_hybrid is not None
        else bool(getattr(cfg.retrieval, "enable_server_hybrid", False))
    )
    hybrid_tool_added = False

    if components.retriever_query_engine_cls is not None:
        try:
            if hybrid_requested:
                from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams
                from src.retrieval.reranking import get_postprocessors as _get_pp

                params = _HybridParams(
                    collection=cfg.database.qdrant_collection,
                    fused_top_k=int(getattr(cfg.retrieval, "fused_top_k", 60)),
                    prefetch_sparse=int(
                        getattr(cfg.retrieval, "prefetch_sparse_limit", 400)
                    ),
                    prefetch_dense=int(
                        getattr(cfg.retrieval, "prefetch_dense_limit", 200)
                    ),
                    fusion_mode=str(getattr(cfg.retrieval, "fusion_mode", "rrf")),
                    dedup_key=str(getattr(cfg.retrieval, "dedup_key", "page_id")),
                )
                retriever = ServerHybridRetriever(params)
                hybrid_post = _get_pp(
                    "hybrid", use_reranking=use_rerank_flag, top_n=normalized_top_k
                )
                hybrid_engine = build_retriever_query_engine(
                    retriever,
                    hybrid_post,
                    llm=llm,
                    engine_cls=components.retriever_query_engine_cls,
                    response_mode="compact",
                    verbose=False,
                )
                tools.append(
                    query_engine_tool_cls(
                        query_engine=hybrid_engine,
                        metadata=tool_metadata_cls(
                            name="hybrid_search",
                            description=(
                                "Hybrid search via Qdrant Query API "
                                "(dense+sparse fusion)"
                            ),
                        ),
                    )
                )
                hybrid_tool_added = True
        except Exception as exc:  # pragma: no cover - hybrid optional
            _warn_once(
                "hybrid",
                "Hybrid retrieval disabled; continuing without fused search",
                reason=str(exc),
            )

    graph_requested = (
        adapter_available
        and pg_index is not None
        and getattr(pg_index, "property_graph_store", None) is not None
        and bool(getattr(cfg, "enable_graphrag", True))
    )

    if (
        pg_index is not None
        and getattr(pg_index, "property_graph_store", None) is not None
        and not adapter_available
    ):
        _warn_once(
            "graph",
            "Knowledge graph retrieval disabled due to missing adapter",
            reason=GRAPH_DEPENDENCY_HINT,
        )

    adapter_name = getattr(adapter, "name", "unavailable") if adapter else "unavailable"
    graph_tool_added = False

    with router_build_span(
        adapter_name=adapter_name,
        kg_requested=graph_requested,
        hybrid_requested=hybrid_requested,
    ) as router_span:
        if graph_requested:
            try:
                graph_depth = int(
                    getattr(getattr(cfg, "graphrag_cfg", cfg), "default_path_depth", 1)
                )
            except Exception:
                graph_depth = 1
            try:
                graph_top_k = int(getattr(cfg.retrieval, "top_k", 10))
            except Exception:
                graph_top_k = 10
            try:
                from src.retrieval.reranking import get_postprocessors as _get_pp

                graph_post = _get_pp(
                    "kg", use_reranking=use_rerank_flag, top_n=normalized_top_k
                )
            except Exception:  # pragma: no cover - reranker optional
                graph_post = None

            try:
                artifacts: GraphQueryArtifacts = build_graph_query_engine(
                    pg_index,
                    llm=llm,
                    include_text=True,
                    path_depth=graph_depth,
                    similarity_top_k=graph_top_k,
                    node_postprocessors=graph_post,
                    adapter=adapter,
                )
                tools.append(
                    query_engine_tool_cls(
                        query_engine=artifacts.query_engine,
                        metadata=tool_metadata_cls(
                            name="knowledge_graph",
                            description=(
                                "Knowledge graph traversal for "
                                "relationship-centric queries"
                            ),
                        ),
                    )
                )
                graph_tool_added = True
            except (MissingGraphAdapterError, ValueError) as exc:
                _warn_once(
                    "graph",
                    "Knowledge graph tool unavailable; falling back to vector/hybrid",
                    reason=str(exc),
                )

        router_query_engine_cls = components.router_query_engine_cls
        llm_single_selector_cls = components.llm_single_selector_cls
        get_pydantic_selector = components.get_pydantic_selector

        class _NoOpLLM:
            def __init__(self) -> None:
                self.metadata = type(
                    "_MD",
                    (),
                    {
                        "context_window": 2048,
                        "num_output": 256,
                        "is_function_calling_model": False,
                        "model_name": "docmind-fallback",
                    },
                )()

            def predict(self, *_args: Any, **_kwargs: Any) -> str:
                return ""

            def complete(self, *_args: Any, **_kwargs: Any) -> str:
                return ""

        def _supports_function_calling(candidate: Any | None) -> bool:
            metadata_obj = getattr(candidate, "metadata", None)
            return bool(getattr(metadata_obj, "is_function_calling_model", False))

        selector = None
        if llm is not None and _supports_function_calling(llm):
            selector = get_pydantic_selector(llm)
        if selector is None:
            selector = llm_single_selector_cls.from_defaults(llm=llm or _NoOpLLM())

        router_kwargs = {
            "selector": selector,
            "query_engine_tools": tools,
            "verbose": False,
            "llm": llm or _NoOpLLM(),
        }

        try:
            router = router_query_engine_cls(**router_kwargs)
        except TypeError:
            router_kwargs.pop("llm", None)
            router = router_query_engine_cls(**router_kwargs)

        with suppress(AttributeError, TypeError):
            router.query_engine_tools = tools
        with suppress(AttributeError, TypeError):
            router._query_engine_tools = tools

        tool_names = [getattr(tool.metadata, "name", "") for tool in tools]
        router_span.set_attribute("router.hybrid_enabled", bool(hybrid_tool_added))
        record_router_selection(
            router_span,
            tool_names=tool_names,
            kg_enabled=graph_tool_added,
        )

    logger.info("Router engine built (kg_present=%s)", graph_tool_added)
    return router


__all__ = ["GRAPH_DEPENDENCY_HINT", "build_router_engine"]
