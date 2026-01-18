"""Router factory orchestrating vector, hybrid, and GraphRAG tools.

Builds a RouterQueryEngine with:
- semantic_search (vector)
- optional hybrid_search (server-side hybrid retriever)
- optional knowledge_graph (GraphRAG)

This module keeps imports dependency-light by routing all llama_index access via
``src.retrieval.llama_index_adapter`` and GraphRAG access via the adapter
registry-backed helpers in ``src.retrieval.graph_config``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import settings as default_settings
from src.retrieval.adapter_registry import (
    GRAPH_DEPENDENCY_HINT,
    MissingGraphAdapterError,
)
from src.retrieval.graph_config import build_graph_query_engine
from src.retrieval.llama_index_adapter import (
    LlamaIndexAdapterProtocol,
    MissingLlamaIndexError,
    get_llama_index_adapter,
)
from src.retrieval.postprocessor_utils import (
    build_retriever_query_engine,
    build_vector_query_engine,
)
from src.telemetry.opentelemetry import record_router_selection, router_build_span
from src.utils.log_safety import build_pii_log_entry

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from llama_index.core.query_engine import RouterQueryEngine as RouterQueryEngineType
else:
    RouterQueryEngineType = Any

_WARNING_FLAGS: dict[str, bool] = {"hybrid": False, "graph": False, "multimodal": False}


def _reset_warnings() -> None:
    """Reset warning flags for test isolation."""
    _WARNING_FLAGS.clear()
    _WARNING_FLAGS.update({"hybrid": False, "graph": False, "multimodal": False})


def _coerce_top_k(value: Any) -> int | None:
    """Coerce a value to int when possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None


def _safe_get_postprocessors() -> Any | None:
    """Safely import reranking postprocessor factory."""
    try:
        from src.retrieval.reranking import get_postprocessors

        return get_postprocessors
    except ImportError:
        return None


def _build_vector_tool(
    *,
    vector_index: Any,
    cfg: Any,
    get_pp: Any | None,
    use_rerank_flag: bool,
    normalized_top_k: int | None,
    query_engine_tool_cls: Any,
    tool_metadata_cls: Any,
) -> Any:
    """Build the mandatory semantic vector search tool."""
    try:
        vector_post = (
            get_pp("vector", use_reranking=use_rerank_flag, top_n=normalized_top_k)
            if get_pp is not None
            else None
        )
        vector_engine = build_vector_query_engine(
            vector_index,
            vector_post,
            similarity_top_k=int(getattr(cfg.retrieval, "top_k", 10)),
        )
    except (TypeError, AttributeError, ValueError) as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="router_factory.vector_engine")
        logger.warning(
            "Vector query engine fallback: using default configuration "
            "(error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )
        vector_engine = vector_index.as_query_engine()

    return query_engine_tool_cls(
        query_engine=vector_engine,
        metadata=tool_metadata_cls(
            name="semantic_search",
            description=(
                "Semantic vector search over the document corpus. Best for general "
                "questions and content lookup."
            ),
        ),
    )


def _maybe_add_multimodal_tool(
    *,
    tools: list[Any],
    cfg: Any,
    get_pp: Any | None,
    use_rerank_flag: bool,
    normalized_top_k: int | None,
    retriever_query_engine_cls: Any,
    query_engine_tool_cls: Any,
    tool_metadata_cls: Any,
    the_llm: Any,
) -> None:
    """Add multimodal search tool when enabled and available."""
    try:
        retrieval_cfg = getattr(cfg, "retrieval", cfg)
        enable_mm = bool(getattr(retrieval_cfg, "enable_image_retrieval", False))
    except Exception:  # pragma: no cover - defensive
        enable_mm = False

    if not enable_mm:
        return
    try:
        from src.retrieval.multimodal_fusion import MultimodalFusionRetriever

        mm_retriever = MultimodalFusionRetriever()
        mm_post = (
            get_pp("hybrid", use_reranking=use_rerank_flag, top_n=normalized_top_k)
            if get_pp is not None
            else None
        )
        mm_engine = build_retriever_query_engine(
            mm_retriever,
            mm_post,
            llm=the_llm,
            response_mode="compact",
            verbose=False,
            engine_cls=retriever_query_engine_cls,
        )
        tools.append(
            query_engine_tool_cls(
                query_engine=mm_engine,
                metadata=tool_metadata_cls(
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
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        _warn_once(
            "multimodal",
            "Multimodal tool construction skipped",
            reason=str(exc),
        )


def _maybe_add_hybrid_tool(
    *,
    tools: list[Any],
    cfg: Any,
    get_pp: Any | None,
    use_rerank_flag: bool,
    normalized_top_k: int | None,
    retriever_query_engine_cls: Any,
    query_engine_tool_cls: Any,
    tool_metadata_cls: Any,
    the_llm: Any,
) -> None:
    """Add hybrid search tool when enabled."""
    try:
        from src.retrieval.hybrid import HybridParams, ServerHybridRetriever

        params = HybridParams(
            collection=cfg.database.qdrant_collection,
            fused_top_k=int(getattr(cfg.retrieval, "fused_top_k", 60)),
            prefetch_sparse=int(getattr(cfg.retrieval, "prefetch_sparse_limit", 400)),
            prefetch_dense=int(getattr(cfg.retrieval, "prefetch_dense_limit", 200)),
            fusion_mode=str(getattr(cfg.retrieval, "fusion_mode", "rrf")),
            dedup_key=str(getattr(cfg.retrieval, "dedup_key", "page_id")),
        )
        retriever = ServerHybridRetriever(params)
        hybrid_post = (
            get_pp("hybrid", use_reranking=use_rerank_flag, top_n=normalized_top_k)
            if get_pp is not None
            else None
        )
        hybrid_engine = build_retriever_query_engine(
            retriever,
            hybrid_post,
            llm=the_llm,
            response_mode="compact",
            verbose=False,
            engine_cls=retriever_query_engine_cls,
        )
        tools.append(
            query_engine_tool_cls(
                query_engine=hybrid_engine,
                metadata=tool_metadata_cls(
                    name="hybrid_search",
                    description=(
                        "Hybrid search via Qdrant Query API (dense+sparse fusion)"
                    ),
                ),
            )
        )
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        _warn_once("hybrid", "Hybrid tool construction skipped", reason=str(exc))


def _maybe_add_graph_tool(
    *,
    tools: list[Any],
    cfg: Any,
    pg_index: Any | None,
    get_pp: Any | None,
    use_rerank_flag: bool,
    normalized_top_k: int | None,
    query_engine_tool_cls: Any,
    tool_metadata_cls: Any,
    the_llm: Any,
) -> None:
    """Add GraphRAG tool when enabled and available."""
    if pg_index is None or getattr(pg_index, "property_graph_store", None) is None:
        return

    graphrag_enabled = (
        cfg.is_graphrag_enabled()
        if hasattr(cfg, "is_graphrag_enabled")
        else getattr(cfg, "enable_graphrag", False)
    )
    if not graphrag_enabled:
        return
    try:
        graph_depth = int(
            getattr(getattr(cfg, "graphrag_cfg", cfg), "default_path_depth", 1)
        )
        graph_top_k = int(getattr(cfg.retrieval, "top_k", 10))
        graph_post = (
            get_pp("kg", use_reranking=use_rerank_flag, top_n=normalized_top_k)
            if get_pp is not None
            else None
        )
        try:
            artifacts = build_graph_query_engine(
                pg_index,
                llm=the_llm,
                include_text=True,
                path_depth=graph_depth,
                similarity_top_k=graph_top_k,
                node_postprocessors=graph_post,
            )
        except TypeError:
            logger.debug(
                "Graph query engine does not support node_postprocessors; "
                "retrying without"
            )
            artifacts = build_graph_query_engine(
                pg_index,
                llm=the_llm,
                include_text=True,
                path_depth=graph_depth,
                similarity_top_k=graph_top_k,
                node_postprocessors=None,
            )
        tools.append(
            query_engine_tool_cls(
                query_engine=artifacts.query_engine,
                metadata=tool_metadata_cls(
                    name="knowledge_graph",
                    description=(
                        "Knowledge graph traversal for relationship-centric queries"
                    ),
                ),
            )
        )
    except MissingGraphAdapterError as exc:
        _warn_once("graph", GRAPH_DEPENDENCY_HINT, reason=str(exc))
    except (ValueError, TypeError, AttributeError, ImportError) as exc:
        from src.utils.log_safety import build_pii_log_entry

        redaction = build_pii_log_entry(str(exc), key_id="router_factory.graph_tool")
        logger.debug(
            "Graph tool construction skipped (error_type={} error={})",
            type(exc).__name__,
            redaction.redacted,
        )


def _select_router(
    *,
    adapter: LlamaIndexAdapterProtocol,
    tools: list[Any],
    the_llm: Any,
) -> Any:
    """Instantiate a router query engine for the assembled tools."""
    selector = adapter.get_pydantic_selector(the_llm)
    if selector is None:
        try:
            selector = (
                adapter.LLMSingleSelector.from_defaults(llm=the_llm)
                if the_llm is not None
                else adapter.LLMSingleSelector.from_defaults()
            )
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                "Failed to build router selector via "
                "LLMSingleSelector.from_defaults(llm=the_llm)."
            ) from exc

    try:
        return adapter.RouterQueryEngine(
            selector=selector,
            query_engine_tools=tools,
            verbose=False,
            llm=the_llm,
        )
    except TypeError:
        return adapter.RouterQueryEngine(
            selector=selector,
            query_engine_tools=tools,
            verbose=False,
        )


def _warn_once(key: str, message: str, *, reason: str) -> None:
    """Emit a warning once, downgrade to debug on subsequent occurrences."""
    reason_redacted = build_pii_log_entry(
        str(reason), key_id=f"router_factory:{key}:reason"
    ).redacted
    if not _WARNING_FLAGS.get(key):
        logger.warning("{} (reason={})", message, reason_redacted)
        _WARNING_FLAGS[key] = True
    else:
        logger.debug("{} (reason={})", message, reason_redacted)


def _resolve_adapter(
    adapter: LlamaIndexAdapterProtocol | None,
) -> LlamaIndexAdapterProtocol:
    """Return an adapter, raising a helpful error when llama_index is absent."""
    if adapter is not None:
        return adapter
    try:
        return get_llama_index_adapter()
    except MissingLlamaIndexError as exc:
        raise MissingLlamaIndexError() from exc


def build_router_engine(
    vector_index: Any,
    pg_index: Any | None = None,
    settings: Any | None = None,
    llm: Any | None = None,
    *,
    enable_hybrid: bool | None = None,
    adapter: LlamaIndexAdapterProtocol | None = None,
) -> RouterQueryEngineType:
    """Create a router engine with vector and optional KG/hybrid tools."""
    cfg = settings or default_settings
    adapter = _resolve_adapter(adapter)

    raw_top_k = getattr(getattr(cfg, "retrieval", cfg), "reranking_top_k", None)
    normalized_top_k = _coerce_top_k(raw_top_k)

    retriever_query_engine_cls = adapter.RetrieverQueryEngine
    query_engine_tool_cls = adapter.QueryEngineTool
    tool_metadata_cls = adapter.ToolMetadata

    try:
        use_rerank_flag = bool(getattr(cfg.retrieval, "use_reranking", True))
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
        use_rerank_flag = True

    the_llm = llm
    if the_llm is None:
        try:
            from src.config.llm_factory import build_llm

            the_llm = build_llm(cfg)
        except ImportError:
            logger.debug("LLM factory import failed; proceeding without LLM")
        except Exception as exc:  # pragma: no cover - best effort
            from src.utils.log_safety import build_pii_log_entry

            redaction = build_pii_log_entry(str(exc), key_id="router_factory.build_llm")
            logger.warning(
                "LLM factory failed; proceeding without LLM (error_type={} error={})",
                type(exc).__name__,
                redaction.redacted,
            )

    get_pp = _safe_get_postprocessors()
    tools: list[Any] = [
        _build_vector_tool(
            vector_index=vector_index,
            cfg=cfg,
            get_pp=get_pp,
            use_rerank_flag=use_rerank_flag,
            normalized_top_k=normalized_top_k,
            query_engine_tool_cls=query_engine_tool_cls,
            tool_metadata_cls=tool_metadata_cls,
        )
    ]

    hybrid_requested = (
        bool(enable_hybrid)
        if enable_hybrid is not None
        else bool(getattr(cfg.retrieval, "enable_server_hybrid", False))
    )

    if hasattr(cfg, "is_graphrag_enabled"):
        kg_requested = bool(pg_index is not None and cfg.is_graphrag_enabled())
    else:
        kg_requested = bool(
            pg_index is not None and getattr(cfg, "enable_graphrag", False)
        )

    with router_build_span(
        adapter_name=getattr(adapter, "__class__", type("A", (), {})).__name__,
        kg_requested=kg_requested,
        hybrid_requested=hybrid_requested,
    ) as span:
        _maybe_add_multimodal_tool(
            tools=tools,
            cfg=cfg,
            get_pp=get_pp,
            use_rerank_flag=use_rerank_flag,
            normalized_top_k=normalized_top_k,
            retriever_query_engine_cls=retriever_query_engine_cls,
            query_engine_tool_cls=query_engine_tool_cls,
            tool_metadata_cls=tool_metadata_cls,
            the_llm=the_llm,
        )

        if hybrid_requested:
            _maybe_add_hybrid_tool(
                tools=tools,
                cfg=cfg,
                get_pp=get_pp,
                use_rerank_flag=use_rerank_flag,
                normalized_top_k=normalized_top_k,
                retriever_query_engine_cls=retriever_query_engine_cls,
                query_engine_tool_cls=query_engine_tool_cls,
                tool_metadata_cls=tool_metadata_cls,
                the_llm=the_llm,
            )

        _maybe_add_graph_tool(
            tools=tools,
            cfg=cfg,
            pg_index=pg_index,
            get_pp=get_pp,
            use_rerank_flag=use_rerank_flag,
            normalized_top_k=normalized_top_k,
            query_engine_tool_cls=query_engine_tool_cls,
            tool_metadata_cls=tool_metadata_cls,
            the_llm=the_llm,
        )

        router = _select_router(adapter=adapter, tools=tools, the_llm=the_llm)

        tool_names: list[str] = []
        for tool in tools:
            metadata = getattr(tool, "metadata", None)
            name = getattr(metadata, "name", None)
            if name:
                tool_names.append(str(name))
        kg_present = "knowledge_graph" in tool_names
        record_router_selection(span, tool_names=tool_names, kg_enabled=kg_present)

    try:
        router.query_engine_tools = tools
    except (AttributeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.debug(
            "Router engine lacks public tool attribute (error_type={})",
            type(exc).__name__,
        )

    logger.info("Router engine built (kg_present={})", kg_present)
    return router


__all__ = [
    "GRAPH_DEPENDENCY_HINT",
    "build_router_engine",
]
