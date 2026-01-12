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

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from llama_index.core.query_engine import RouterQueryEngine as RouterQueryEngineType
else:
    RouterQueryEngineType = Any

_WARNING_FLAGS: dict[str, bool] = {"hybrid": False, "graph": False, "multimodal": False}


def _warn_once(key: str, message: str, *, reason: str) -> None:
    """Emit a warning once, downgrade to debug on subsequent occurrences."""
    if not _WARNING_FLAGS.get(key, False):
        logger.warning("{} (reason={})", message, reason)
        _WARNING_FLAGS[key] = True
    else:
        logger.debug("{} (reason={})", message, reason)


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

    def _coerce_top_k(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(str(value))
            except (TypeError, ValueError):
                return None

    raw_top_k = getattr(getattr(cfg, "retrieval", cfg), "reranking_top_k", None)
    normalized_top_k = _coerce_top_k(raw_top_k)

    router_query_engine_cls = adapter.RouterQueryEngine
    retriever_query_engine_cls = adapter.RetrieverQueryEngine
    query_engine_tool_cls = adapter.QueryEngineTool
    tool_metadata_cls = adapter.ToolMetadata
    llm_single_selector_cls = adapter.LLMSingleSelector
    get_pydantic_selector = adapter.get_pydantic_selector

    try:
        use_rerank_flag = bool(getattr(cfg.retrieval, "use_reranking", True))
    except (AttributeError, TypeError, ValueError):  # pragma: no cover - defensive
        use_rerank_flag = True

    the_llm = llm if llm is not None else None

    # Vector tool (always on)
    try:
        from src.retrieval.reranking import get_postprocessors as _get_pp

        vector_post = _get_pp(
            "vector", use_reranking=use_rerank_flag, top_n=normalized_top_k
        )
        vector_engine = build_vector_query_engine(
            vector_index,
            vector_post,
            similarity_top_k=int(getattr(cfg.retrieval, "top_k", 10)),
        )
    except (TypeError, AttributeError, ValueError, ImportError):
        vector_engine = vector_index.as_query_engine()

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
    tools: list[Any] = [vector_tool]

    hybrid_requested = (
        bool(enable_hybrid)
        if enable_hybrid is not None
        else bool(getattr(cfg.retrieval, "enable_server_hybrid", False))
    )

    if hasattr(cfg, "is_graphrag_enabled"):
        kg_requested = bool(pg_index is not None and cfg.is_graphrag_enabled())
    else:
        kg_requested = bool(
            pg_index is not None and getattr(cfg, "enable_graphrag", True)
        )

    with router_build_span(
        adapter_name=getattr(adapter, "__class__", type("A", (), {})).__name__,
        kg_requested=kg_requested,
        hybrid_requested=hybrid_requested,
    ) as span:
        # Optional multimodal tool: fuse text hybrid + image retrieval (SigLIP).
        try:
            enable_mm = bool(getattr(cfg.retrieval, "enable_image_retrieval", True))
        except Exception:  # pragma: no cover - defensive
            enable_mm = True

        if enable_mm:
            try:
                from src.retrieval.multimodal_fusion import MultimodalFusionRetriever
                from src.retrieval.reranking import get_postprocessors as _get_pp

                mm_retriever = MultimodalFusionRetriever()
                mm_post = _get_pp(
                    "hybrid", use_reranking=use_rerank_flag, top_n=normalized_top_k
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
                                "Multimodal search (final-release): fuses text "
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

        # Optional hybrid tool
        if hybrid_requested:
            try:
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
                                "Hybrid search via Qdrant Query API "
                                "(dense+sparse fusion)"
                            ),
                        ),
                    )
                )
            except (ValueError, TypeError, AttributeError, ImportError) as exc:
                _warn_once(
                    "hybrid", "Hybrid tool construction skipped", reason=str(exc)
                )

        # Optional GraphRAG tool
        if (
            pg_index is not None
            and getattr(pg_index, "property_graph_store", None) is not None
            and (
                bool(cfg.is_graphrag_enabled())
                if hasattr(cfg, "is_graphrag_enabled")
                else bool(getattr(cfg, "enable_graphrag", True))
            )
        ):
            try:
                graph_depth = int(
                    getattr(getattr(cfg, "graphrag_cfg", cfg), "default_path_depth", 1)
                )
                graph_top_k = int(getattr(cfg.retrieval, "top_k", 10))
                from src.retrieval.reranking import get_postprocessors as _get_pp

                graph_post = _get_pp(
                    "kg", use_reranking=use_rerank_flag, top_n=normalized_top_k
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
                                "Knowledge graph traversal for "
                                "relationship-centric queries"
                            ),
                        ),
                    )
                )
            except MissingGraphAdapterError as exc:
                _warn_once("graph", GRAPH_DEPENDENCY_HINT, reason=str(exc))
            except (ValueError, TypeError, AttributeError, ImportError) as exc:
                logger.debug("Graph tool construction skipped: %s", exc)

        class _NoOpLLM:
            """Minimal no-op LLM stub for router defaults."""

            def __init__(self) -> None:
                self.metadata = type(
                    "_MD", (), {"context_window": 2048, "num_output": 256}
                )()

            def predict(self, *_args: Any, **_kwargs: Any) -> str:
                """Return an empty prediction string."""
                return ""

            def complete(self, *_args: Any, **_kwargs: Any) -> str:
                """Return an empty completion string."""
                return ""

        selector = get_pydantic_selector(the_llm)
        if selector is None:
            selector = llm_single_selector_cls.from_defaults(llm=the_llm or _NoOpLLM())

        try:
            router = router_query_engine_cls(
                selector=selector,
                query_engine_tools=tools,
                verbose=False,
                llm=the_llm or _NoOpLLM(),
            )
        except TypeError:
            router = router_query_engine_cls(
                selector=selector,
                query_engine_tools=tools,
                verbose=False,
            )

        tool_names: list[str] = []
        for tool in tools:
            metadata = getattr(tool, "metadata", None)
            name = getattr(metadata, "name", None)
            if name:
                tool_names.append(str(name))
        record_router_selection(
            span,
            tool_names=tool_names,
            kg_enabled="knowledge_graph" in tool_names,
        )

    try:
        router.query_engine_tools = tools
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        logger.debug("Router engine lacks public tool attribute", exc_info=True)

    logger.info(
        "Router engine built (kg_present=%s)",
        any(
            getattr(getattr(t, "metadata", None), "name", "") == "knowledge_graph"
            for t in tools
        ),
    )
    return router


__all__ = [
    "GRAPH_DEPENDENCY_HINT",
    "build_router_engine",
]
