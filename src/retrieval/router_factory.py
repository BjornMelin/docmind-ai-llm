"""Router factory for composing RouterQueryEngine (library-first).

Builds a RouterQueryEngine with vector semantic search and optional
``knowledge_graph`` tool (GraphRAG). The graph tool is wrapped via
RetrieverQueryEngine with a retriever enforcing default path depth.
Selector policy uses PydanticSingleSelector when available, otherwise
falls back to LLMSingleSelector. Fail-open to vector-only when graph
is missing or unhealthy.
"""

from __future__ import annotations

from typing import Any

from llama_index.core import Settings
from llama_index.core.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from loguru import logger

from src.config.settings import settings as default_settings
from src.retrieval.hybrid import ServerHybridRetriever, _HybridParams


def build_router_engine(
    vector_index: Any,
    pg_index: Any | None = None,
    settings: Any | None = None,
    llm: Any | None = None,
    *,
    enable_hybrid: bool | None = None,
) -> RouterQueryEngine:
    """Create a router engine with vector and optional KG tools.

    Args:
        vector_index: Vector index (semantic search) instance.
        pg_index: Optional PropertyGraphIndex for the knowledge_graph tool.
        settings: Optional settings object; defaults to module settings.
        llm: Optional LLM override for the router.
        enable_hybrid: When True, include server-side hybrid tool; defaults to
            settings.retrieval.enable_server_hybrid when None.

    Returns:
        RouterQueryEngine: Configured router engine.
    """
    cfg = settings or default_settings
    the_llm = llm or Settings.llm

    # Vector semantic tool
    try:
        from src.retrieval.reranking import get_postprocessors as _get_pp

        _v_use = bool(getattr(cfg.retrieval, "use_reranking", True))
        _v_post = _get_pp("vector", use_reranking=_v_use)
        v_engine = vector_index.as_query_engine(
            similarity_top_k=cfg.retrieval.top_k,
            node_postprocessors=_v_post,
        )
    except (TypeError, AttributeError, ValueError):
        # Last-resort default
        v_engine = vector_index.as_query_engine()

    vector_tool = QueryEngineTool(
        query_engine=v_engine,
        metadata=ToolMetadata(
            name="semantic_search",
            description=(
                "Semantic vector search over the document corpus. Best for general "
                "questions and content lookup."
            ),
        ),
    )
    tools = [vector_tool]

    # Hybrid search tool via Qdrant Query API (behind flag)
    try:
        if enable_hybrid is not None:
            hybrid_ok = bool(enable_hybrid)
        else:
            # When explicit settings provided, allow either flag to enable hybrid.
            # When settings is None (uses default_settings), only honor the explicit
            # server-side flag to avoid surprises in generic callers/tests.
            if settings is None:
                hybrid_ok = bool(getattr(cfg.retrieval, "enable_server_hybrid", False))
            else:
                hybrid_ok = bool(
                    getattr(cfg.retrieval, "enable_server_hybrid", False)
                    or getattr(cfg.retrieval, "hybrid_enabled", False)
                )
        if hybrid_ok:
            params = _HybridParams(
                collection=cfg.database.qdrant_collection,
                fused_top_k=int(getattr(cfg.retrieval, "fused_top_k", 60)),
                prefetch_sparse=400,
                prefetch_dense=200,
                fusion_mode=str(getattr(cfg.retrieval, "fusion_mode", "rrf")),
                dedup_key=str(getattr(cfg.retrieval, "dedup_key", "page_id")),
            )
            retr = ServerHybridRetriever(params)
            from src.retrieval.reranking import get_postprocessors as _get_pp

            _h_use = bool(getattr(cfg.retrieval, "use_reranking", True))
            _h_post = _get_pp("hybrid", use_reranking=_h_use)
            try:
                h_engine = RetrieverQueryEngine.from_args(
                    retriever=retr,
                    llm=the_llm,
                    response_mode="compact",
                    verbose=False,
                    node_postprocessors=_h_post,
                )
            except TypeError:
                h_engine = RetrieverQueryEngine.from_args(
                    retriever=retr, llm=the_llm, response_mode="compact", verbose=False
                )
            tools.append(
                QueryEngineTool(
                    query_engine=h_engine,
                    metadata=ToolMetadata(
                        name="hybrid_search",
                        description=(
                            "Hybrid search via Qdrant Query API (dense+sparse fusion)"
                        ),
                    ),
                )
            )
    except (ValueError, TypeError, AttributeError) as e:  # pragma: no cover - defensive
        logger.debug(f"Hybrid tool construction skipped: {e}")

    # Optional graph tool
    try:
        if (
            pg_index is not None
            and getattr(pg_index, "property_graph_store", None) is not None
        ):
            default_depth = (
                int(getattr(getattr(cfg, "graphrag_cfg", cfg), "default_path_depth", 1))
                if cfg is not None
                else 1
            )
            g_engine = None
            if hasattr(pg_index, "as_retriever"):
                retr = pg_index.as_retriever(
                    include_text=True, path_depth=default_depth
                )
                from src.retrieval.reranking import get_postprocessors as _get_pp

                _g_use = bool(getattr(cfg.retrieval, "use_reranking", True))
                _g_post = _get_pp(
                    "kg",
                    use_reranking=_g_use,
                    top_n=int(getattr(cfg.retrieval, "reranking_top_k", 5)),
                )
                try:
                    g_engine = RetrieverQueryEngine.from_args(
                        retriever=retr,
                        llm=the_llm,
                        response_mode="compact",
                        verbose=False,
                        node_postprocessors=_g_post,
                    )
                except TypeError:
                    g_engine = RetrieverQueryEngine.from_args(
                        retriever=retr,
                        llm=the_llm,
                        response_mode="compact",
                        verbose=False,
                    )
            elif hasattr(pg_index, "as_query_engine"):
                from src.retrieval.reranking import get_postprocessors as _get_pp

                _g_use2 = bool(getattr(cfg.retrieval, "use_reranking", True))
                _g_post2 = _get_pp(
                    "kg",
                    use_reranking=_g_use2,
                    top_n=int(getattr(cfg.retrieval, "reranking_top_k", 5)),
                )
                try:
                    g_engine = pg_index.as_query_engine(
                        include_text=True, node_postprocessors=_g_post2
                    )
                except TypeError:
                    g_engine = pg_index.as_query_engine(include_text=True)
            tools.append(
                QueryEngineTool(
                    query_engine=g_engine,
                    metadata=ToolMetadata(
                        name="knowledge_graph",
                        description=(
                            "Knowledge graph traversal for relationship-centric queries"
                        ),
                    ),
                )
            )
    except (ValueError, TypeError, AttributeError) as e:  # pragma: no cover - defensive
        logger.debug(f"Graph tool construction skipped: {e}")
    # Selector: prefer PydanticSingleSelector when available, else LLM selector
    try:
        from llama_index.core.selectors import PydanticSingleSelector

        selector = PydanticSingleSelector.from_defaults(llm=the_llm)
    except (ImportError, AttributeError):
        selector = LLMSingleSelector.from_defaults(llm=the_llm)

    router = RouterQueryEngine(
        selector=selector, query_engine_tools=tools, verbose=False
    )
    # Expose both public and private tool lists for compatibility across LI versions
    try:
        router.query_engine_tools = tools
        router._query_engine_tools = tools
    except Exception:  # pragma: no cover - defensive
        logger.debug("Router tool list attribute shim failed", exc_info=True)
    logger.info(
        "Router engine built (kg_present=%s)",
        any(t.metadata.name == "knowledge_graph" for t in tools),
    )
    return router


__all__ = ["build_router_engine"]
