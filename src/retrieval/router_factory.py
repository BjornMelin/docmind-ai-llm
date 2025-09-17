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

from llama_index.core.query_engine import (
    RetrieverQueryEngine as _RetrieverQueryEngine,
)
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from loguru import logger

from src.config.settings import settings as default_settings
from src.retrieval.graph_config import build_graph_query_engine
from src.retrieval.postprocessor_utils import (
    build_retriever_query_engine,
    build_vector_query_engine,
)

RetrieverQueryEngine = _RetrieverQueryEngine


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
    # Unified gating flag (DRY): resolve once
    try:
        use_rerank_flag = bool(getattr(cfg.retrieval, "use_reranking", True))
    except Exception:  # pragma: no cover - defensive
        use_rerank_flag = True
    the_llm = llm if llm is not None else None

    # Vector semantic tool
    try:
        from src.retrieval.reranking import get_postprocessors as _get_pp

        _v_post = _get_pp("vector", use_reranking=use_rerank_flag)
        v_engine = build_vector_query_engine(
            vector_index, _v_post, similarity_top_k=cfg.retrieval.top_k
        )
    except (TypeError, AttributeError, ValueError, ImportError):
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
            # Single authoritative flag per ADR-024
            hybrid_ok = bool(getattr(cfg.retrieval, "enable_server_hybrid", False))
        if hybrid_ok:
            # Import retriever on-demand to avoid heavy imports at module load.
            from src.retrieval.hybrid import (
                ServerHybridRetriever as _shr,  # noqa: N813
            )
            from src.retrieval.hybrid import _HybridParams as _hp

            params = _hp(
                collection=cfg.database.qdrant_collection,
                fused_top_k=int(getattr(cfg.retrieval, "fused_top_k", 60)),
                prefetch_sparse=int(
                    getattr(cfg.retrieval, "prefetch_sparse_limit", 400)
                ),
                prefetch_dense=int(getattr(cfg.retrieval, "prefetch_dense_limit", 200)),
                fusion_mode=str(getattr(cfg.retrieval, "fusion_mode", "rrf")),
                dedup_key=str(getattr(cfg.retrieval, "dedup_key", "page_id")),
            )
            retr = _shr(params)
            from src.retrieval.reranking import get_postprocessors as _get_pp

            _h_post = _get_pp("hybrid", use_reranking=use_rerank_flag)
            h_engine = build_retriever_query_engine(
                retr,
                _h_post,
                llm=the_llm,
                response_mode="compact",
                verbose=False,
                engine_cls=RetrieverQueryEngine,
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
    # pragma: no cover - defensive
    except (ValueError, TypeError, AttributeError, ImportError) as e:
        logger.debug(f"Hybrid tool construction skipped: {e}")

    # Optional graph tool (GraphRAG)
    try:
        if (
            pg_index is not None
            and getattr(pg_index, "property_graph_store", None) is not None
            and bool(getattr(cfg, "enable_graphrag", True))
        ):
            graph_depth = int(
                getattr(getattr(cfg, "graphrag_cfg", cfg), "default_path_depth", 1)
            )
            graph_top_k = int(getattr(cfg.retrieval, "top_k", 10))
            from src.retrieval.reranking import get_postprocessors as _get_pp

            graph_post = _get_pp(
                "kg",
                use_reranking=use_rerank_flag,
                top_n=int(getattr(cfg.retrieval, "reranking_top_k", 5)),
            )
            artifacts = build_graph_query_engine(
                pg_index,
                llm=the_llm,
                include_text=True,
                path_depth=graph_depth,
                similarity_top_k=graph_top_k,
                node_postprocessors=graph_post,
            )
            tools.append(
                QueryEngineTool(
                    query_engine=artifacts.query_engine,
                    metadata=ToolMetadata(
                        name="knowledge_graph",
                        description=(
                            "Knowledge graph traversal for relationship-centric queries"
                        ),
                    ),
                )
            )
    except (
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
    ) as exc:  # pragma: no cover - defensive
        logger.debug(f"Graph tool construction skipped: {exc}")

    # No-op LLM stub to prevent LlamaIndex from resolving Settings.llm
    class _NoOpLLM:
        """Minimal no-op LLM stub for router defaults.

        Provides required attributes and methods so that LlamaIndex router
        construction does not attempt to implicitly resolve ``Settings.llm``.
        """

        def __init__(self):
            self.metadata = type(
                "_MD", (), {"context_window": 2048, "num_output": 256}
            )()

        def predict(self, *_args: Any, **_kwargs: Any) -> str:
            """No-op predict."""
            return ""

        def complete(self, *_args: Any, **_kwargs: Any) -> str:
            """No-op complete."""
            return ""

    # Selector: prefer PydanticSingleSelector when available, else LLM selector
    try:
        from llama_index.core.selectors import PydanticSingleSelector

        selector = PydanticSingleSelector.from_defaults(llm=the_llm)
    except (ImportError, AttributeError):
        selector = LLMSingleSelector.from_defaults(llm=the_llm or _NoOpLLM())

    try:
        router = RouterQueryEngine(
            selector=selector,
            query_engine_tools=tools,
            verbose=False,
            llm=the_llm or _NoOpLLM(),
        )
    except TypeError:
        router = RouterQueryEngine(
            selector=selector,
            query_engine_tools=tools,
            verbose=False,
        )
    # Expose both public and private tool lists for compatibility across LI versions
    # Expose public tool list for downstream introspection
    try:
        router.query_engine_tools = tools
    except Exception:  # pragma: no cover - defensive
        logger.debug("Router engine lacks public tool attribute", exc_info=True)
    logger.info(
        "Router engine built (kg_present=%s)",
        any(t.metadata.name == "knowledge_graph" for t in tools),
    )
    return router


__all__ = ["RetrieverQueryEngine", "build_router_engine"]
