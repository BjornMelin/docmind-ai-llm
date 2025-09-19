"""Router factory for composing RouterQueryEngine (library-first).

Builds a RouterQueryEngine with vector semantic search and optional
``knowledge_graph`` tool (GraphRAG). The graph tool is wrapped via
RetrieverQueryEngine with a retriever enforcing default path depth.
Selector policy uses PydanticSingleSelector when available, otherwise
falls back to LLMSingleSelector. Fail-open to vector-only when graph
is missing or unhealthy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from src.config.settings import settings as default_settings
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

if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from llama_index.core.query_engine import RouterQueryEngine as RouterQueryEngineType
else:
    RouterQueryEngineType = Any


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
    """Create a router engine with vector and optional KG tools.

    Args:
        vector_index: Vector index (semantic search) instance.
        pg_index: Optional PropertyGraphIndex for the knowledge_graph tool.
        settings: Optional settings object; defaults to module settings.
        llm: Optional LLM override for the router.
        enable_hybrid: When True, include server-side hybrid tool; defaults to
            settings.retrieval.enable_server_hybrid when None.
        adapter: LlamaIndex adapter (real or stub). Tests inject fakes to keep
            the module importable without the third-party dependency.

    Returns:
        RouterQueryEngine: Configured router engine.
    """
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

    try:
        from src.retrieval.reranking import get_postprocessors as _get_pp

        _v_post = _get_pp(
            "vector", use_reranking=use_rerank_flag, top_n=normalized_top_k
        )
        v_engine = build_vector_query_engine(
            vector_index, _v_post, similarity_top_k=cfg.retrieval.top_k
        )
    except (TypeError, AttributeError, ValueError, ImportError):
        v_engine = vector_index.as_query_engine()

    vector_tool = query_engine_tool_cls(
        query_engine=v_engine,
        metadata=tool_metadata_cls(
            name="semantic_search",
            description=(
                "Semantic vector search over the document corpus. Best for general "
                "questions and content lookup."
            ),
        ),
    )
    tools = [vector_tool]

    try:
        if enable_hybrid is not None:
            hybrid_ok = bool(enable_hybrid)
        else:
            hybrid_ok = bool(getattr(cfg.retrieval, "enable_server_hybrid", False))
        if hybrid_ok:
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

            _h_post = _get_pp(
                "hybrid", use_reranking=use_rerank_flag, top_n=normalized_top_k
            )
            h_engine = build_retriever_query_engine(
                retr,
                _h_post,
                llm=the_llm,
                response_mode="compact",
                verbose=False,
                engine_cls=retriever_query_engine_cls,
            )
            tools.append(
                query_engine_tool_cls(
                    query_engine=h_engine,
                    metadata=tool_metadata_cls(
                        name="hybrid_search",
                        description=(
                            "Hybrid search via Qdrant Query API (dense+sparse fusion)"
                        ),
                    ),
                )
            )
    except (
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
    ) as e:  # pragma: no cover - defensive
        logger.debug(f"Hybrid tool construction skipped: {e}")

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
                "kg", use_reranking=use_rerank_flag, top_n=normalized_top_k
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
    except (
        ValueError,
        TypeError,
        AttributeError,
        ImportError,
    ) as exc:  # pragma: no cover - defensive
        logger.debug(f"Graph tool construction skipped: {exc}")

    class _NoOpLLM:
        """Minimal no-op LLM stub for router defaults."""

        def __init__(self) -> None:
            """Provide basic metadata expected by the router."""
            self.metadata = type(
                "_MD", (), {"context_window": 2048, "num_output": 256}
            )()

        def predict(self, *_args: Any, **_kwargs: Any) -> str:
            """Return an empty response for predict calls."""
            return ""

        def complete(self, *_args: Any, **_kwargs: Any) -> str:
            """Return an empty response for completion calls."""
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

    try:
        router.query_engine_tools = tools
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        logger.debug(
            "Router engine lacks public tool attribute",
            exc_info=True,
        )

    logger.info(
        "Router engine built (kg_present=%s)",
        any(t.metadata.name == "knowledge_graph" for t in tools),
    )
    return router


try:
    adapter_cls = get_llama_index_adapter()
    RetrieverQueryEngine = adapter_cls.RetrieverQueryEngine  # type: ignore[assignment]
except MissingLlamaIndexError:

    class _MissingRetriever:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            raise MissingLlamaIndexError()

    RetrieverQueryEngine = _MissingRetriever  # type: ignore[assignment]


__all__ = ["RetrieverQueryEngine", "build_router_engine"]
