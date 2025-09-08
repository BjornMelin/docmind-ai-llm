"""RouterQueryEngine implementation for adaptive retrieval strategy selection.

This module implements the complete architectural replacement of QueryFusionRetriever
with RouterQueryEngine per ADR-003, providing intelligent strategy selection based
on query characteristics.

Key features:
- LLMSingleSelector for automatic strategy selection
- QueryEngineTool definitions for vector/hybrid/sub_question/graph/multimodal strategies
- Multimodal query detection for CLIP image search
- Fallback mechanisms for robustness
- Integration with BGE-M3 embeddings and CrossEncoder reranking
"""

import os
from contextlib import suppress
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from llama_index.core import Settings
from llama_index.core.query_engine import (
    RetrieverQueryEngine,
    RouterQueryEngine,
    SubQuestionQueryEngine,
)
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from src.processing.utils import sha256_id
from src.retrieval.reranking import MultimodalReranker
from src.retrieval.sparse_query import encode_to_qdrant as _encode_sparse_query
from src.utils.security import build_owner_filter
from src.utils.storage import get_client_config

# Router Engine Configuration Constants
DEFAULT_DENSE_SIMILARITY_TOP_K = 10
DEFAULT_SUB_QUESTION_SIMILARITY_TOP_K = 15
DEFAULT_KG_SIMILARITY_TOP_K = 10
DEFAULT_MULTIMODAL_SIMILARITY_TOP_K = 10
DEFAULT_IMAGE_SIMILARITY_TOP_K = 5
QUERY_TRUNCATE_LENGTH = 100
FUSED_TOP_K_DEFAULT = 60
SPARSE_PREFETCH_DEFAULT = 400
DENSE_PREFETCH_DEFAULT = 200


@dataclass
class _HybridParams:
    collection: str
    fused_top_k: int = FUSED_TOP_K_DEFAULT
    prefetch_sparse: int = SPARSE_PREFETCH_DEFAULT
    prefetch_dense: int = DENSE_PREFETCH_DEFAULT
    fusion_mode: str = "rrf"
    dedup_key: str = "page_id"


class ServerHybridRetriever:
    """Retriever that uses Qdrant Query API server-side fusion (RRF/DBSF).

    - Computes BGE-M3 dense+sparse for the query text.
    - Prefetches sparse+dense using named vectors 'text-sparse'/'text-dense'.
    - De-dups by page_id before final fused cut.
    - Returns LlamaIndex NodeWithScore list.
    """

    def __init__(self, params: _HybridParams) -> None:
        """Initialize with Qdrant client and a text embedder.

        Args:
            params: Hybrid retrieval parameters (collection, limits, fusion).
        """
        self.params = params
        self._client = QdrantClient(**get_client_config())

    def _embed_query(self, text: str) -> tuple[np.ndarray, qmodels.SparseVector | None]:
        """Encode query text into dense and sparse vectors.

        Dense uses LlamaIndex Settings.embed_model to align with index-time
        embeddings. Sparse uses FastEmbed (BM42 → BM25 fallback) to align with
        Qdrant hybrid sparse indexing; returns empty mapping when unavailable.
        """
        # Dense via LI Settings for alignment
        dvec = Settings.embed_model.get_query_embedding(text)  # type: ignore[attr-defined]
        dense_vec = np.asarray(dvec, dtype=np.float32)

        # Sparse via FastEmbed for alignment with Qdrant hybrid sparse
        svec = _encode_sparse_query(text)
        return dense_vec, svec

    def _fusion(self) -> qmodels.FusionQuery:
        """Create fusion query based on configured mode."""
        mode = self.params.fusion_mode
        if mode == "dbsf":
            return qmodels.FusionQuery(fusion=qmodels.Fusion.DBSF)
        return qmodels.FusionQuery(fusion=qmodels.Fusion.RRF)

    def retrieve(self, query: str | QueryBundle) -> list[NodeWithScore]:
        """Execute a server-side hybrid query (RRF/DBSF) and de-dup by page_id.

        Args:
            query: Query string or QueryBundle to execute.

        Returns:
            list[NodeWithScore]: Retrieved nodes with scores, deduplicated by page_id.
        """
        qtext = query.query_str if isinstance(query, QueryBundle) else str(query)
        t0 = perf_counter()
        dense_vec, sparse_vec = self._embed_query(qtext)

        prefetch: list[qmodels.Prefetch] = []
        if sparse_vec is not None:
            # Accept dict-based sparse {index: weight} and convert to typed model
            try:
                if isinstance(sparse_vec, dict):
                    # Deterministic ordering by index
                    idxs, vals = (
                        zip(*sorted(sparse_vec.items()), strict=False)
                        if sparse_vec
                        else ([], [])
                    )
                    sv = qmodels.SparseVector(indices=list(idxs), values=list(vals))  # type: ignore[arg-type]
                else:
                    sv = sparse_vec  # already typed or supported format
            except Exception:
                # Last-resort fallback: attempt to wrap values as needed
                if isinstance(sparse_vec, dict):
                    sv = qmodels.SparseVector(
                        indices=list(sparse_vec.keys()),
                        values=list(sparse_vec.values()),
                    )  # type: ignore[arg-type]
                else:
                    sv = sparse_vec
            prefetch.append(
                qmodels.Prefetch(
                    query=sv,
                    using="text-sparse",
                    limit=self.params.prefetch_sparse,
                )
            )
        d_list = dense_vec.tolist() if hasattr(dense_vec, "tolist") else list(dense_vec)
        # Prefer typed VectorInput when available as a real class; otherwise pass list
        vi_query: Any = d_list
        vi = getattr(qmodels, "VectorInput", None)
        if vi is not None:
            try:  # support older client versions where VectorInput was a model
                vi_query = vi(vector=d_list)
            except (
                TypeError
            ):  # new clients expose it as typing.Union and cannot be instantiated
                vi_query = d_list
        prefetch.append(
            qmodels.Prefetch(
                query=vi_query,  # type: ignore[arg-type]
                using="text-dense",
                limit=self.params.prefetch_dense,
            )
        )

        # Optional RBAC-style filter via env (owner scoping)
        owner = os.getenv("DOCMIND_OWNER_ID", "").strip()
        qfilter = None
        if owner:
            # Build typed filter to avoid schema drift across client versions
            try:
                cond = qmodels.FieldCondition(
                    key="owner_id", match=qmodels.MatchValue(value=owner)
                )
                qfilter = qmodels.Filter(must=[cond])
            except (TypeError, ValueError, RuntimeError):
                # Fallback to dict-based helper if client types mismatch
                fdict = build_owner_filter(owner)
                qfilter = qmodels.Filter(**fdict)  # type: ignore[arg-type]

        result = self._client.query_points(
            collection_name=self.params.collection,
            prefetch=prefetch,
            query=self._fusion(),
            limit=self.params.fused_top_k,
            query_filter=qfilter,
            with_payload=[
                "doc_id",
                "page_id",
                "chunk_id",
                "text",
                "modality",
                "image_path",
            ],
        )

        # De-dup by configured key before final cut
        best: dict[str, tuple[float, Any]] = {}
        orig = len(result.points)
        for p in result.points:
            payload = p.payload or {}
            key_name = (self.params.dedup_key or "page_id").strip()
            key = str(payload.get(key_name) or p.id)
            score = float(getattr(p, "score", 0.0))
            cur = best.get(key)
            if cur is None or score > cur[0]:
                best[key] = (score, p)

        # Preserve ranking by score desc
        dedup_sorted = sorted(best.values(), key=lambda x: x[0], reverse=True)
        nodes: list[NodeWithScore] = []
        for score, p in dedup_sorted[: self.params.fused_top_k]:
            payload = p.payload or {}
            text = payload.get("text") or ""
            # Deterministic node id based on available identifiers
            nid = str(p.id) if p.id is not None else sha256_id(text)
            node = TextNode(text=text, id_=nid)
            node.metadata.update({k: v for k, v in payload.items() if k != "text"})
            nodes.append(NodeWithScore(node=node, score=score))

        t1 = perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        dedup_filtered = max(0, orig - len(best))
        try:
            from src.utils.telemetry import log_jsonl

            # Emit sparse fallback flag when no sparse prefetch was used
            if sparse_vec is None:
                with suppress(Exception):  # pragma: no cover - defensive
                    log_jsonl({"retrieval.sparse_fallback": True})
            log_jsonl(
                {
                    "retrieval.fusion_mode": self.params.fusion_mode,
                    "retrieval.prefetch_dense_limit": int(self.params.prefetch_dense),
                    "retrieval.prefetch_sparse_limit": int(self.params.prefetch_sparse),
                    "retrieval.fused_limit": int(self.params.fused_top_k),
                    "retrieval.return_count": len(nodes),
                    "retrieval.latency_ms": int(latency_ms),
                    "dedup.before": int(orig),
                    "dedup.after": len(best),
                    "dedup.dropped": int(dedup_filtered),
                    "dedup.key": (self.params.dedup_key or "page_id"),
                }
            )
        except Exception:  # pylint: disable=broad-exception-caught
            logger.info(
                (
                    "Qdrant hybrid: top_k=%d dense=%d "
                    "sparse=%d fusion=%s latency=%.1fms dedup=%d"
                ),
                self.params.fused_top_k,
                self.params.prefetch_dense,
                self.params.prefetch_sparse,
                self.params.fusion_mode,
                latency_ms,
                dedup_filtered,
            )
        return nodes


class AdaptiveRouterQueryEngine:
    """Adaptive RouterQueryEngine for FEAT-002 retrieval system.

    Uses RouterQueryEngine with LLMSingleSelector to intelligently
    choose between different retrieval strategies based on query characteristics.
    Replaces QueryFusionRetriever with modern adaptive routing per ADR-003.

    Supported Strategies:
    - Dense semantic search (BGE-M3 dense vectors)
    - Hybrid search (BGE-M3 dense + sparse vectors with RRF fusion)
    - Sub-question search (query decomposition for complex questions)
    - Knowledge graph search (GraphRAG relationships, optional)
    - Multimodal search (CLIP image-text cross-modal retrieval)

    Performance targets (RTX 4090 Laptop):
    - <50ms strategy selection overhead
    - <2s P95 query latency including reranking
    - >90% correct strategy selection accuracy
    """

    def __init__(
        self,
        *,
        vector_index: Any,
        kg_index: Any | None = None,
        hybrid_retriever: Any | None = None,
        multimodal_index: Any | None = None,
        reranker: Any | None = None,
        llm: Any | None = None,
    ):
        """Initialize AdaptiveRouterQueryEngine.

        Args:
            vector_index: Primary vector index for semantic search
            kg_index: Optional knowledge graph index for relationships
            hybrid_retriever: Optional hybrid retriever for dense+sparse search
            multimodal_index: Optional multimodal index for CLIP image-text search
            reranker: Optional reranker for result quality improvement
            llm: Optional LLM for strategy selection (defaults to Settings.llm)
        """
        self.vector_index = vector_index
        self.kg_index = kg_index
        self.hybrid_retriever = hybrid_retriever
        self.multimodal_index = multimodal_index
        self.reranker = reranker
        self.llm = llm or Settings.llm
        self._query_engine_tools = self._create_query_engine_tools()
        self.router_engine = self._create_router_engine()

    def _create_query_engine_tools(self) -> list[QueryEngineTool]:
        """Create QueryEngineTool instances for router selection.

        Each tool represents a different retrieval strategy with detailed
        descriptions for the LLM selector to make optimal routing decisions.

        Returns:
            list[QueryEngineTool]: List of QueryEngineTool instances for
                                   RouterQueryEngine.
        """
        tools: list[QueryEngineTool] = []
        hybrid_tool: QueryEngineTool | None = None
        vector_tool: QueryEngineTool | None = None

        # 1. Hybrid Search Tool (Primary - BGE-M3 Dense + Sparse)
        if self.hybrid_retriever:
            hybrid_engine = RetrieverQueryEngine.from_args(
                retriever=self.hybrid_retriever,
                llm=self.llm,
                node_postprocessors=[self.reranker] if self.reranker else [],
                response_mode="compact",
                streaming=True,
            )
            hybrid_tool = QueryEngineTool(
                query_engine=hybrid_engine,
                metadata=ToolMetadata(
                    name="hybrid_search",
                    description=(
                        "Advanced hybrid search combining BGE-M3 unified dense "
                        "and sparse embeddings with RRF fusion. This strategy "
                        "provides the best balance of semantic understanding and "
                        "keyword precision. Optimal for: comprehensive document "
                        "retrieval, complex queries requiring both conceptual "
                        "understanding and specific term matching, technical "
                        "documentation search, multi-faceted questions needing "
                        "diverse result types. Uses BGE-M3's 8K context and "
                        "cross-encoder reranking for superior relevance."
                    ),
                ),
            )
            tools.append(hybrid_tool)
        else:
            # Provide server-side hybrid by default with Query API
            try:
                from src.config import settings as app_settings

                collection = app_settings.database.qdrant_collection
                # Derive params from settings with sane defaults
                fused_top = int(
                    getattr(app_settings.retrieval, "fused_top_k", FUSED_TOP_K_DEFAULT)
                )
                fusion_mode = str(
                    getattr(app_settings.retrieval, "fusion_mode", "rrf")
                ).lower()
                dedup_key = str(
                    getattr(app_settings.retrieval, "dedup_key", "page_id")
                ).lower()
                params = _HybridParams(
                    collection=collection,
                    fused_top_k=fused_top,
                    prefetch_sparse=SPARSE_PREFETCH_DEFAULT,
                    prefetch_dense=DENSE_PREFETCH_DEFAULT,
                    fusion_mode=fusion_mode,
                    dedup_key=dedup_key,
                )
                retriever = ServerHybridRetriever(params)
                hybrid_engine = RetrieverQueryEngine.from_args(
                    retriever=retriever,
                    llm=self.llm,
                    node_postprocessors=[self.reranker] if self.reranker else [],
                    response_mode="compact",
                    streaming=True,
                )
                tools.append(
                    QueryEngineTool(
                        query_engine=hybrid_engine,
                        metadata=ToolMetadata(
                            name="hybrid_search",
                            description=(
                                "Server-side hybrid via Qdrant Query API. "
                                "RRF default; DBSF optional. "
                                "Prefetch dense+sparse; de-dup page_id; fused_top_k=60."
                            ),
                        ),
                    )
                )
            # If hybrid construction fails, warn and continue with dense-only
            except (ValueError, RuntimeError) as e:  # pragma: no cover - optional
                logger.warning("Server hybrid unavailable: %s", e)

        # 2. Dense Semantic Search Tool (BGE-M3 Dense Only)
        dense_engine = self.vector_index.as_query_engine(
            similarity_top_k=DEFAULT_DENSE_SIMILARITY_TOP_K,
            node_postprocessors=[self.reranker] if self.reranker else [],
            response_mode="compact",
            streaming=True,
        )
        vector_tool = QueryEngineTool(
            query_engine=dense_engine,
            metadata=ToolMetadata(
                name="semantic_search",
                description=(
                    "Dense semantic search using BGE-M3 unified 1024-dimensional "
                    "embeddings for deep conceptual understanding. Excels at: "
                    "finding semantically similar content, conceptual questions, "
                    "summarization tasks, meaning-based retrieval, cross-lingual "
                    "queries, and abstract concept exploration. Uses cosine "
                    "similarity for precise semantic matching with BGE-M3's "
                    "multilingual capabilities and 8K context window."
                ),
            ),
        )
        tools.append(vector_tool)

        # 3. Sub-Question Search Tool (Query Decomposition via SubQuestionQueryEngine)
        try:
            subq_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=[
                    t for t in [vector_tool, hybrid_tool] if t is not None
                ],
                llm=self.llm,
                use_async=True,
                verbose=False,
            )
            tools.append(
                QueryEngineTool(
                    query_engine=subq_engine,
                    metadata=ToolMetadata(
                        name="sub_question_search",
                        description=(
                            "Sub-question decomposition strategy for complex, "
                            "multi-part questions. Optimal for: analytical and "
                            "compare/contrast queries requiring multiple retrieval "
                            "hops and synthesis. Uses tree summarization over "
                            "sub-queries."
                        ),
                    ),
                )
            )
        except (ImportError, RuntimeError, ValueError, AttributeError) as e:
            logger.warning("SubQuestionQueryEngine unavailable: %s", e)
            # Fallback: expose a tree-summarize vector engine under the same tool name
            fallback_engine = self.vector_index.as_query_engine(
                similarity_top_k=DEFAULT_SUB_QUESTION_SIMILARITY_TOP_K,
                node_postprocessors=[self.reranker] if self.reranker else [],
                response_mode="tree_summarize",
                streaming=True,
            )
            tools.append(
                QueryEngineTool(
                    query_engine=fallback_engine,
                    metadata=ToolMetadata(
                        name="sub_question_search",
                        description=(
                            "Fallback sub-question search using tree summarization "
                            "over semantic results when SubQuestionQueryEngine is "
                            "unavailable."
                        ),
                    ),
                )
            )

        # 4. Knowledge Graph Search Tool (Relationships - Optional)
        if self.kg_index:
            kg_engine = self.kg_index.as_query_engine(
                similarity_top_k=DEFAULT_KG_SIMILARITY_TOP_K,
                include_text=True,
                node_postprocessors=[self.reranker] if self.reranker else [],
                response_mode="compact",
                streaming=True,
            )
            tools.append(
                QueryEngineTool(
                    query_engine=kg_engine,
                    metadata=ToolMetadata(
                        name="knowledge_graph",
                        description=(
                            "Knowledge graph search for exploring entity relationships "
                            "and structured connections within documents. Specialized "
                            "for: relationship queries ('how are X and Y connected?'), "
                            "entity-centric questions, hierarchical structure "
                            "exploration, dependency analysis, network analysis of "
                            "concepts. Combines graph traversal with semantic search "
                            "to understand how different concepts relate to each other "
                            "across the document corpus."
                        ),
                    ),
                )
            )

        # 5. Multimodal Search Tool (CLIP Image-Text Cross-Modal - Optional)
        if self.multimodal_index:
            multimodal_engine = self.multimodal_index.as_query_engine(
                similarity_top_k=DEFAULT_MULTIMODAL_SIMILARITY_TOP_K,
                image_similarity_top_k=DEFAULT_IMAGE_SIMILARITY_TOP_K,
                node_postprocessors=[self.reranker] if self.reranker else [],
                response_mode="compact",
                streaming=True,
            )
            tools.append(
                QueryEngineTool(
                    query_engine=multimodal_engine,
                    metadata=ToolMetadata(
                        name="multimodal_search",
                        description=(
                            "Multimodal search using CLIP backbones for cross-modal "
                            "image-text retrieval. Specialized for: image-related "
                            "queries, visual content questions, diagrams and charts "
                            "analysis, text-to-image search, image-to-text search, "
                            "and visual similarity matching. Embeddings are derived "
                            "at runtime from the selected backbone. Optimized for "
                            "low VRAM usage while maintaining high accuracy for "
                            "visual/textual content correlation."
                        ),
                    ),
                )
            )

        logger.info("Created {} query engine tools for adaptive routing", len(tools))
        return tools

    def _detect_multimodal_query(self, query_str: str) -> bool:
        """Detect if a query involves multimodal/image content.

        Args:
            query_str: User query string.

        Returns:
            bool: True if query likely involves images/visual content.
        """
        # Pattern-based detection for image-related queries
        image_keywords = [
            "image",
            "picture",
            "photo",
            "diagram",
            "chart",
            "graph",
            "figure",
            "screenshot",
            "visualization",
            "visual",
            "show me",
            "display",
            "view",
            "illustration",
            "drawing",
            "sketch",
            "icon",
            "logo",
            "banner",
            "infographic",
        ]

        image_phrases = [
            "show me diagrams",
            "find images",
            "visual representation",
            "what does it look like",
            "similar images",
            "image of",
            "picture of",
            "screenshot of",
            "diagram showing",
            "chart displaying",
            "graph of",
        ]

        query_lower = query_str.lower()

        # Check for image keywords
        if any(keyword in query_lower for keyword in image_keywords):
            return True

        # Check for image-related phrases
        if any(phrase in query_lower for phrase in image_phrases):
            return True

        # Pattern matching for specific image requests
        return "file:" in query_lower or ".jpg" in query_lower or ".png" in query_lower

    def _create_router_engine(self) -> RouterQueryEngine:
        """Create RouterQueryEngine with LLMSingleSelector.

        Uses LLMSingleSelector for intelligent routing decisions based on
        query analysis. Provides fallback mechanisms for robustness.

        Returns:
            RouterQueryEngine: Configured RouterQueryEngine with adaptive routing.
        """
        query_engine_tools = self._query_engine_tools

        if not query_engine_tools:
            raise ValueError("No query engine tools available for router")

        # Create LLM selector for intelligent routing
        selector = LLMSingleSelector.from_defaults(llm=self.llm)

        # Create router with fallback to first tool (semantic search)
        router_engine = RouterQueryEngine(
            selector=selector,
            query_engine_tools=query_engine_tools,
            verbose=True,  # Enable routing decision logging
        )

        logger.info(
            "RouterQueryEngine created with LLMSingleSelector for adaptive routing"
        )
        return router_engine

    def query(self, query_str: str, **kwargs: Any) -> Any:
        """Execute query through adaptive routing.

        The RouterQueryEngine analyzes the query and automatically selects
        the optimal retrieval strategy based on query characteristics.

        Args:
            query_str: User query text.
            **kwargs: Additional query parameters.

        Returns:
            Any: Query response with metadata about selected strategy.
        """
        try:
            logger.info(
                "Executing adaptive query: {}...",
                query_str[:QUERY_TRUNCATE_LENGTH],
            )

            # Execute through RouterQueryEngine
            response = self.router_engine.query(query_str, **kwargs)

            # Log selected strategy if available
            selected_tool = getattr(response, "metadata", {}).get("selector_result")
            if selected_tool:
                logger.info("Router selected strategy: {}", selected_tool)
            else:
                logger.info(
                    "Router executed query (strategy selection metadata unavailable)"
                )

            return response

        except (RuntimeError, ValueError, TimeoutError) as e:
            logger.error("RouterQueryEngine failed: {}", e)
            # Fallback to direct semantic search
            logger.info("Falling back to direct semantic search")
            return self.vector_index.as_query_engine().query(query_str, **kwargs)

    async def aquery(self, query_str: str, **kwargs: Any) -> Any:
        """Async query execution through adaptive routing.

        Args:
            query_str: User query text.
            **kwargs: Additional query parameters.

        Returns:
            Any: Query response with metadata about selected strategy.
        """
        try:
            logger.info(
                "Executing async adaptive query: {}...",
                query_str[:QUERY_TRUNCATE_LENGTH],
            )

            response = await self.router_engine.aquery(query_str, **kwargs)

            # Log selected strategy if available
            selected_tool = getattr(response, "metadata", {}).get("selector_result")
            if selected_tool:
                logger.info("Router selected strategy: {}", selected_tool)

            return response

        except (RuntimeError, ValueError, TimeoutError) as e:
            logger.error("Async RouterQueryEngine failed: {}", e)
            # Fallback to direct semantic search
            logger.info("Falling back to async semantic search")
            return await self.vector_index.as_query_engine().aquery(query_str, **kwargs)

    def get_available_strategies(self) -> list[str]:
        """Get list of available retrieval strategies.

        Returns:
            list[str]: List of strategy names available for routing.
        """
        return [tool.metadata.name for tool in self._query_engine_tools]


def create_adaptive_router_engine(  # pylint: disable=too-many-positional-arguments
    vector_index: Any,
    kg_index: Any | None = None,
    hybrid_retriever: Any | None = None,
    multimodal_index: Any | None = None,
    reranker: Any | None = None,
    llm: Any | None = None,
) -> AdaptiveRouterQueryEngine:
    """Factory function for creating adaptive router engine.

    Factory function following library-first principle for easy instantiation
    with comprehensive strategy support including multimodal CLIP search.

    Args:
        vector_index: Primary vector index for semantic search.
        kg_index: Optional knowledge graph index for relationships.
        hybrid_retriever: Optional hybrid retriever for dense+sparse search.
        multimodal_index: Optional multimodal index for CLIP image-text search.
        reranker: Optional reranker for result quality improvement.
        llm: Optional LLM for strategy selection (defaults to Settings.llm).

    Returns:
        AdaptiveRouterQueryEngine: Configured AdaptiveRouterQueryEngine.
    """
    # Build reranker if not provided
    # Import settings lazily to avoid any potential test-time import shadowing
    from src.config import settings as app_settings  # local import by design

    if reranker is None and getattr(app_settings.retrieval, "use_reranking", True):
        # Always-on modality-aware reranking (text + visual policy)
        reranker = MultimodalReranker()

    return AdaptiveRouterQueryEngine(
        vector_index=vector_index,
        kg_index=kg_index,
        hybrid_retriever=hybrid_retriever,
        multimodal_index=multimodal_index,
        reranker=reranker,
        llm=llm,
    )


def configure_router_settings(_router_engine: AdaptiveRouterQueryEngine) -> None:
    """Configure LlamaIndex Settings for RouterQueryEngine.

    Updates global Settings to use the AdaptiveRouterQueryEngine
    as the primary query interface.

    Args:
        _router_engine: Configured AdaptiveRouterQueryEngine instance.
    """
    try:
        # Note: Settings doesn't have a direct query_engine property
        # This would be handled at the application level
        logger.info("RouterQueryEngine configured for adaptive retrieval")
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.error("Failed to configure router settings: {}", e)
        raise
