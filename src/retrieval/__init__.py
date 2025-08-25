"""FEAT-002 Retrieval & Search System.

Complete architectural replacement implementing BGE-M3 unified embeddings,
RouterQueryEngine adaptive retrieval, and CrossEncoder reranking per ADR-002,
ADR-003, and ADR-006.
"""

from .embeddings.bge_m3_manager import BGEM3Embedding, create_bgem3_embedding
from .postprocessor.cross_encoder_rerank import (
    BGECrossEncoderRerank,
    create_bge_cross_encoder_reranker,
)
from .query_engine.router_engine import (
    AdaptiveRouterQueryEngine,
    create_adaptive_router_engine,
)
from .vector_store.qdrant_unified import (
    QdrantUnifiedVectorStore,
    create_unified_qdrant_store,
)

__all__ = [
    # BGE-M3 Unified Embeddings
    "BGEM3Embedding",
    "create_bgem3_embedding",
    # CrossEncoder Reranking
    "BGECrossEncoderRerank",
    "create_bge_cross_encoder_reranker",
    # RouterQueryEngine Adaptive Retrieval
    "AdaptiveRouterQueryEngine",
    "create_adaptive_router_engine",
    # Qdrant Unified Vector Store
    "QdrantUnifiedVectorStore",
    "create_unified_qdrant_store",
]
