"""FEAT-002 Retrieval & Search System.

Complete architectural replacement implementing BGE-M3 unified embeddings,
RouterQueryEngine adaptive retrieval, and CrossEncoder reranking per ADR-002,
ADR-003, and ADR-006.

Flattened directory structure with consolidated modules per Task 3.1.1.
"""

# BGE-M3 and CLIP embeddings (consolidated)
from .embeddings import (
    BGEM3Embedding,
    ClipConfig,
    configure_bgem3_settings,
    create_bgem3_embedding,
    create_clip_embedding,
    setup_clip_for_llamaindex,
)

# Property graph configuration
from .graph_config import (
    PropertyGraphConfig,
    calculate_entity_confidence,
    create_property_graph_index,
    create_property_graph_index_async,
    create_tech_schema,
    extend_property_graph_index,
    extract_entities,
    extract_relationships,
    traverse_graph,
)

# DSPy query optimization
from .optimization import (
    DocMindRAG,
    DSPyABTest,
    DSPyConfig,
    DSPyOptimizer,
    OptimizationMode,
    QueryStrategy,
    classify_query_strategy,
    measure_quality_improvement,
    progressive_optimization_pipeline,
)

# Adaptive router query engine
from .query_engine import (
    AdaptiveRouterQueryEngine,
    configure_router_settings,
    create_adaptive_router_engine,
)

# CrossEncoder reranking
from .reranking import (
    BGECrossEncoderRerank,
    benchmark_reranking_latency,
    create_bge_cross_encoder_reranker,
)

# Unified Qdrant vector store
from .vector_store import QdrantUnifiedVectorStore, create_unified_qdrant_store

__all__ = [
    # BGE-M3 Unified Embeddings
    "BGEM3Embedding",
    "create_bgem3_embedding",
    "configure_bgem3_settings",
    # CLIP Multimodal Embeddings
    "ClipConfig",
    "create_clip_embedding",
    "setup_clip_for_llamaindex",
    # CrossEncoder Reranking
    "BGECrossEncoderRerank",
    "create_bge_cross_encoder_reranker",
    "benchmark_reranking_latency",
    # RouterQueryEngine Adaptive Retrieval
    "AdaptiveRouterQueryEngine",
    "create_adaptive_router_engine",
    "configure_router_settings",
    # Qdrant Unified Vector Store
    "QdrantUnifiedVectorStore",
    "create_unified_qdrant_store",
    # Property Graph Configuration
    "PropertyGraphConfig",
    "create_property_graph_index",
    "create_property_graph_index_async",
    "create_tech_schema",
    "extract_entities",
    "extract_relationships",
    "traverse_graph",
    "calculate_entity_confidence",
    "extend_property_graph_index",
    # DSPy Optimization
    "DSPyConfig",
    "DSPyOptimizer",
    "DSPyABTest",
    "DocMindRAG",
    "OptimizationMode",
    "QueryStrategy",
    "classify_query_strategy",
    "measure_quality_improvement",
    "progressive_optimization_pipeline",
]
