"""Retrieval & Search System.

Complete architectural replacement implementing BGE-M3 unified embeddings,
RouterQueryEngine adaptive retrieval, and CrossEncoder reranking per ADR-002,
ADR-003, and ADR-006.
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

# Modality-aware reranking
from .reranking import (
    MultimodalReranker,
    build_text_reranker,
    build_visual_reranker,
)

# Unified Qdrant vector store

__all__ = [
    "AdaptiveRouterQueryEngine",
    "BGEM3Embedding",
    "ClipConfig",
    "DSPyABTest",
    "DSPyConfig",
    "DSPyOptimizer",
    "DocMindRAG",
    "MultimodalReranker",
    "OptimizationMode",
    "PropertyGraphConfig",
    "QueryStrategy",
    "build_text_reranker",
    "build_visual_reranker",
    "calculate_entity_confidence",
    "classify_query_strategy",
    "configure_bgem3_settings",
    "configure_router_settings",
    "create_adaptive_router_engine",
    "create_bgem3_embedding",
    "create_clip_embedding",
    "create_property_graph_index",
    "create_property_graph_index_async",
    "create_tech_schema",
    "extend_property_graph_index",
    "extract_entities",
    "extract_relationships",
    "measure_quality_improvement",
    "progressive_optimization_pipeline",
    "setup_clip_for_llamaindex",
    "traverse_graph",
]
