"""Retrieval & Search System.

Implements BGE-M3 unified embeddings and RouterQueryEngine adaptive retrieval
per ADR-002 and ADR-003, with modality-aware reranking (ADR-037) using
ColPali (visual) and BGE v2-m3 CrossEncoder (text).
"""

# BGE-M3 and CLIP embeddings (consolidated)
from .bge_m3_index import (
    build_bge_m3_index,
    build_bge_m3_retriever,
    get_default_bge_m3_retriever,
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
    "DSPyABTest",
    "DSPyConfig",
    "DSPyOptimizer",
    "DocMindRAG",
    "MultimodalReranker",
    "OptimizationMode",
    "PropertyGraphConfig",
    "QueryStrategy",
    "build_bge_m3_index",
    "build_bge_m3_retriever",
    "build_text_reranker",
    "build_visual_reranker",
    "calculate_entity_confidence",
    "classify_query_strategy",
    "configure_router_settings",
    "create_adaptive_router_engine",
    "create_property_graph_index",
    "create_property_graph_index_async",
    "create_tech_schema",
    "extend_property_graph_index",
    "extract_entities",
    "extract_relationships",
    "get_default_bge_m3_retriever",
    "measure_quality_improvement",
    "progressive_optimization_pipeline",
    "traverse_graph",
]
