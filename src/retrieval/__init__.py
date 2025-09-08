"""Retrieval & Search System.

Implements RouterQueryEngine adaptive retrieval per ADR-003, with
server-side hybrid retrieval (Qdrant Query API) and modality-aware reranking
(ADR-037) using text and visual rerankers.
"""

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

# DSPy query optimization removed in favor of server-side hybrid + reranking
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
    "MultimodalReranker",
    "PropertyGraphConfig",
    "build_text_reranker",
    "build_visual_reranker",
    "calculate_entity_confidence",
    "configure_router_settings",
    "create_adaptive_router_engine",
    "create_property_graph_index",
    "create_property_graph_index_async",
    "create_tech_schema",
    "extend_property_graph_index",
    "extract_entities",
    "extract_relationships",
    "traverse_graph",
]
