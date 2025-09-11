"""Retrieval & Search System.

Implements RouterQueryEngine adaptive retrieval per ADR-003, with
server-side hybrid retrieval (Qdrant Query API) and modality-aware reranking
(ADR-037) using text and visual rerankers.
"""

# Property graph configuration
from .graph_config import (
    PropertyGraphConfig,
    create_graph_rag_components,
    create_property_graph_index,
    create_property_graph_index_async,
    create_tech_schema,
    export_graph_jsonl,
    export_graph_parquet,
    extract_entities,
    extract_relationships,
    traverse_graph,
)
from .hybrid import ServerHybridRetriever

# Note: DSPy query optimization is handled in the agents layer
# (see src/dspy_integration.py). The retrieval package stays
# library-first (server-side hybrid + reranking) without direct DSPy deps.
# Modality-aware reranking
from .reranking import (
    MultimodalReranker,
    build_text_reranker,
    build_visual_reranker,
)

# Router factory
from .router_factory import build_router_engine

# Unified Qdrant vector store

__all__ = [
    "MultimodalReranker",
    "PropertyGraphConfig",
    "ServerHybridRetriever",
    "build_router_engine",
    "build_text_reranker",
    "build_visual_reranker",
    "create_graph_rag_components",
    "create_property_graph_index",
    "create_property_graph_index_async",
    "create_tech_schema",
    "export_graph_jsonl",
    "export_graph_parquet",
    "extract_entities",
    "extract_relationships",
    "traverse_graph",
]
