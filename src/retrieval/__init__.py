"""Retrieval & Search System.

Implements RouterQueryEngine adaptive retrieval per ADR-003, with
server-side hybrid retrieval (Qdrant Query API) and modality-aware reranking
(ADR-037) using text and visual rerankers.
"""

# Property graph configuration
from .graph_config import (
    GraphQueryArtifacts,
    build_graph_query_engine,
    build_graph_retriever,
    export_graph_jsonl,
    export_graph_parquet,
    get_export_seed_ids,
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
    "GraphQueryArtifacts",
    "MultimodalReranker",
    "ServerHybridRetriever",
    "build_graph_query_engine",
    "build_graph_retriever",
    "build_router_engine",
    "build_text_reranker",
    "build_visual_reranker",
    "export_graph_jsonl",
    "export_graph_parquet",
    "get_export_seed_ids",
]
