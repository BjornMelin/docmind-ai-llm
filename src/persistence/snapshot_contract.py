"""Import-light persisted snapshot shape owned by DocMind."""

from __future__ import annotations

PROPERTY_GRAPH_NATIVE_PATHS = frozenset(
    {
        "graph/default__vector_store.json",
        "graph/docstore.json",
        "graph/graph_store.json",
        "graph/image__vector_store.json",
        "graph/index_store.json",
        "graph/property_graph_store.json",
    }
)

__all__ = ["PROPERTY_GRAPH_NATIVE_PATHS"]
