"""GraphRAG health reporting for the required LlamaIndex dependency."""

from __future__ import annotations

from llama_index.core.indices.property_graph import PropertyGraphIndex


def get_graphrag_health(*, force_refresh: bool = False) -> tuple[bool, str, str]:
    """Return GraphRAG support, backend name, and operator guidance."""
    del force_refresh
    return (
        PropertyGraphIndex is not None,
        "llama_index",
        "LlamaIndex core PropertyGraphIndex is available.",
    )


__all__ = [
    "get_graphrag_health",
]
