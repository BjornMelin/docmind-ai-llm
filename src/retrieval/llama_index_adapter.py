"""GraphRAG health reporting for the required LlamaIndex dependency."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal


@dataclass(frozen=True, slots=True)
class GraphRAGHealth:
    """Import-light installation or action-time GraphRAG capability state."""

    status: Literal["installed", "ready", "unavailable"]
    adapter_name: str
    hint: str


def get_graphrag_health(*, force_refresh: bool = False) -> GraphRAGHealth:
    """Return import-light GraphRAG installation health and operator guidance."""
    try:
        installed_version = version("llama-index-core")
    except PackageNotFoundError:
        return GraphRAGHealth(
            status="unavailable",
            adapter_name="unavailable",
            hint="The required llama-index-core distribution is not installed.",
        )
    if not force_refresh:
        return GraphRAGHealth(
            status="installed",
            adapter_name="llama_index",
            hint=f"llama-index-core {installed_version} is installed; GraphRAG "
            "APIs are validated when graph work begins.",
        )
    try:
        from llama_index.core.indices.property_graph import PropertyGraphIndex
    except (ImportError, AttributeError):
        return GraphRAGHealth(
            status="unavailable",
            adapter_name="unavailable",
            hint="The installed llama-index-core lacks required GraphRAG APIs.",
        )
    _ = PropertyGraphIndex
    return GraphRAGHealth(
        status="ready",
        adapter_name="llama_index",
        hint=f"llama-index-core {installed_version} PropertyGraphIndex is available.",
    )


__all__ = [
    "GraphRAGHealth",
    "get_graphrag_health",
]
