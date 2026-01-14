"""Retrieval & Search System.

This package contains router construction, hybrid retrieval, reranking, and
GraphRAG helpers. Keep package import side-effects minimal: importing any
submodule executes this ``__init__`` first, so we lazily expose convenience
symbols to avoid pulling heavy dependencies during unrelated imports.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from src.retrieval.graph_config import (
        GraphQueryArtifacts,
        build_graph_query_engine,
        build_graph_retriever,
        export_graph_jsonl,
        export_graph_parquet,
        get_export_seed_ids,
    )
    from src.retrieval.hybrid import ServerHybridRetriever
    from src.retrieval.reranking import (
        MultimodalReranker,
        build_text_reranker,
        build_visual_reranker,
    )
    from src.retrieval.router_factory import build_router_engine

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

_EXPORTS: dict[str, tuple[str, str]] = {
    "GraphQueryArtifacts": (".graph_config", "GraphQueryArtifacts"),
    "build_graph_query_engine": (".graph_config", "build_graph_query_engine"),
    "build_graph_retriever": (".graph_config", "build_graph_retriever"),
    "export_graph_jsonl": (".graph_config", "export_graph_jsonl"),
    "export_graph_parquet": (".graph_config", "export_graph_parquet"),
    "get_export_seed_ids": (".graph_config", "get_export_seed_ids"),
    "ServerHybridRetriever": (".hybrid", "ServerHybridRetriever"),
    "MultimodalReranker": (".reranking", "MultimodalReranker"),
    "build_text_reranker": (".reranking", "build_text_reranker"),
    "build_visual_reranker": (".reranking", "build_visual_reranker"),
    "build_router_engine": (".router_factory", "build_router_engine"),
}


def __getattr__(name: str) -> Any:
    """Lazily import heavy exports on first access."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name, package=__name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Customize dir() to include all exports."""
    return sorted(set(list(globals()) + __all__))
