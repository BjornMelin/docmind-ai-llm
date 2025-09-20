"""GraphRAG helpers orchestrated through the adapter registry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.retrieval.adapter_registry import (
    MissingGraphAdapterError,
    ensure_default_adapter,
    get_adapter,
)
from src.retrieval.adapters.protocols import (
    AdapterFactoryProtocol,
    GraphQueryArtifacts,
    GraphRetrieverProtocol,
)
from src.telemetry.opentelemetry import (
    graph_export_span,
    record_graph_export_event,
)

__all__ = [
    "GraphQueryArtifacts",
    "build_graph_query_engine",
    "build_graph_retriever",
    "export_graph_jsonl",
    "export_graph_parquet",
    "get_export_seed_ids",
]


def _resolve_adapter(adapter: AdapterFactoryProtocol | None) -> AdapterFactoryProtocol:
    if adapter is not None:
        return adapter
    ensure_default_adapter()
    return get_adapter()


def build_graph_query_engine(
    property_graph_index: Any,
    *,
    llm: Any | None = None,
    include_text: bool = True,
    similarity_top_k: int = 10,
    path_depth: int = 1,
    node_postprocessors: Sequence[Any] | None = None,
    response_mode: str = "compact",
    adapter: AdapterFactoryProtocol | None = None,
) -> GraphQueryArtifacts:
    """Construct graph retriever/query engine via the active adapter."""
    factory = _resolve_adapter(adapter)
    return factory.build_graph_artifacts(
        property_graph_index=property_graph_index,
        llm=llm,
        include_text=include_text,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
        node_postprocessors=node_postprocessors,
        response_mode=response_mode,
    )


def build_graph_retriever(
    property_graph_index: Any,
    *,
    include_text: bool = True,
    similarity_top_k: int = 10,
    path_depth: int = 1,
    adapter: AdapterFactoryProtocol | None = None,
) -> GraphRetrieverProtocol:
    """Return the graph retriever from the configured adapter."""
    artifacts = build_graph_query_engine(
        property_graph_index,
        include_text=include_text,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
        adapter=adapter,
    )
    return artifacts.retriever


@dataclass(frozen=True)
class _ExportContext:
    property_graph_index: Any
    retriever: GraphRetrieverProtocol


def _prepare_export_context(
    property_graph_index: Any,
    *,
    adapter: AdapterFactoryProtocol | None = None,
) -> _ExportContext:
    try:
        artifacts = build_graph_query_engine(
            property_graph_index,
            adapter=adapter,
        )
    except (MissingGraphAdapterError, ValueError) as exc:
        raise ValueError(f"Graph export requires a valid adapter: {exc}") from exc
    return _ExportContext(
        property_graph_index=property_graph_index, retriever=artifacts.retriever
    )


def _collect_seed_ids(
    *,
    retriever: GraphRetrieverProtocol | None,
    property_graph_index: Any | None,
    vector_index: Any | None,
    adapter: AdapterFactoryProtocol | None,
    cap: int,
) -> list[str]:
    if retriever is not None:
        try:
            nodes = retriever.retrieve("seed")
            out: list[str] = []
            seen: set[str] = set()
            for node_with_score in nodes:
                node = getattr(node_with_score, "node", None)
                node_id = str(getattr(node, "id_", getattr(node, "id", "")))
                if node_id and node_id not in seen:
                    out.append(node_id)
                    seen.add(node_id)
                if len(out) >= cap:
                    return out
            if out:
                return out
        except Exception:  # pragma: no cover - adapter failures logged for debugging
            logger.debug(
                "Adapter retriever failed to provide export seeds", exc_info=True
            )
    return get_export_seed_ids(
        property_graph_index, vector_index, cap=cap, adapter=adapter
    )


def export_graph_jsonl(
    property_graph_index: Any,
    *,
    output_path: Path | str,
    depth: int = 2,
    seed_node_ids: Sequence[str] | None = None,
    adapter: AdapterFactoryProtocol | None = None,
) -> None:
    """Serialize the graph to JSON Lines using adapter-provided artefacts."""
    ctx = _prepare_export_context(property_graph_index, adapter=adapter)
    store = getattr(property_graph_index, "property_graph_store", None)
    if store is None:
        raise ValueError("property_graph_index must expose property_graph_store")
    node_ids = list(seed_node_ids or [])
    if not node_ids:
        node_ids = _collect_seed_ids(
            retriever=ctx.retriever,
            property_graph_index=property_graph_index,
            vector_index=None,
            adapter=adapter,
            cap=32,
        )
    rel_map = store.get_rel_map(node_ids=node_ids, depth=depth)
    path_obj = Path(output_path)
    active_adapter = _resolve_adapter(adapter)
    with graph_export_span(
        adapter_name=active_adapter.name,
        fmt="jsonl",
        depth=depth,
        seed_count=len(node_ids),
    ) as span:
        path_obj.write_text("\n".join(rel_map), encoding="utf-8")
        bytes_written = path_obj.stat().st_size
        record_graph_export_event(
            span,
            path=path_obj,
            bytes_written=bytes_written,
        )
    telemetry = active_adapter.get_telemetry_hooks()
    telemetry.graph_exported(
        adapter_name=active_adapter.name,
        fmt="jsonl",
        bytes_written=bytes_written,
        depth=depth,
        seed_count=len(node_ids),
    )


def export_graph_parquet(
    property_graph_index: Any,
    *,
    output_path: Path | str,
    depth: int = 2,
    seed_node_ids: Sequence[str] | None = None,
    adapter: AdapterFactoryProtocol | None = None,
) -> None:
    """Serialize the graph to Parquet using the adapter registry."""
    ctx = _prepare_export_context(property_graph_index, adapter=adapter)
    store = getattr(property_graph_index, "property_graph_store", None)
    if store is None or not hasattr(store, "store_rel_map_df"):
        raise ValueError(
            "property_graph_index must expose property_graph_store.store_rel_map_df"
        )
    node_ids = list(seed_node_ids or [])
    if not node_ids:
        node_ids = _collect_seed_ids(
            retriever=ctx.retriever,
            property_graph_index=property_graph_index,
            vector_index=None,
            adapter=adapter,
            cap=32,
        )
    df = store.store_rel_map_df(node_ids=node_ids, depth=depth)
    path_obj = Path(output_path)
    active_adapter = _resolve_adapter(adapter)
    with graph_export_span(
        adapter_name=active_adapter.name,
        fmt="parquet",
        depth=depth,
        seed_count=len(node_ids),
    ) as span:
        df.to_parquet(path_obj)
        bytes_written = path_obj.stat().st_size
        record_graph_export_event(
            span,
            path=path_obj,
            bytes_written=bytes_written,
        )
    telemetry = active_adapter.get_telemetry_hooks()
    telemetry.graph_exported(
        adapter_name=active_adapter.name,
        fmt="parquet",
        bytes_written=bytes_written,
        depth=depth,
        seed_count=len(node_ids),
    )


def get_export_seed_ids(
    property_graph_index: Any | None,
    vector_index: Any | None,
    *,
    cap: int = 32,
    adapter: AdapterFactoryProtocol | None = None,
) -> list[str]:
    """Derive export seeds via adapter retrievers with vector fallback."""
    try:
        if property_graph_index is not None:
            retriever = build_graph_retriever(property_graph_index, adapter=adapter)
            nodes = retriever.retrieve("seed")
            out: list[str] = []
            seen: set[str] = set()
            for node_with_score in nodes:
                node = getattr(node_with_score, "node", None)
                node_id = str(getattr(node, "id_", getattr(node, "id", "")))
                if node_id and node_id not in seen:
                    out.append(node_id)
                    seen.add(node_id)
                if len(out) >= cap:
                    return out
            if out:
                return out
    except Exception:  # pragma: no cover - adapter failures are logged for visibility
        logger.debug("Graph retriever failed to provide export seeds", exc_info=True)

    try:
        if vector_index is not None and hasattr(vector_index, "as_retriever"):
            retr = vector_index.as_retriever(similarity_top_k=cap)
            nodes = retr.retrieve("seed")
            out = []
            seen_vec: set[str] = set()
            for node_with_score in nodes:
                node = getattr(node_with_score, "node", None)
                node_id = str(getattr(node, "id_", getattr(node, "id", "")))
                if node_id and node_id not in seen_vec:
                    out.append(node_id)
                    seen_vec.add(node_id)
                if len(out) >= cap:
                    return out
            if out:
                return out
    except Exception:  # pragma: no cover - fallback path
        logger.debug("Vector retriever failed to provide export seeds", exc_info=True)

    return [str(idx) for idx in range(cap)]
