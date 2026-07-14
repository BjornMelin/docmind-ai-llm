"""LlamaIndex GraphRAG construction and export helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.type import ResponseMode
from loguru import logger

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


@dataclass(frozen=True)
class GraphQueryArtifacts:
    """Graph retriever and query engine used by the router."""

    retriever: Any
    query_engine: Any


def build_graph_query_engine(
    property_graph_index: Any,
    *,
    llm: Any | None = None,
    include_text: bool = True,
    similarity_top_k: int = 10,
    path_depth: int = 1,
    node_postprocessors: Sequence[Any] | None = None,
) -> GraphQueryArtifacts:
    """Construct graph retrieval directly through LlamaIndex core."""
    if property_graph_index is None or not hasattr(
        property_graph_index, "as_retriever"
    ):
        raise ValueError("property_graph_index must expose as_retriever()")
    retriever = property_graph_index.as_retriever(
        include_text=include_text,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
    )
    engine_kwargs: dict[str, Any] = {
        "retriever": retriever,
        "llm": llm,
        "response_mode": ResponseMode.NO_TEXT,
        "verbose": False,
    }
    if node_postprocessors:
        engine_kwargs["node_postprocessors"] = list(node_postprocessors)
    query_engine = RetrieverQueryEngine.from_args(**engine_kwargs)
    return GraphQueryArtifacts(retriever=retriever, query_engine=query_engine)


def build_graph_retriever(
    property_graph_index: Any,
    *,
    include_text: bool = True,
    similarity_top_k: int = 10,
    path_depth: int = 1,
) -> Any:
    """Return the LlamaIndex graph retriever."""
    if property_graph_index is None or not hasattr(
        property_graph_index, "as_retriever"
    ):
        raise ValueError("property_graph_index must expose as_retriever()")
    return property_graph_index.as_retriever(
        include_text=include_text,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
    )


def get_export_seed_ids(
    property_graph_index: Any | None,
    vector_index: Any | None,
    *,
    cap: int = 32,
) -> list[str]:
    """Derive export seeds via graph retrieval with vector fallback."""
    try:
        if property_graph_index is not None:
            retriever = build_graph_retriever(property_graph_index)
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
    except (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover
        logger.debug(
            "Graph retriever failed to provide export seeds (error_type={})",
            type(exc).__name__,
        )

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
    except (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover
        logger.debug(
            "Vector retriever failed to provide export seeds (error_type={})",
            type(exc).__name__,
        )

    return [str(i) for i in range(max(0, int(cap)))]


def _relation_label(edge: Any) -> str:
    for key in ("label", "type"):
        val = getattr(edge, key, None)
        if val:
            return str(val)
    return "related"


def _node_identifier(node: Any) -> str:
    return str(getattr(node, "id", getattr(node, "name", node)))


def _source_ids(node: Any) -> list[str]:
    for attr in ("source_id", "doc_id", "page_id", "ref_doc_id"):
        value = getattr(node, attr, None)
        if value:
            return [str(value)]
    props = getattr(node, "properties", {}) or {}
    for attr in ("source_id", "doc_id", "page_id", "ref_doc_id"):
        if attr in props:
            return [str(props[attr])]
    return []


def _iter_edges(
    path_nodes: Iterable[Any], max_depth_cap: int
) -> Iterable[tuple[Any, Any, Any | None, int]]:
    items = list(path_nodes)
    if len(items) < 2:
        return
    path_depth = max(1, min(max_depth_cap, len(items) - 1))
    j = 0
    while j < len(items) - 1:
        current = items[j]
        next_item = items[j + 1]
        if j + 2 < len(items) and any(
            hasattr(next_item, key) for key in ("label", "type")
        ):
            yield current, items[j + 2], next_item, path_depth
            j += 2
        else:
            yield current, next_item, None, path_depth
            j += 1


def _build_rel_map_rows(
    *,
    store: Any,
    seed_node_ids: Sequence[str],
    depth: int,
) -> list[dict[str, Any]]:
    """Best-effort conversion of store rel_map to dict rows."""
    if not (hasattr(store, "get") and hasattr(store, "get_rel_map")):
        return []
    try:
        nodes = list(store.get(ids=list(seed_node_ids)))
        rel_paths = list(store.get_rel_map(nodes, depth=depth))
    except (
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:  # pragma: no cover
        logger.debug(
            "get_rel_map(nodes, depth=...) failed (error_type={})",
            type(exc).__name__,
        )
        return []

    rows: list[dict[str, Any]] = []
    for path_idx, rel_path in enumerate(rel_paths):
        try:
            for head, tail, maybe_edge, path_depth in _iter_edges(rel_path, depth):
                relation = (
                    _relation_label(maybe_edge) if maybe_edge is not None else "related"
                )
                sources = list({*(_source_ids(head) + _source_ids(tail))})
                rows.append(
                    {
                        "subject": _node_identifier(head),
                        "relation": relation,
                        "object": _node_identifier(tail),
                        "depth": path_depth,
                        "path_id": path_idx,
                        "source_ids": sources,
                    }
                )
        except TypeError:  # pragma: no cover - defensive
            continue
    return rows


def _render_rel_map_jsonl(
    *,
    store: Any,
    seed_node_ids: Sequence[str],
    depth: int,
) -> list[str]:
    """Best-effort conversion of store rel_map to JSONL lines."""
    import json

    return [
        json.dumps(row, ensure_ascii=False)
        for row in _build_rel_map_rows(
            store=store, seed_node_ids=seed_node_ids, depth=depth
        )
    ]


def export_graph_jsonl(
    property_graph_index: Any,
    *,
    output_path: Path | str,
    depth: int = 2,
    seed_node_ids: Sequence[str] | None = None,
) -> None:
    """Serialize the graph to JSON Lines using LlamaIndex store APIs."""
    store = getattr(property_graph_index, "property_graph_store", None)
    if store is None:
        raise ValueError("property_graph_index must expose property_graph_store")
    node_ids = list(seed_node_ids or [])
    if not node_ids:
        node_ids = get_export_seed_ids(property_graph_index, None, cap=32)

    lines = _render_rel_map_jsonl(store=store, seed_node_ids=node_ids, depth=int(depth))
    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with graph_export_span(
        adapter_name="llama_index",
        fmt="jsonl",
        depth=int(depth),
        seed_count=len(node_ids),
    ) as span:
        path_obj.write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        bytes_written = path_obj.stat().st_size if path_obj.exists() else 0
        record_graph_export_event(span, path=path_obj, bytes_written=bytes_written)


def export_graph_parquet(
    property_graph_index: Any,
    *,
    output_path: Path | str,
    depth: int = 2,
    seed_node_ids: Sequence[str] | None = None,
) -> None:
    """Serialize the graph to Parquet when PyArrow is available."""
    store = getattr(property_graph_index, "property_graph_store", None)
    if store is None:
        raise ValueError("property_graph_index must expose property_graph_store")
    node_ids = list(seed_node_ids or [])
    if not node_ids:
        node_ids = get_export_seed_ids(property_graph_index, None, cap=32)

    path_obj = Path(output_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:  # pragma: no cover - optional dependency
        logger.debug("PyArrow not installed; skipping Parquet export")
        return

    rows = _build_rel_map_rows(store=store, seed_node_ids=node_ids, depth=int(depth))
    if not rows:
        return

    with graph_export_span(
        adapter_name="llama_index",
        fmt="parquet",
        depth=int(depth),
        seed_count=len(node_ids),
    ) as span:
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, str(path_obj))
        bytes_written = path_obj.stat().st_size if path_obj.exists() else 0
        record_graph_export_event(span, path=path_obj, bytes_written=bytes_written)
