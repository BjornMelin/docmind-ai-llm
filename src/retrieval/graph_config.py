"""GraphRAG helpers using documented LlamaIndex APIs.

This module intentionally keeps the surface area small:

* ``build_graph_retriever``/``build_graph_query_engine`` wrap
  ``PropertyGraphIndex`` helpers exposed by LlamaIndex to construct
  retrievers/query engines without any custom behaviour.
* ``get_export_seed_ids`` derives export seeds using library retrievers with
  deterministic fallback.
* ``export_graph_jsonl`` and ``export_graph_parquet`` serialize relation paths
  via ``property_graph_store.get_rel_map`` for reproducible snapshot exports.

All functions include Google-style docstrings and avoid bespoke state. When
LlamaIndex is unavailable, helper functions raise ``ValueError`` so callers can
fail fast instead of silently degrading behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from llama_index.core import PropertyGraphIndex
    from llama_index.core.query_engine import RetrieverQueryEngine
else:  # Library is optional during import; runtime checks guard usage.
    PropertyGraphIndex = Any  # type: ignore
    RetrieverQueryEngine = Any  # type: ignore

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
    """Container holding graph retriever and query engine instances."""

    retriever: Any
    query_engine: Any


def _require_llamaindex() -> None:
    """Raise ValueError when LlamaIndex components are unavailable."""
    if RetrieverQueryEngine is Any:  # type: ignore[str-bytes-safe]
        raise ValueError(
            "LlamaIndex is required for GraphRAG operations but is not installed"
        )


def build_graph_retriever(
    pg_index: Any,
    *,
    include_text: bool = True,
    similarity_top_k: int = 10,
    path_depth: int = 1,
) -> Any:
    """Return a graph-aware retriever using documented LlamaIndex APIs.

    Args:
        pg_index: ``PropertyGraphIndex`` instance.
        include_text: Whether retrieved nodes should include source text.
        similarity_top_k: Number of nodes to retrieve per query.
        path_depth: Maximum relation depth to traverse.

    Returns:
        Retriever instance produced by ``PropertyGraphIndex.as_retriever``.

    Raises:
        ValueError: If ``pg_index`` does not expose the expected API.
    """
    if pg_index is None or not hasattr(pg_index, "as_retriever"):
        raise ValueError("PropertyGraphIndex with as_retriever is required")
    return pg_index.as_retriever(
        include_text=include_text,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
    )


def build_graph_query_engine(
    pg_index: Any,
    *,
    llm: Any | None = None,
    include_text: bool = True,
    similarity_top_k: int = 10,
    path_depth: int = 1,
    node_postprocessors: Sequence[Any] | None = None,
    response_mode: str = "compact",
) -> GraphQueryArtifacts:
    """Construct a ``RetrieverQueryEngine`` for GraphRAG queries.

    Args:
        pg_index: ``PropertyGraphIndex`` instance.
        llm: Optional LLM override passed to the query engine.
        include_text: Whether retrieved nodes should include source text.
        similarity_top_k: Number of nodes to retrieve.
        path_depth: Maximum relation depth to traverse.
        node_postprocessors: Optional node postprocessors to pass to
            ``RetrieverQueryEngine.from_args``.
        response_mode: Response mode forwarded to the query engine.

    Returns:
        GraphQueryArtifacts containing the retriever and query engine.

    Raises:
        ValueError: If LlamaIndex is unavailable or the supplied index lacks the
            required API surface (``as_retriever``).
    """
    _require_llamaindex()
    retriever = build_graph_retriever(
        pg_index,
        include_text=include_text,
        similarity_top_k=similarity_top_k,
        path_depth=path_depth,
    )

    engine_kwargs: dict[str, Any] = {
        "retriever": retriever,
        "llm": llm,
        "response_mode": response_mode,
        "verbose": False,
    }
    if node_postprocessors:
        engine_kwargs["node_postprocessors"] = list(node_postprocessors)

    try:
        query_engine = RetrieverQueryEngine.from_args(**engine_kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to build graph query engine: {exc}") from exc

    return GraphQueryArtifacts(retriever=retriever, query_engine=query_engine)


def get_export_seed_ids(
    pg_index: Any | None,
    vector_index: Any | None,
    *,
    cap: int = 32,
) -> list[str]:
    """Derive seed IDs for exports via documented retrievers.

    Order of preference:
    1) ``PropertyGraphIndex.as_retriever`` (graph seeds)
    2) ``VectorStoreIndex.as_retriever`` (semantic seeds)
    3) Deterministic fallback ``["0", "1", ...]``
    """
    try:
        if pg_index is not None and hasattr(pg_index, "as_retriever"):
            retr = pg_index.as_retriever(
                include_text=False, path_depth=1, similarity_top_k=cap
            )
            nodes = retr.retrieve("seed")
            out: list[str] = []
            seen: set[str] = set()
            for nws in nodes:
                nid = str(
                    getattr(getattr(nws, "node", object()), "id_", None)
                    or getattr(getattr(nws, "node", object()), "id", "")
                )
                if nid and nid not in seen:
                    out.append(nid)
                    seen.add(nid)
                if len(out) >= cap:
                    return out
            if out:
                return out
    except (RuntimeError, ValueError, TypeError, AttributeError):  # pragma: no cover
        logger.debug("Failed deriving seed IDs from property graph", exc_info=True)

    try:
        if vector_index is not None and hasattr(vector_index, "as_retriever"):
            retr = vector_index.as_retriever(similarity_top_k=cap)
            nodes = retr.retrieve("seed")
            out = []
            seen: set[str] = set()
            for nws in nodes:
                nid = str(
                    getattr(getattr(nws, "node", object()), "id_", None)
                    or getattr(getattr(nws, "node", object()), "id", "")
                )
                if nid and nid not in seen:
                    out.append(nid)
                    seen.add(nid)
                if len(out) >= cap:
                    return out
            if out:
                return out
    except (RuntimeError, ValueError, TypeError, AttributeError):  # pragma: no cover
        logger.debug("Failed deriving seed IDs from vector index", exc_info=True)

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


def export_graph_jsonl(
    index: Any,
    path: Path,
    seed_ids: list[str] | None = None,
    depth: int = 1,
) -> None:
    """Export relation paths to newline-delimited JSON.

    Args:
        index: Property graph index instance.
        path: Destination JSONL file.
        seed_ids: Optional list of node identifiers to seed traversal.
        depth: Maximum traversal depth.
    """
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = getattr(index, "property_graph_store", None)
    if store is None or not seed_ids:
        logger.warning("JSONL export skipped: missing store or seeds")
        return
    if not hasattr(store, "get") or not hasattr(store, "get_rel_map"):
        out.write_text("[]\n", encoding="utf-8")
        return

    try:
        nodes = list(store.get(ids=seed_ids))
        rel_paths = list(store.get_rel_map(nodes, depth=depth))
    except (
        AttributeError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning("JSONL export failed to build rel_map: %s", exc)
        return

    with out.open("w", encoding="utf-8") as handle:
        for path_idx, rel_path in enumerate(rel_paths):
            try:
                for head, tail, maybe_edge, path_depth in _iter_edges(rel_path, depth):
                    relation = (
                        _relation_label(maybe_edge)
                        if maybe_edge is not None
                        else "related"
                    )
                    sources = list({*(_source_ids(head) + _source_ids(tail))})
                    row = {
                        "subject": _node_identifier(head),
                        "relation": relation,
                        "object": _node_identifier(tail),
                        "depth": path_depth,
                        "path_id": path_idx,
                        "source_ids": sources,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            except TypeError:  # pragma: no cover - defensive
                continue


def export_graph_parquet(
    index: Any,
    path: Path,
    seed_ids: list[str] | None = None,
    depth: int = 1,
) -> None:
    """Export relation paths to Parquet.

    Args:
        index: Property graph index instance.
        path: Destination Parquet file.
        seed_ids: Optional list of node identifiers to seed traversal.
        depth: Maximum traversal depth.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        logger.warning("PyArrow not available, skipping Parquet export: %s", exc)
        return

    store = getattr(index, "property_graph_store", None)
    if store is None or not seed_ids:
        logger.warning("Parquet export skipped: missing store or seeds")
        return
    if not hasattr(store, "get") or not hasattr(store, "get_rel_map"):
        return

    try:
        nodes = list(store.get(ids=seed_ids))
        rel_paths = list(store.get_rel_map(nodes, depth=depth))
    except (
        AttributeError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning("Parquet export failed to build rel_map: %s", exc)
        return

    rows: list[dict[str, Any]] = []
    for path_idx, rel_path in enumerate(rel_paths):
        try:
            for head, tail, maybe_edge, path_depth in _iter_edges(rel_path, depth):
                relation = (
                    _relation_label(maybe_edge) if maybe_edge is not None else "related"
                )
                rows.append(
                    {
                        "subject": _node_identifier(head),
                        "relation": relation,
                        "object": _node_identifier(tail),
                        "depth": path_depth,
                        "path_id": path_idx,
                    }
                )
        except TypeError:  # pragma: no cover - defensive
            continue

    if not rows:
        logger.warning("No edges produced for Parquet export")
        return

    table = pa.Table.from_pylist(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out))
