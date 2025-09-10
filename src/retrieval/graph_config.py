"""Property graph configuration and utilities.

This module exposes portable, library-first helpers for constructing and
querying LlamaIndex ``PropertyGraphIndex`` and exporting relations. It relies
exclusively on documented LlamaIndex APIs (e.g., ``as_retriever``,
``as_query_engine``, and ``SimplePropertyGraphStore.get`` / ``get_rel_map``)
to ensure compatibility across graph store backends.

No mutation of ``PropertyGraphIndex`` instances is performed; helper functions
and small wrappers are provided instead. All functions include Google-style
docstrings and type hints.

Changelog:
- 2025-09-09:
    * Legacy confidence scoring removed — consumers should rely on extractor
      confidence metadata where available or downstream validation logic.
    * Legacy count/find helpers removed to ensure portability and API hygiene.
    * Legacy relationship traversal and count helpers removed.
"""

import asyncio
from pathlib import Path
from typing import Any

from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.graph_stores import SimplePropertyGraphStore
from llama_index.core.indices.property_graph import (
    ImplicitPathExtractor,
    SchemaLLMPathExtractor,
    SimpleLLMPathExtractor,
)
from loguru import logger
from pydantic import BaseModel, Field

# Property Graph Configuration Constants
DEFAULT_MAX_PATHS_PER_CHUNK = 20
DEFAULT_NUM_WORKERS = 4
DEFAULT_PATH_DEPTH = 1
DEFAULT_MAX_TRIPLETS_PER_CHUNK = 10
DEFAULT_TRAVERSAL_TIMEOUT = 3.0
DEFAULT_MAX_HOPS = 2


class PropertyGraphConfig(BaseModel):
    """Configuration for PropertyGraphIndex with domain settings.

    Attributes:
        entities: Allowed entity types for extraction.
        relations: Allowed relation types for extraction.
        max_paths_per_chunk: Maximum paths to extract per chunk.
        num_workers: Number of workers for parallel extraction.
        path_depth: Maximum traversal depth for multi-hop queries.
        strict_schema: Whether to enforce strict schema validation.
    """

    entities: list[str] = Field(
        default_factory=lambda: [
            "FRAMEWORK",
            "LIBRARY",
            "MODEL",
            "HARDWARE",
            "PERSON",
            "ORG",
        ],
        description="Entity types for extraction",
    )
    relations: list[str] = Field(
        default_factory=lambda: [
            "USES",
            "OPTIMIZED_FOR",
            "PART_OF",
            "CREATED_BY",
            "SUPPORTS",
        ],
        description="Relationship types for extraction",
    )
    max_paths_per_chunk: int = Field(
        default=DEFAULT_MAX_PATHS_PER_CHUNK,
        description="Maximum paths to extract per chunk",
    )
    num_workers: int = Field(
        default=DEFAULT_NUM_WORKERS,
        description="Number of workers for parallel extraction",
    )
    path_depth: int = Field(
        default=DEFAULT_PATH_DEPTH,
        description="Maximum traversal depth for multi-hop queries",
    )
    strict_schema: bool = Field(
        default=True,
        description="Enforce strict schema validation",
    )


def create_tech_schema() -> dict[str, list[str]]:
    """Create a domain schema for technical documentation.

    Returns:
        dict[str, list[str]]: Dictionary with keys ``entities`` and ``relations``.
    """
    return {
        "entities": ["FRAMEWORK", "LIBRARY", "MODEL", "HARDWARE", "PERSON", "ORG"],
        "relations": ["USES", "OPTIMIZED_FOR", "PART_OF", "CREATED_BY", "SUPPORTS"],
    }


def create_property_graph_index(
    documents: list[Any],
    *,
    schema: dict[str, list[str]] | None = None,
    llm: Any | None = None,
    **kwargs: Any,
) -> PropertyGraphIndex:
    """Create a PropertyGraphIndex with domain-specific extractors.

    This function configures extractors using documented LlamaIndex features and
    avoids mutating the returned index instance. All optional parameters should
    be passed explicitly.

    Args:
        documents: Documents or nodes to build the graph from.
        schema: Domain-specific schema for entities/relations. Defaults to
            ``create_tech_schema()`` when not provided.
        llm: Language model instance for path extraction. Defaults to
            ``Settings.llm`` when not provided.
        **kwargs: Optional parameters:
            vector_store: Optional vector store for hybrid retrieval.
            max_paths_per_chunk: Maximum paths per chunk (default: 20).
            num_workers: Number of parallel workers (default: 4).
            path_depth: Maximum traversal depth for multi-hop queries
                (default: 2).

    Returns:
        PropertyGraphIndex: Configured index instance.
    """
    if schema is None:
        schema = create_tech_schema()

    if llm is None:
        llm = Settings.llm

    logger.info("Creating PropertyGraphIndex with domain extractors")

    # Backward-compatible kwargs handling
    vector_store = kwargs.get("vector_store")
    max_paths_per_chunk = int(
        kwargs.get("max_paths_per_chunk", DEFAULT_MAX_PATHS_PER_CHUNK)
    )
    num_workers = int(kwargs.get("num_workers", DEFAULT_NUM_WORKERS))
    # Note: path_depth is controlled at retrieval-time via retriever

    # Configure extractors - all native LlamaIndex
    kg_extractors = [
        # LLM-based extraction for technical relationships
        SimpleLLMPathExtractor(
            llm=llm,
            max_paths_per_chunk=max_paths_per_chunk,
            num_workers=num_workers,
        ),
        # Schema-guided extraction for consistent entity types
        SchemaLLMPathExtractor(llm=llm),
        # Implicit relationship extraction from structure
        ImplicitPathExtractor(),
    ]

    # Create property graph store and index
    property_graph_store = SimplePropertyGraphStore()
    index = PropertyGraphIndex.from_documents(
        documents,
        kg_extractors=kg_extractors,
        property_graph_store=property_graph_store,
        vector_store=vector_store,
        show_progress=True,
        max_triplets_per_chunk=DEFAULT_MAX_TRIPLETS_PER_CHUNK,
    )

    logger.info("PropertyGraphIndex created with %d documents", len(documents))
    return index


async def create_property_graph_index_async(
    documents: list[Any],
    *,
    schema: dict[str, list[str]] | None = None,
    llm: Any | None = None,
    **kwargs: Any,
) -> PropertyGraphIndex:
    """Create a PropertyGraphIndex asynchronously.

    Offloads the synchronous constructor via ``asyncio.to_thread`` to avoid
    blocking the event loop.

    Args:
        documents: Documents or nodes to build the graph from.
        schema: Optional domain schema.
        llm: Optional LLM for extraction.
        **kwargs: Same as :func:`create_property_graph_index`.

    Returns:
        PropertyGraphIndex: Configured index instance.
    """
    return await asyncio.to_thread(
        create_property_graph_index, documents, schema=schema, llm=llm, **kwargs
    )


def create_graph_rag_components(
    index: PropertyGraphIndex,
    *,
    llm: Any | None = None,
    include_text: bool = True,
    similarity_top_k: int = 10,
    path_depth: int = DEFAULT_PATH_DEPTH,
) -> dict[str, Any]:
    """Create GraphRAG components from an existing ``PropertyGraphIndex``.

    This factory returns a minimal set of components using documented LlamaIndex
    APIs only. It exposes the underlying property graph store, a query engine
    derived from the index, and a retriever configured for graph-aware queries.

    Args:
        index: A configured ``PropertyGraphIndex``.
        llm: Optional LLM override for the query engine; when omitted, index
            defaults are used.
        include_text: Whether to include source text in query engine responses.
        similarity_top_k: Top-K nodes for retriever.
        path_depth: Maximum path depth for graph-aware retrieval.

    Returns:
        dict[str, Any]: A mapping with keys ``graph_store``, ``query_engine``,
        and ``retriever``.

    Raises:
        ValueError: If the index has no ``property_graph_store``.
    """
    store = getattr(index, "property_graph_store", None)
    if store is None:
        raise ValueError("PropertyGraphIndex has no property_graph_store")

    query_engine = index.as_query_engine(include_text=include_text, llm=llm)
    retriever = index.as_retriever(
        include_text=False, similarity_top_k=similarity_top_k, path_depth=path_depth
    )
    return {"graph_store": store, "query_engine": query_engine, "retriever": retriever}


async def extract_entities(
    index: PropertyGraphIndex, seed_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Extract entities reachable from seeds using documented store APIs.

    This function derives entities via the graph store using `get` and
    `get_rel_map`. When no seeds are provided, it returns an empty list. To
    obtain seeds, call ``index.as_retriever(...).retrieve(query)`` and derive
    node identifiers from the result nodes.

    Args:
        index: PropertyGraphIndex instance.
        seed_ids: Optional list of node identifiers to seed traversal.

    Returns:
        list[dict[str, Any]]: List of entity dicts with `id` and `name` keys.
    """
    store = getattr(index, "property_graph_store", None)
    if store is None or not seed_ids:
        return []
    # Fetch nodes by id using documented API
    nodes = await asyncio.to_thread(lambda: list(store.get(ids=seed_ids)))
    entities: list[dict[str, Any]] = []
    for node in nodes:
        name = getattr(node, "name", str(getattr(node, "id", "")))
        entities.append({"id": str(getattr(node, "id", name)), "name": str(name)})
    return entities


async def extract_relationships(
    index: PropertyGraphIndex,
    seed_ids: list[str] | None = None,
    max_hops: int = DEFAULT_MAX_HOPS,
) -> list[dict[str, str]]:
    """Extract relationships as edges from the relation map.

    Uses `property_graph_store.get_rel_map` to construct a list of edges
    (``head``, ``relation``, ``tail``). If relation labels are not available,
    the placeholder ``related`` is used.

    Args:
        index: PropertyGraphIndex instance.
        seed_ids: Optional list of seed node ids; when omitted returns empty list.
        max_hops: Maximum traversal depth.

    Returns:
        list[dict[str, str]]: Edges with keys `head`, `relation`, `tail`.
    """
    store = getattr(index, "property_graph_store", None)
    if store is None or not seed_ids:
        return []
    # Fetch seed nodes and build relation map
    nodes = await asyncio.to_thread(lambda: list(store.get(ids=seed_ids)))
    rel_paths = await asyncio.to_thread(
        lambda: list(store.get_rel_map(nodes, depth=max_hops))
    )
    edges: list[dict[str, str]] = []
    for path in rel_paths:
        # Interpret each path as a sequence of nodes; derive pairwise edges
        try:
            items = list(path)
        except TypeError:
            continue
        for i in range(len(items) - 1):
            head = str(getattr(items[i], "id", getattr(items[i], "name", items[i])))
            tail = str(
                getattr(items[i + 1], "id", getattr(items[i + 1], "name", items[i + 1]))
            )
            edges.append({"head": head, "relation": "related", "tail": tail})
    return edges


async def traverse_graph(
    index: PropertyGraphIndex,
    query: str,
    max_depth: int = DEFAULT_PATH_DEPTH,
    timeout: float = DEFAULT_TRAVERSAL_TIMEOUT,
) -> list[Any]:
    """Retrieve graph-aware nodes for a query.

    Thin wrapper over `index.as_retriever(...).retrieve(query)` executed in a
    thread to avoid blocking the event loop. Returns the retriever's node list
    or an empty list on timeout.

    Args:
        index: PropertyGraphIndex instance.
        query: Natural language query string.
        max_depth: Maximum path depth for retrieval.
        timeout: Timeout in seconds for the retrieval.

    Returns:
        list[Any]: Retrieved nodes (implementation-specific node types).
    """
    retriever = index.as_retriever(include_text=False, path_depth=max_depth)
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(retriever.retrieve, query), timeout=timeout
        )
    except TimeoutError:
        logger.warning("Graph traversal timed out after %.1fs", timeout)
        return []


# Note: Legacy helpers relying on undocumented graph store internals were removed
# to ensure portability and API hygiene. Avoid usage of `get_nodes`/`get_edges`
# in production code; rely on `get`/`get_rel_map` documented APIs instead.


def get_export_seed_ids(
    pg_index: Any | None,
    vector_index: Any | None,
    *,
    cap: int = 32,
) -> list[str]:
    """Derive export seed IDs using retrievers with deterministic fallback.

    Order of preference:
    1) PropertyGraphIndex.as_retriever(similarity_top_k=cap, path_depth=1)
    2) VectorStoreIndex.as_retriever(similarity_top_k=cap)
    3) Deterministic fallback: ["0", "1", ..., str(cap-1)]
    """
    try:
        if pg_index is not None:
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
        import logging

        logging.getLogger(__name__).debug(
            "Failed deriving seed IDs from property graph", exc_info=True
        )

    try:
        if vector_index is not None:
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
        import logging

        logging.getLogger(__name__).debug(
            "Failed deriving seed IDs from vector index", exc_info=True
        )

    return [str(i) for i in range(max(0, int(cap)))]


def export_graph_jsonl(
    index: PropertyGraphIndex,
    path: Path,
    seed_ids: list[str] | None = None,
    depth: int = DEFAULT_PATH_DEPTH,
) -> None:
    """Export relation edges to JSONL using `get_rel_map`.

    Args:
        index: PropertyGraphIndex to export from.
        path: Output JSONL file path.
        seed_ids: Optional list of seed node ids. When omitted, no export occurs.
        depth: Maximum traversal depth for relation map.
    """
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    store = getattr(index, "property_graph_store", None)
    if store is None or not seed_ids:
        logger.warning("JSONL export skipped: missing store or seeds")
        return
    try:
        nodes = list(store.get(ids=seed_ids))
        rel_paths = list(store.get_rel_map(nodes, depth=depth))
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning("JSONL export failed to build rel_map: %s", exc)
        return

    def _sources_for(node: Any) -> list[str]:
        # Attempt to derive source ids from common fields
        for key in ("source_id", "doc_id", "page_id", "ref_doc_id"):
            val = getattr(node, key, None)
            if val:
                return [str(val)]
        props = getattr(node, "properties", {}) or {}
        for key in ("source_id", "doc_id", "page_id", "ref_doc_id"):
            if key in props:
                return [str(props[key])]
        return []

    def _relation_label(_a: Any, b: Any) -> str:
        # Best-effort: some stores return triplets or edges with a label
        # Attempt to read 'label' or 'type' on intermediate relation if present
        # Fallback to 'related'.
        for key in ("label", "type"):
            val = getattr(b, key, None)
            if val:
                return str(val)
        return "related"

    with out.open("w", encoding="utf-8") as f:
        for path_idx, path_nodes in enumerate(rel_paths):
            try:
                items = list(path_nodes)
            except TypeError:
                continue
            j = 0
            while j < len(items) - 1:
                a = items[j]
                # Triplet pattern: [node, relation, node]
                if j + 2 < len(items) and any(
                    hasattr(items[j + 1], k) for k in ("label", "type")
                ):
                    b_rel = items[j + 1]
                    c = items[j + 2]
                    subj = str(getattr(a, "id", getattr(a, "name", a)))
                    obj = str(getattr(c, "id", getattr(c, "name", c)))
                    rel = _relation_label(a, b_rel)
                    row = {
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                        "depth": min(depth, len(items) - 1),
                        "path_id": path_idx,
                        "source_ids": list({*(_sources_for(a) + _sources_for(c))}),
                    }
                    f.write(json.dumps(row) + "\n")
                    j += 2
                else:
                    b = items[j + 1]
                    subj = str(getattr(a, "id", getattr(a, "name", a)))
                    obj = str(getattr(b, "id", getattr(b, "name", b)))
                    rel = _relation_label(a, b)
                    row = {
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                        "depth": min(depth, len(items) - 1),
                        "path_id": path_idx,
                        "source_ids": list({*(_sources_for(a) + _sources_for(b))}),
                    }
                    f.write(json.dumps(row) + "\n")
                    j += 1


def export_graph_parquet(
    index: PropertyGraphIndex,
    path: Path,
    seed_ids: list[str] | None = None,
    depth: int = DEFAULT_PATH_DEPTH,
) -> None:
    """Export relation edges to Parquet using `get_rel_map`.

    Args:
        index: PropertyGraphIndex to export from.
        path: Output Parquet file path.
        seed_ids: Optional list of seed node ids. When omitted, no export occurs.
        depth: Maximum traversal depth for relation map.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional
        logger.warning("PyArrow not available, skipping Parquet export: %s", exc)
        return

    store = getattr(index, "property_graph_store", None)
    if store is None or not seed_ids:
        logger.warning("Parquet export skipped: missing store or seeds")
        return
    try:
        nodes = list(store.get(ids=seed_ids))
        rel_paths = list(store.get_rel_map(nodes, depth=depth))
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning("Parquet export failed to build rel_map: %s", exc)
        return

    def _relation_label(_a: Any, b: Any) -> str:
        for key in ("label", "type"):
            val = getattr(b, key, None)
            if val:
                return str(val)
        return "related"

    rows: list[dict[str, Any]] = []
    for path_idx, path_nodes in enumerate(rel_paths):
        try:
            items = list(path_nodes)
        except TypeError:
            continue
        j = 0
        while j < len(items) - 1:
            a = items[j]
            if j + 2 < len(items) and any(
                hasattr(items[j + 1], k) for k in ("label", "type")
            ):
                b_rel = items[j + 1]
                c = items[j + 2]
                subj = str(getattr(a, "id", getattr(a, "name", a)))
                obj = str(getattr(c, "id", getattr(c, "name", c)))
                rel = _relation_label(a, b_rel)
                rows.append(
                    {
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                        "depth": min(depth, len(items) - 1),
                        "path_id": path_idx,
                    }
                )
                j += 2
            else:
                b = items[j + 1]
                subj = str(getattr(a, "id", getattr(a, "name", a)))
                obj = str(getattr(b, "id", getattr(b, "name", b)))
                rel = _relation_label(a, b)
                rows.append(
                    {
                        "subject": subj,
                        "relation": rel,
                        "object": obj,
                        "depth": min(depth, len(items) - 1),
                        "path_id": path_idx,
                    }
                )
                j += 1

    if not rows:
        logger.warning("No edges to export; Parquet file will not be created")
        return
    table = pa.Table.from_pylist(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out))
