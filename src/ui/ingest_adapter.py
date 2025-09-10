"""UI ingestion adapter.

This module provides a thin adapter between Streamlit file uploads and the
document processing pipeline. It saves uploaded files to disk, runs the
asynchronous processing pipeline, builds/updates the vector index in Qdrant
for hybrid retrieval, and records lightweight analytics.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from llama_index.core import Document as LIDocument
from llama_index.core import StorageContext, VectorStoreIndex

from src.config.settings import settings
from src.core.analytics import AnalyticsConfig, AnalyticsManager
from src.processing.document_processor import DocumentProcessor
from src.retrieval.graph_config import (
    create_property_graph_index,
    export_graph_jsonl,
    export_graph_parquet,
    get_export_seed_ids,
)
from src.utils.storage import create_vector_store
from src.utils.telemetry import log_jsonl


def _save_uploaded_file(file: Any) -> Path:
    """Persist a Streamlit uploaded file to the uploads directory.

    Args:
        file: Uploaded file-like object with `name` and `getbuffer()`.

    Returns:
        Path to the saved file.
    """
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.name).name.replace("..", "_")
    path = upload_dir / safe_name
    with path.open("wb") as f:
        f.write(file.getbuffer())
    return path


def _select_export_seeds(k: int = 32) -> list[str]:
    """Seed selection policy: graph retriever → vector retriever → deterministic."""
    # We only have vector store here; graph index may be built below
    try:
        vs = create_vector_store(
            settings.database.qdrant_collection, enable_hybrid=True
        )
        vector_index = VectorStoreIndex.from_vector_store(vs)
    except Exception:  # pragma: no cover - best-effort
        vector_index = None
    pg_index = None
    return get_export_seed_ids(pg_index, vector_index, cap=int(k))


def ingest_files(files: Sequence[Any], enable_graphrag: bool = False) -> dict[str, Any]:
    """Ingest a collection of uploaded files and index them for retrieval.

    - Saves files locally.
    - Processes them via DocumentProcessor (Unstructured + LI pipeline).
    - Builds/updates Qdrant vector index for hybrid retrieval.
    - Logs analytics best-effort.

    Args:
        files: Sequence of uploaded file-like objects.
        enable_graphrag: Reserved flag for future GraphRAG actions.

    Returns:
        dict: {"count": int, "pg_index": PropertyGraphIndex | None}
    """
    proc = DocumentProcessor(settings=settings)
    count = 0
    t0 = perf_counter()

    # Collect LI documents for indexing into Qdrant after processing
    li_docs: list[LIDocument] = []

    for file in files:
        path = _save_uploaded_file(file)
        result = asyncio.run(proc.process_document_async(path))
        # Convert processed elements with non-empty text to LlamaIndex Documents
        for el in result.elements:
            if el.text and el.text.strip():
                li_docs.append(LIDocument(text=el.text, metadata=el.metadata))
        count += 1

    elapsed_ms = (perf_counter() - t0) * 1000.0

    # Build/update vector index in Qdrant for hybrid retrieval (best-effort)
    vs = None
    if li_docs:
        with contextlib.suppress(Exception):
            vs = create_vector_store(
                settings.database.qdrant_collection, enable_hybrid=True
            )
            storage = StorageContext.from_defaults(vector_store=vs)
            VectorStoreIndex.from_documents(li_docs, storage_context=storage)

    # Optional: GraphRAG build + export when requested (best-effort)
    pg_index = None
    if enable_graphrag and li_docs:
        with contextlib.suppress(Exception):
            pg_index = create_property_graph_index(
                li_docs,
                vector_store=vs,  # reuse store when available
            )
            out_dir = settings.data_dir / "graph"
            seeds = _select_export_seeds(k=32)
            export_graph_parquet(
                pg_index, out_dir / "graph.parquet", seed_ids=seeds, depth=1
            )
            export_graph_jsonl(
                pg_index, out_dir / "graph.jsonl", seed_ids=seeds, depth=1
            )
            with contextlib.suppress(Exception):
                log_jsonl(
                    {
                        "export_performed": True,
                        "export_type": "graph_both",
                        "seed_count": len(seeds),
                        "capped": len(seeds) >= 32,
                        "dest_relpath": str(out_dir.relative_to(settings.data_dir)),
                    }
                )

    # Best-effort analytics logging at adapter level
    if getattr(settings, "analytics_enabled", False):
        cfg = AnalyticsConfig(
            enabled=True,
            db_path=(
                settings.analytics_db_path
                or (settings.data_dir / "analytics" / "analytics.duckdb")
            ),
            retention_days=settings.analytics_retention_days,
        )
        am = AnalyticsManager.instance(cfg)
        with contextlib.suppress(Exception):
            am.log_embedding(
                model="unstructured+index", items=count, latency_ms=elapsed_ms
            )

    return {"count": count, "pg_index": pg_index}


__all__ = ["ingest_files"]
