#!/usr/bin/env python3
"""Ingest one local document and query it through DocMind's internal API."""

from __future__ import annotations

import argparse
from pathlib import Path

from llama_index.core import StorageContext, VectorStoreIndex

from src.agents.coordinator import MultiAgentCoordinator
from src.config import settings
from src.models.processing import IngestionConfig, IngestionInput
from src.processing.ingestion_api import generate_stable_id
from src.processing.ingestion_pipeline import ingest_documents_sync
from src.retrieval.router_factory import build_router_engine
from src.utils.storage import create_vector_store


def query_document(path: Path, query: str) -> str:
    """Ingest ``path`` and return the coordinator's answer to ``query``."""
    source = path.expanduser().resolve(strict=True)
    ingestion = ingest_documents_sync(
        IngestionConfig(cache_dir=settings.cache.dir / "ingestion"),
        [
            IngestionInput(
                document_id=generate_stable_id(source),
                source_path=source,
                metadata={"source": source.name},
            )
        ],
    )
    if not ingestion.nodes:
        raise RuntimeError("Ingestion produced no searchable nodes")

    vector_store = create_vector_store(
        settings.database.qdrant_collection,
        enable_hybrid=settings.retrieval.enable_server_hybrid,
    )
    router = None
    coordinator = None
    try:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        vector_index = VectorStoreIndex(
            ingestion.nodes,
            storage_context=storage_context,
            show_progress=False,
        )
        router = build_router_engine(vector_index, pg_index=None, settings=settings)
        coordinator = MultiAgentCoordinator()
        response = coordinator.process_query(
            query,
            settings_override={"router_engine": router},
            thread_id=f"document-{ingestion.manifest.corpus_hash[:12]}",
            user_id="local",
        )
        return response.content
    finally:
        if router is not None:
            router.close()
        vector_store.client.close()
        if coordinator is not None:
            coordinator.close()


def main() -> None:
    """Parse command-line arguments and print one grounded answer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path, help="Local document to ingest")
    parser.add_argument(
        "query",
        nargs="?",
        default="Summarize the key findings and action items.",
    )
    args = parser.parse_args()
    print(query_document(args.document, args.query))


if __name__ == "__main__":
    main()
