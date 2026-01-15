"""DocMind processing package.

Public entrypoints for the library-first ingestion pipeline and PDF page-image
rendering helpers.
"""

from .ingestion_api import (
    clear_ingestion_cache,
    collect_paths,
    generate_stable_id,
    load_documents,
    sanitize_document_metadata,
)
from .ingestion_pipeline import (
    build_ingestion_pipeline,
    ingest_documents,
    ingest_documents_sync,
    reindex_page_images_sync,
)
from .pdf_pages import pdf_pages_to_image_documents, save_pdf_page_images

__all__ = [
    "build_ingestion_pipeline",
    "clear_ingestion_cache",
    "collect_paths",
    "generate_stable_id",
    "ingest_documents",
    "ingest_documents_sync",
    "load_documents",
    "pdf_pages_to_image_documents",
    "reindex_page_images_sync",
    "sanitize_document_metadata",
    "save_pdf_page_images",
]
