"""DocMind AI Document Processing Package.

ADR-009 compliant document processing built on Unstructured + LlamaIndex
IngestionPipeline, with utilities for deterministic IDs and PDF page images.

Exports a compact, convenient surface:
- Core: DocumentProcessor, UnstructuredTransformation, ProcessingError
- Models: DocumentElement, ProcessingResult, ProcessingStrategy
- Embedding result models (for convenience): EmbeddingParameters/EmbeddingResult
- PDF helpers: pdf_pages_to_image_documents, save_pdf_page_images
- Utils: sha256_id, is_unstructured_like
"""

# Import document processor and factory functions
from src.models.embeddings import (
    EmbeddingParameters,
    EmbeddingResult,
)
from src.models.processing import (
    DocumentElement,
    ProcessingResult,
    ProcessingStrategy,
)
from src.processing.document_processor import (
    DocumentProcessor,
    ProcessingError,
    UnstructuredTransformation,
)
from src.processing.pdf_pages import (
    pdf_pages_to_image_documents,
    save_pdf_page_images,
)
from src.processing.utils import (
    is_unstructured_like,
    sha256_id,
)

__all__ = [
    "DocumentElement",
    "DocumentProcessor",
    "EmbeddingParameters",
    "EmbeddingResult",
    "ProcessingError",
    "ProcessingResult",
    "ProcessingStrategy",
    "UnstructuredTransformation",
    "is_unstructured_like",
    "pdf_pages_to_image_documents",
    "save_pdf_page_images",
    "sha256_id",
]
