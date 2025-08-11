"""Advanced document loading utilities for DocMind AI.

This module provides comprehensive document processing capabilities including:
- Multi-format document loading (PDF, DOCX, TXT, HTML, etc.)
- Multimodal content extraction (text + images)
- Video and audio processing with transcription
- Native FastEmbed multimodal embeddings
- High-resolution parsing with Unstructured library
- Semantic chunking with overlap preservation
- Base64 image encoding and metadata preservation

Supported formats:
- Documents: PDF, DOCX, PPTX, HTML, TXT, MD
- Images: PNG, JPEG, GIF, TIFF (embedded in documents)
- Media: MP4, AVI (video), MP3, WAV (audio)
- Structured: Tables, figures, captions

Key features:
- Unstructured library integration for high-fidelity parsing
- PyMuPDF for PDF image extraction
- Whisper integration for audio transcription
- LlamaParse fallback for complex documents
- GPU acceleration for transcription and embedding
- Streaming and batch processing support

Example:
    Basic document loading::

        from utils.document_loader import load_documents_unstructured

        # Load with multimodal support
        docs = load_documents_unstructured("document.pdf")

        # Check for extracted images
        for doc in docs:
            if doc.metadata.get('has_images'):
                print(f"Found images on page {doc.metadata['page_number']}")

        # Load with media transcription
        media_docs = load_documents_llama(files, parse_media=True)

Attributes:
    settings (AppSettings): Global application settings for document processing.
"""

import asyncio
import base64
import gc
import hashlib
import io
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import diskcache
import psutil
import torch
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import ImageDocument

try:
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
except ImportError:
    # FastEmbedEmbedding not available, will use fallback if needed
    FastEmbedEmbedding = None
try:
    from llama_parse import LlamaParse
except ImportError:
    # LlamaParse not available, will use fallback if needed
    LlamaParse = None
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from unstructured.partition.auto import partition
from whisper import load_model as whisper_load

from models.core import settings
from utils.exceptions import (
    DocumentLoadingError,
    handle_document_error,
    handle_embedding_error,
)
from utils.logging_utils import log_error_with_context, log_performance

# settings is now imported from models.core
from utils.model_manager import ModelManager  # noqa: E402
from utils.retry_utils import (  # noqa: E402
    async_with_timeout,
    document_retry,
    with_fallback,
)


# Performance and Caching Optimizations
@dataclass
class ProcessingMetrics:
    """Document processing performance metrics."""

    operation: str
    file_path: str
    file_size_mb: float
    duration_seconds: float
    document_count: int
    table_count: int
    image_count: int
    success: bool
    error_message: str | None = None


class DocumentCache:
    """Hash-based caching for processed documents using diskcache."""

    def __init__(self, cache_dir: str = "./cache/documents"):
        """Initialize document cache with specified directory."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(self.cache_dir))

    def get_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash for file content."""
        with open(file_path, "rb") as f:
            content = f.read()
        return hashlib.sha256(content).hexdigest()

    def is_cached(self, file_path: str) -> bool:
        """Check if document is cached."""
        file_hash = self.get_file_hash(file_path)
        return file_hash in self.cache

    def get_cached(self, file_path: str) -> list[Document] | None:
        """Get cached document."""
        if not self.is_cached(file_path):
            return None
        file_hash = self.get_file_hash(file_path)
        return self.cache.get(file_hash)

    def cache_document(self, file_path: str, documents: list[Document]) -> None:
        """Cache processed document with 1 hour expiry."""
        file_hash = self.get_file_hash(file_path)
        self.cache.set(file_hash, documents, expire=3600)


class MemoryMonitor:
    """Memory usage monitoring and optimization."""

    @staticmethod
    def get_memory_usage() -> dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }

    @contextmanager
    def memory_managed_processing(self, operation: str):
        """Context manager for memory-efficient processing."""
        initial_memory = self.get_memory_usage()
        logger.info(f"Starting {operation} - Memory: {initial_memory['rss_mb']:.1f}MB")

        try:
            yield
        finally:
            # Force garbage collection
            collected = gc.collect()
            final_memory = self.get_memory_usage()

            memory_delta = final_memory["rss_mb"] - initial_memory["rss_mb"]
            logger.info(
                f"Completed {operation} - Memory: {final_memory['rss_mb']:.1f}MB "
                f"(Δ{memory_delta:+.1f}MB), GC collected: {collected} objects"
            )


# NOTE: PerformanceMonitor class consolidated to utils.monitoring.py
# ProcessingMetrics kept for backward compatibility with existing code


# Global optimization instances
_document_cache = DocumentCache()
_memory_monitor = MemoryMonitor()

# Use unified performance monitoring
from .monitoring import get_performance_monitor

_performance_monitor = get_performance_monitor()


def cached_document_processing(processor_func):
    """Decorator for caching document processing results."""

    @wraps(processor_func)
    def wrapper(file_path: str, *args, **kwargs):
        # Check cache first
        cached_docs = _document_cache.get_cached(file_path)
        if cached_docs:
            logger.info(f"Using cached document for {Path(file_path).name}")
            return cached_docs

        # Process and cache
        documents = processor_func(file_path, *args, **kwargs)
        _document_cache.cache_document(file_path, documents)
        return documents

    return wrapper


def memory_efficient_processing(func):
    """Decorator for memory-efficient processing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        with _memory_monitor.memory_managed_processing(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def monitor_performance(operation: str):
    """Decorator for monitoring function performance using unified monitoring system."""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(file_path: str, *args, **kwargs):
            # Use unified async monitoring for async functions
            async with _performance_monitor.measure_document_processing(
                operation, file_path
            ) as metric:
                result = await func(file_path, *args, **kwargs)

                # Update metric with results
                if isinstance(result, list):
                    metric.document_count = len(result)
                    # Count tables and images from metadata
                    metric.table_count = sum(
                        1
                        for doc in result
                        if doc.metadata.get("element_type") == "Table"
                    )
                    metric.image_count = sum(
                        1 for doc in result if doc.metadata.get("has_images", False)
                    )

                return result

        @wraps(func)
        def sync_wrapper(file_path: str, *args, **kwargs):
            # For sync functions, we'll use a simple approach with manual timing
            # since the unified system is async-focused
            import time

            start_time = time.perf_counter()

            try:
                result = func(file_path, *args, **kwargs)

                # Manual logging for sync compatibility
                duration = time.perf_counter() - start_time
                file_size_mb = Path(file_path).stat().st_size / 1024 / 1024

                # Count results if it's a document list
                doc_count = len(result) if isinstance(result, list) else 0
                table_count = (
                    sum(
                        1
                        for doc in result
                        if isinstance(result, list)
                        and doc.metadata.get("element_type") == "Table"
                    )
                    if isinstance(result, list)
                    else 0
                )

                logger.info(
                    f"✅ {operation}: {Path(file_path).name} - "
                    f"{duration:.2f}s, {file_size_mb:.1f}MB, {doc_count} docs, {table_count} tables"
                )

                return result
            except Exception as e:
                duration = time.perf_counter() - start_time
                logger.error(
                    f"❌ {operation}: {Path(file_path).name} failed in {duration:.2f}s - {e}"
                )
                raise

        # Return async wrapper if function is async, sync wrapper otherwise
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


@with_fallback(lambda pdf_path: [])
def extract_images_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """Extract images from PDF files using PyMuPDF with structured error handling.

    Extracts all embedded images from a PDF document and converts them to
    base64-encoded PNG format for storage and processing. Handles various
    image formats and color spaces with automatic conversion. Includes
    comprehensive error handling and performance monitoring.

    Args:
        pdf_path: Absolute or relative path to the PDF file to process.
            Must be a valid PDF file accessible for reading.

    Returns:
        List of dictionaries containing extracted image data:
        - 'image_data' (str): Base64-encoded PNG image data
        - 'page_number' (int): Source page number (1-indexed)
        - 'image_index' (int): Index of image within the page (0-indexed)
        - 'format' (str): Output format (always 'PNG')
        - 'size' (tuple): Image dimensions as (width, height)

    Raises:
        DocumentLoadingError: If PDF processing fails critically.
        FileNotFoundError: If the PDF file does not exist.

    Note:
        Only processes images in GRAY or RGB color spaces to avoid
        corruption. CMYK and other complex color spaces are skipped.
        Images are converted to PNG format for consistent handling.
        Falls back to empty list on errors.

    Example:
        >>> images = extract_images_from_pdf("document.pdf")
        >>> for img in images:
        ...     print(f"Page {img['page_number']}: {img['size']} pixels")
        ...     # Use img['image_data'] for base64 image content
    """
    start_time = time.perf_counter()
    images = []

    try:
        from pathlib import Path

        import fitz  # PyMuPDF

        # Validate file exists and is readable
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise DocumentLoadingError(
                f"PDF file not found: {pdf_path}",
                context={"pdf_path": str(pdf_path)},
                operation="pdf_file_validation",
            )

        # Use proper resource management for fitz document
        with fitz.open(str(pdf_path)) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = None
                    try:
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("ppm")

                            # Use context manager for BytesIO and PIL Image
                            with (
                                io.BytesIO(img_data) as img_stream,
                                Image.open(img_stream) as img_pil,
                            ):
                                # Convert to base64 for storage
                                with io.BytesIO() as buffer:
                                    img_pil.save(buffer, format="PNG")
                                    img_base64 = base64.b64encode(
                                        buffer.getvalue()
                                    ).decode()

                                images.append(
                                    {
                                        "image_data": img_base64,
                                        "page_number": page_num + 1,
                                        "image_index": img_index,
                                        "format": "PNG",
                                        "size": img_pil.size,
                                    }
                                )
                    finally:
                        # Ensure pixmap is properly cleaned up
                        if pix is not None:
                            pix = None

        duration = time.perf_counter() - start_time
        log_performance(
            "pdf_image_extraction",
            duration,
            pdf_path=str(pdf_path),
            image_count=len(images),
        )

        logger.success(
            f"Extracted {len(images)} images from PDF",
            extra={
                "pdf_path": str(pdf_path),
                "image_count": len(images),
                "duration_ms": round(duration * 1000, 2),
            },
        )

    except Exception as e:
        log_error_with_context(
            e,
            "pdf_image_extraction",
            context={
                "pdf_path": str(pdf_path),
                "attempted_images": len(images),
            },
        )
        # Re-raise to trigger fallback decorator
        raise DocumentLoadingError(
            f"PDF image extraction failed for {pdf_path}",
            context={"pdf_path": str(pdf_path)},
            original_error=e,
            operation="pdf_image_extraction",
        ) from e

    return images


@with_fallback(
    lambda text, images=None: {
        "text_embedding": None,
        "image_embeddings": [],
        "combined_embedding": None,
        "provider_used": "failed_fallback",
    }
)
@memory_efficient_processing
def create_native_multimodal_embeddings(
    text: str,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create multimodal embeddings using FastEmbed with structured error handling.

    Generates embeddings for both text and image content using FastEmbed's
    LateInteractionMultimodalEmbedding model. Provides unified embedding
    space for cross-modal similarity search and retrieval with comprehensive
    error handling and performance monitoring.

    Features:
    - FastEmbed native multimodal processing
    - Automatic provider detection (CUDA/CPU)
    - Memory-efficient batch processing
    - Temporary file management for images
    - Graceful fallbacks for missing components
    - Structured error handling and retry logic

    Args:
        text: Text content to embed. Can be any length but will be truncated
            to model's maximum sequence length.
        images: Optional list of image dictionaries containing 'image_data'
            key with base64-encoded image content. Defaults to None.

    Returns:
        Dictionary containing embedding results:
        - 'text_embedding' (list[float] | None): Text embedding vector
        - 'image_embeddings' (list[dict]): List of image embedding objects with
          'embedding' and 'metadata' keys
        - 'combined_embedding' (list[float] | None): Combined embedding vector
        - 'provider_used' (str): Embedding provider that was used

    Raises:
        EmbeddingError: If embedding creation fails critically.

    Note:
        Images are temporarily saved to disk for FastEmbed processing and
        automatically cleaned up after embedding computation. Falls back
        to text-only embeddings if multimodal models are unavailable.

    Example:
        >>> text = "This is a sample document."
        >>> images = [{'image_data': 'base64_encoded_image'}]
        >>> embeddings = create_native_multimodal_embeddings(text, images)
        >>> if embeddings['provider_used'] == 'fastembed_native_multimodal':
        ...     print("Successfully created multimodal embeddings")
        >>> text_emb = embeddings['text_embedding']
        >>> img_embs = embeddings['image_embeddings']
    """
    start_time = time.perf_counter()

    embeddings = {
        "text_embedding": None,
        "image_embeddings": [],
        "combined_embedding": None,
        "provider_used": None,
    }

    logger.info(
        "Creating multimodal embeddings",
        extra={
            "text_length": len(text),
            "image_count": len(images) if images else 0,
            "cuda_available": torch.cuda.is_available(),
        },
    )

    try:
        # Use FastEmbed native LateInteractionMultimodalEmbedding
        try:
            # Use singleton pattern to prevent redundant model initializations
            model = ModelManager.get_multimodal_embedding_model()

            # Process text embedding with native FastEmbed
            text_embedding = list(model.embed_text([text]))[0]
            embeddings["text_embedding"] = text_embedding.flatten().tolist()

            # Process image embeddings with native FastEmbed
            if images:
                # Save base64 images to temporary files for FastEmbed
                image_paths = []
                for i, img_data in enumerate(images):
                    try:
                        img_bytes = base64.b64decode(img_data["image_data"])
                        temp_path = f"{tempfile.gettempdir()}/multimodal_img_{i}.png"
                        with open(temp_path, "wb") as f:
                            f.write(img_bytes)
                        image_paths.append(temp_path)
                    except Exception as e:
                        logging.warning("Failed to save image %s: %s", i, e)

                if image_paths:
                    try:
                        # Native FastEmbed image embeddings
                        image_embeddings = list(model.embed_image(image_paths))

                        for i, img_emb in enumerate(image_embeddings):
                            embeddings["image_embeddings"].append(
                                {
                                    "embedding": img_emb.flatten().tolist(),
                                    "metadata": images[i] if i < len(images) else {},
                                }
                            )
                    finally:
                        # Always clean up temporary files
                        import contextlib

                        for path in image_paths:
                            with contextlib.suppress(Exception):
                                os.unlink(path)

            # FastEmbed handles combined embeddings natively
            embeddings["combined_embedding"] = embeddings["text_embedding"]
            embeddings["provider_used"] = "fastembed_native_multimodal"

            logging.info("Using FastEmbed native LateInteractionMultimodalEmbedding")

        except ImportError:
            logging.warning(
                "FastEmbed LateInteractionMultimodalEmbedding not available, "
                "using text-only"
            )
            # Fallback to FastEmbed text-only
            if FastEmbedEmbedding is not None:
                fastembed_model = FastEmbedEmbedding(
                    model_name=settings.dense_embedding_model,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    cache_dir="./embeddings_cache",
                )
                embeddings["text_embedding"] = fastembed_model.get_text_embedding(text)
                embeddings["combined_embedding"] = embeddings["text_embedding"]
                embeddings["provider_used"] = "fastembed_text_only"
            else:
                # FastEmbed not available, use basic text embedding
                embeddings["text_embedding"] = [0.1] * 384  # Mock embedding
                embeddings["combined_embedding"] = embeddings["text_embedding"]
                embeddings["provider_used"] = "mock_fallback"

    except Exception as e:
        log_error_with_context(
            e,
            "multimodal_embedding_creation",
            context={
                "text_length": len(text),
                "image_count": len(images) if images else 0,
                "cuda_available": torch.cuda.is_available(),
            },
        )

        # Ultimate fallback to FastEmbed text-only
        try:
            logger.warning(
                "Multimodal embedding failed, falling back to text-only",
                extra={"original_error": str(e)},
            )

            if FastEmbedEmbedding is not None:
                fastembed_model = FastEmbedEmbedding(
                    model_name=settings.dense_embedding_model,
                    cache_dir="./embeddings_cache",
                )
                embeddings["text_embedding"] = fastembed_model.get_text_embedding(text)
                embeddings["combined_embedding"] = embeddings["text_embedding"]
                embeddings["provider_used"] = "fastembed_fallback"
            else:
                # FastEmbed not available, use mock embedding
                embeddings["text_embedding"] = [0.1] * 384  # Mock embedding
                embeddings["combined_embedding"] = embeddings["text_embedding"]
                embeddings["provider_used"] = "mock_ultimate_fallback"

        except Exception as fallback_e:
            log_error_with_context(
                fallback_e,
                "multimodal_embedding_fallback",
                context={
                    "text_length": len(text),
                    "original_error": str(e),
                },
            )

            # Re-raise to trigger decorator fallback
            raise handle_embedding_error(
                fallback_e,
                operation="multimodal_embedding_creation",
                text_length=len(text),
                image_count=len(images) if images else 0,
                original_error=e,
            ) from e

    # Log performance and results
    duration = time.perf_counter() - start_time
    log_performance(
        "multimodal_embedding_creation",
        duration,
        text_length=len(text),
        image_count=len(images) if images else 0,
        provider_used=embeddings["provider_used"],
        success=embeddings["provider_used"] != "failed",
    )

    logger.success(
        f"Multimodal embeddings created using {embeddings['provider_used']}",
        extra={
            "text_length": len(text),
            "image_count": len(images) if images else 0,
            "provider": embeddings["provider_used"],
            "duration_ms": round(duration * 1000, 2),
        },
    )

    return embeddings


@document_retry
@with_fallback(
    lambda file_path: [
        Document(
            text=f"Failed to load document: {file_path}",
            metadata={
                "source": file_path,
                "type": "error_fallback",
                "has_images": False,
            },
        )
    ]
)
@cached_document_processing
@monitor_performance("unstructured_processing")
@memory_efficient_processing
def load_documents_unstructured(file_path: str) -> list[Document]:
    """Load documents using Unstructured with structured error handling.

    Uses Unstructured library to extract text, images, tables, and other elements
    from documents with high-res strategy for best quality. Supports embedded
    image extraction and semantic chunking while preserving document structure.
    Includes comprehensive error handling, performance monitoring, and retries.

    Args:
        file_path: Path to the document file to process.

    Returns:
        List of Document objects with multimodal elements and metadata.

    Raises:
        DocumentLoadingError: If document parsing fails critically after retries.

    Note:
        Falls back gracefully to existing loader on Unstructured failures.
        Includes automatic retry on transient failures and performance logging.
    """
    start_time = time.perf_counter()

    logger.info(
        "Loading document with Unstructured",
        extra={
            "file_path": file_path,
            "strategy": settings.parse_strategy,
            "chunk_size": settings.chunk_size,
        },
    )

    try:
        # Validate file exists before processing
        if not os.path.exists(file_path):
            raise DocumentLoadingError(
                f"Document file not found: {file_path}",
                context={"file_path": file_path},
                operation="file_validation",
            )

        # Partition document with high-res strategy for multimodal extraction
        elements = partition(
            filename=file_path,
            strategy=settings.parse_strategy,  # "hi_res" for best quality
            include_page_breaks=True,
            extract_images_in_pdf=True,  # Extract embedded images
            extract_image_block_types=["Image", "FigureCaption"],
            extract_image_block_output_dir=None,  # Keep images in base64
            infer_table_structure=True,  # Better table handling
            chunking_strategy="by_title",  # Semantic chunking
            max_characters=settings.chunk_size,
            combine_text_under_n_chars=100,  # Combine small text blocks
            new_after_n_chars=settings.chunk_size - settings.chunk_overlap,
        )

        # Convert elements to LlamaIndex documents
        documents = []
        current_page = None
        page_images = []

        try:
            for element in elements:
                metadata = {
                    "element_type": element.category,
                    "page_number": (
                        element.metadata.page_number
                        if hasattr(element.metadata, "page_number")
                        else None
                    ),
                    "filename": (
                        element.metadata.filename
                        if hasattr(element.metadata, "filename")
                        else os.path.basename(file_path)
                    ),
                    "coordinates": (
                        element.metadata.coordinates
                        if hasattr(element.metadata, "coordinates")
                        else None
                    ),
                }

            # Handle different element types for multimodal processing
            if element.category == "Image":
                # Extract image data for multimodal embedding
                image_data = None
                if (
                    hasattr(element.metadata, "image_base64")
                    and element.metadata.image_base64
                ):
                    image_data = element.metadata.image_base64
                elif hasattr(element, "text") and element.text:
                    # Sometimes image data is in text field
                    try:
                        # Check if text is base64 encoded image
                        base64.b64decode(element.text)
                        image_data = element.text
                    except (ValueError, TypeError):
                        # Not base64 data, skip this element
                        image_data = None

                if image_data:
                    # Create ImageDocument for multimodal index
                    doc = ImageDocument(
                        image=image_data,
                        metadata={**metadata, "image_base64": image_data},
                    )
                    documents.append(doc)
                    page_images.append(
                        {
                            "image_data": image_data,
                            "page_number": metadata.get("page_number", 1),
                            "element_type": "Image",
                        }
                    )
                    page_num = metadata.get("page_number", "unknown")
                    logger.debug(f"Extracted image from page {page_num}")

            elif element.category in [
                "Table",
                "FigureCaption",
                "Title",
                "NarrativeText",
                "Text",
            ]:
                # Create TextNode for text elements
                text_content = str(element).strip()
                if text_content:  # Only create document if there's actual content
                    # Add page images to text metadata for multimodal context
                    enhanced_metadata = {**metadata}
                    if page_images and metadata.get("page_number") == current_page:
                        enhanced_metadata["page_images"] = page_images
                        enhanced_metadata["has_images"] = True
                    else:
                        enhanced_metadata["has_images"] = False

                    doc = Document(
                        text=text_content,
                        metadata=enhanced_metadata,
                    )
                    documents.append(doc)

                # Track current page for image association
                if metadata.get("page_number") != current_page:
                    current_page = metadata.get("page_number")
                    page_images = []  # Reset for new page

        finally:
            # Clean up large base64 data from elements to prevent memory leaks
            if elements:
                for elem in elements:
                    if hasattr(elem, "metadata") and hasattr(
                        elem.metadata, "image_base64"
                    ):
                        # Clear large base64 data after processing
                        elem.metadata.image_base64 = None
                # Explicit cleanup
                del elements

        # Apply additional chunking if documents are too large
        if settings.chunk_size > 0:
            documents = chunk_documents_structured(documents)

        # Log performance and results
        duration = time.perf_counter() - start_time
        log_performance(
            "unstructured_document_loading",
            duration,
            file_path=file_path,
            document_count=len(documents),
            strategy=settings.parse_strategy,
        )

        logger.success(
            f"Loaded {len(documents)} multimodal elements from {file_path}",
            extra={
                "file_path": file_path,
                "document_count": len(documents),
                "strategy": settings.parse_strategy,
                "duration_ms": round(duration * 1000, 2),
            },
        )
        return documents

    except Exception as e:
        log_error_with_context(
            e,
            "unstructured_document_loading",
            context={
                "file_path": file_path,
                "strategy": settings.parse_strategy,
                "chunk_size": settings.chunk_size,
            },
        )

        logger.warning(
            f"Unstructured loading failed for {file_path}, falling back to LlamaParse",
            extra={"original_error": str(e)},
        )

        # Fall back to existing loader - create temporary file list for compatibility
        try:

            class TempFile:
                def __init__(self, path: str):
                    self.name = os.path.basename(path)
                    with open(path, "rb") as f:
                        self._content = f.read()

                def getvalue(self):
                    return self._content

            temp_files = [TempFile(file_path)]
            fallback_docs = load_documents_llama(
                uploaded_files=temp_files, parse_media=False, enable_multimodal=True
            )

            logger.info(
                f"Fallback successful: loaded {len(fallback_docs)} "
                "documents via LlamaParse"
            )
            return fallback_docs

        except Exception as fallback_error:
            log_error_with_context(
                fallback_error,
                "document_loading_fallback",
                context={
                    "file_path": file_path,
                    "original_error": str(e),
                },
            )

            # Re-raise to trigger decorator fallback
            raise handle_document_error(
                fallback_error,
                operation="document_loading_with_fallback",
                file_path=file_path,
                original_error=e,
            ) from e


def chunk_documents_structured(documents: list[Document]) -> list[Document]:
    """Apply semantic chunking to documents while preserving structure.

    Uses LlamaIndex SentenceSplitter for semantic chunking that respects
    sentence boundaries and maintains context through overlapping chunks.

    Args:
        documents: List of Document objects to chunk.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    from llama_index.core.node_parser import SentenceSplitter

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        paragraph_separator="\n\n",  # Better paragraph detection
        secondary_chunking_regex="[^,.;。]+[,.;。]?",  # Sentence-aware chunking
        tokenizer=None,  # Use default tokenizer
    )

    chunked_docs = []
    for doc in documents:
        # Only chunk text documents, keep images as-is
        if isinstance(doc, ImageDocument):
            chunked_docs.append(doc)
        else:
            # Apply chunking while preserving metadata
            chunks = splitter.get_nodes_from_documents([doc])
            chunked_docs.extend(chunks)

    return chunked_docs


@document_retry
@memory_efficient_processing
def load_documents_llama(
    uploaded_files: list[Any], parse_media: bool = False, enable_multimodal: bool = True
) -> list[Document]:
    """Load documents using LlamaParse with structured error handling.

    Supports standard document formats plus video/audio ingestion and PDF image
    extraction for multimodal processing with comprehensive error handling,
    performance monitoring, and retry logic.

    Args:
        uploaded_files: List of uploaded file objects.
        parse_media: Whether to parse video/audio files. Defaults to False.
        enable_multimodal: Whether to enable multimodal processing for PDFs.
            Defaults to True.

    Returns:
        List of loaded Document objects with multimodal embeddings where applicable.

    Raises:
        DocumentLoadingError: If critical document loading
        operations fail after retries.

    Note:
        Includes automatic retries for transient failures and comprehensive
        error context preservation for debugging.
    """
    start_time = time.perf_counter()

    logger.info(
        "Loading documents with LlamaParse",
        extra={
            "file_count": len(uploaded_files),
            "parse_media": parse_media,
            "enable_multimodal": enable_multimodal,
        },
    )

    if LlamaParse is None:
        raise ImportError(
            "LlamaParse is not available. Please install llama-parse package."
        )
    parser = LlamaParse(result_type="markdown")  # Latest for tables/images
    docs: list[Document] = []

    for file in uploaded_files:
        file_path = None
        try:
            # Create temporary file with proper cleanup
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(file.name)[1]
            ) as tmp_file:
                tmp_file.write(file.getvalue())
                file_path = tmp_file.name

            if parse_media and (
                file_path.endswith((".mp4", ".avi"))
                or file_path.endswith((".mp3", ".wav"))
            ):
                if "video" in file.type:
                    clip = None
                    audio_path = None
                    try:
                        clip = VideoFileClip(file_path)

                        # Use context manager for temporary audio file
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ) as audio_tmp:
                            audio_path = audio_tmp.name

                        clip.audio.write_audiofile(audio_path)
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        model = whisper_load(
                            "base", device=device
                        )  # GPU offload if available
                        result = model.transcribe(audio_path)
                        text: str = result["text"]

                        # Extract frames at intervals (e.g., every 5s) per practices
                        frames: list[Any] = []
                        for t in range(0, int(clip.duration), 5):
                            frame = clip.get_frame(t)
                            img = Image.fromarray(frame)
                            frames.append(img)

                        docs.append(
                            Document(
                                text=text,
                                metadata={
                                    "images": frames,
                                    "source": file.name,
                                    "type": "video",
                                },
                            )
                        )
                    finally:
                        # Clean up video resources
                        if clip is not None:
                            clip.close()
                        # Clean up temporary audio file
                        if audio_path and os.path.exists(audio_path):
                            os.remove(audio_path)

                elif "audio" in file.type:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = whisper_load("base", device=device)
                    result = model.transcribe(file_path)
                    text = result["text"]
                    docs.append(
                        Document(
                            text=text, metadata={"source": file.name, "type": "audio"}
                        )
                    )
            else:
                # Standard document processing with multimodal support
                reader = SimpleDirectoryReader(
                    input_files=[file_path], file_extractor={".*": parser}
                )
                loaded_docs = reader.load_data()

                # Enhanced multimodal processing for PDFs
                if enable_multimodal and file_path.lower().endswith(".pdf"):
                    try:
                        # Extract images from PDF
                        pdf_images = extract_images_from_pdf(file_path)

                        # Process each document with multimodal embeddings
                        for doc in loaded_docs:
                            if pdf_images:
                                # Create local multimodal embeddings (offline)
                                multimodal_embeddings = (
                                    create_native_multimodal_embeddings(
                                        text=doc.text,
                                        images=pdf_images,
                                    )
                                )

                                # Update document with multimodal data
                                doc.metadata.update(
                                    {
                                        "source": file.name,
                                        "type": "pdf_multimodal",
                                        "image_count": len(pdf_images),
                                        "has_images": True,
                                        "multimodal_embeddings": multimodal_embeddings,
                                    }
                                )

                                logging.info(
                                    "Created multimodal embeddings for %s (%s images)",
                                    file.name,
                                    len(pdf_images),
                                )
                            else:
                                # PDF without images - standard text processing
                                doc.metadata.update(
                                    {
                                        "source": file.name,
                                        "type": "pdf_text_only",
                                        "has_images": False,
                                    }
                                )

                    except Exception as e:
                        logging.warning(
                            "Multimodal processing failed for %s: %s", file.name, e
                        )
                        # Fallback to standard processing
                        for doc in loaded_docs:
                            doc.metadata.update(
                                {
                                    "source": file.name,
                                    "type": "pdf_fallback",
                                    "has_images": False,
                                }
                            )
                else:
                    # Standard processing for non-PDF files
                    for doc in loaded_docs:
                        doc.metadata.update(
                            {
                                "source": file.name,
                                "type": "standard_document",
                                "has_images": False,
                            }
                        )

                docs.extend(loaded_docs)

        except FileNotFoundError as e:
            log_error_with_context(
                e,
                "file_not_found",
                context={"file_name": file.name},
            )
            logger.error(f"File not found: {file.name}")
        except ValueError as e:
            log_error_with_context(
                e,
                "invalid_file_format",
                context={"file_name": file.name},
            )
            logger.error(f"Invalid file format: {file.name}")
        except Exception as e:
            log_error_with_context(
                e,
                "document_loading_error",
                context={
                    "file_name": file.name,
                    "parse_media": parse_media,
                    "enable_multimodal": enable_multimodal,
                },
            )
            logger.error(f"Unexpected error loading {file.name}: {e}")
        finally:
            # Ensure temporary file is always cleaned up
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    logging.warning(
                        "Failed to cleanup temp file %s: %s", file_path, cleanup_error
                    )

    # Log performance and results
    duration = time.perf_counter() - start_time
    log_performance(
        "llamaparse_document_loading",
        duration,
        file_count=len(uploaded_files),
        document_count=len(docs),
        parse_media=parse_media,
        enable_multimodal=enable_multimodal,
    )

    logger.success(
        f"Loaded {len(docs)} documents via LlamaParse",
        extra={
            "file_count": len(uploaded_files),
            "document_count": len(docs),
            "parse_media": parse_media,
            "enable_multimodal": enable_multimodal,
            "duration_ms": round(duration * 1000, 2),
        },
    )
    return docs


# Streaming Document Processing for Async Performance


@async_with_timeout(timeout_seconds=300)
async def stream_document_processing(file_paths: list[str]) -> AsyncIterator[Document]:
    """Stream document processing for large document sets with error handling.

    Processes documents asynchronously and yields results as they
    become available, reducing memory usage and improving responsiveness.
    Provides significant performance improvements for large document collections
    with comprehensive error handling and timeout protection.

    Args:
        file_paths: List of file paths to process.

    Yields:
        Processed Document objects as they become available.

    Raises:
        DocumentLoadingError: If critical streaming operations fail.
        asyncio.TimeoutError: If processing exceeds timeout limit.

    Note:
        Uses semaphore to limit concurrent processing and prevent resource exhaustion.
        Falls back gracefully on individual document processing failures.
        Includes timeout protection and performance monitoring.
    """
    start_time = time.perf_counter()
    processed_count = 0
    error_count = 0

    logger.info(
        "Starting stream document processing",
        extra={"file_count": len(file_paths), "semaphore_limit": 5},
    )

    semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

    async def process_single_file(file_path: str) -> list[Document]:
        """Process a single file with semaphore limiting and error handling."""
        async with semaphore:
            try:
                return await asyncio.to_thread(load_documents_unstructured, file_path)
            except Exception as e:
                nonlocal error_count
                error_count += 1

                log_error_with_context(
                    e,
                    "stream_document_processing_single_file",
                    context={"file_path": file_path},
                )

                logger.warning(
                    (
                        f"Document processing failed for {file_path}, "
                        "continuing with others"
                    ),
                    extra={"error_count": error_count, "file_path": file_path},
                )
                return []  # Return empty list on failure

    # Create tasks for all files
    tasks = [
        asyncio.create_task(process_single_file(file_path)) for file_path in file_paths
    ]

    # Yield results as they complete
    for completed_task in asyncio.as_completed(tasks):
        try:
            documents = await completed_task
            for doc in documents:
                processed_count += 1
                yield doc
        except Exception as e:
            error_count += 1
            log_error_with_context(
                e,
                "stream_document_processing_task_completion",
                context={
                    "processed_count": processed_count,
                    "error_count": error_count,
                },
            )
            logger.error(
                f"Document processing task failed: {e}",
                extra={"processed_count": processed_count, "error_count": error_count},
            )
            # Continue with other documents
            continue

    # Log final performance metrics
    duration = time.perf_counter() - start_time
    log_performance(
        "stream_document_processing",
        duration,
        file_count=len(file_paths),
        processed_count=processed_count,
        error_count=error_count,
    )

    logger.success(
        "Stream document processing completed",
        extra={
            "file_count": len(file_paths),
            "processed_count": processed_count,
            "error_count": error_count,
            "success_rate": (processed_count / len(file_paths)) if file_paths else 0,
            "duration_seconds": round(duration, 2),
        },
    )


async def load_documents_parallel(
    file_paths: list[str], max_concurrent: int = 5
) -> list[Document]:
    """Load multiple documents in parallel with concurrency limit."""
    import asyncio
    from asyncio import Semaphore

    from loguru import logger

    semaphore = Semaphore(max_concurrent)

    async def load_with_limit(file_path: str) -> list[Document]:
        """Load single document with concurrency limit."""
        async with semaphore:
            try:
                if file_path.endswith(".pdf"):
                    return await load_documents_unstructured_async(file_path)
                else:
                    # Use asyncio.to_thread for sync loaders
                    return await asyncio.to_thread(load_documents_llama_sync, file_path)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                return []

    # Load all documents in parallel
    logger.info(f"Loading {len(file_paths)} documents in parallel")

    start_time = asyncio.get_event_loop().time()
    doc_lists = await asyncio.gather(
        *[load_with_limit(path) for path in file_paths], return_exceptions=False
    )

    # Flatten results
    all_docs = []
    for docs in doc_lists:
        if docs:
            all_docs.extend(docs)

    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(
        f"Loaded {len(all_docs)} documents from {len(file_paths)} files "
        f"in {elapsed:.2f}s"
    )

    return all_docs


async def load_documents_unstructured_async(file_path: str) -> list[Document]:
    """Async wrapper for Unstructured document loading."""
    return await asyncio.to_thread(load_documents_unstructured, file_path)


@monitor_performance("llamaparse_processing")
@memory_efficient_processing
def load_documents_llama_sync(file_path: str) -> list[Document]:
    """Synchronous wrapper for single file loading with LlamaParse."""

    # Create mock file object for existing load_documents_llama
    class MockFile:
        def __init__(self, path: str):
            self.name = os.path.basename(path)
            with open(path, "rb") as f:
                self._content = f.read()

        def getvalue(self):
            return self._content

    mock_file = MockFile(file_path)
    return load_documents_llama([mock_file], parse_media=False, enable_multimodal=True)


@async_with_timeout(timeout_seconds=600)  # 10 minute timeout for large batches
async def batch_embed_documents(
    documents: list[Document], batch_size: int = 32
) -> list[list[float]]:
    """Batch document embedding with structured error handling.

    Embeds documents in parallel batches for optimal performance.
    Uses asyncio.to_thread to prevent blocking the event loop during
    CPU-intensive embedding operations. Includes comprehensive error
    handling, timeout protection, and performance monitoring.

    Args:
        documents: Documents to embed.
        batch_size: Size of processing batches for optimal memory usage.

    Returns:
        List of embedding vectors corresponding to input documents.

    Raises:
        EmbeddingError: If embedding operations fail critically.
        asyncio.TimeoutError: If embedding exceeds timeout limit.

    Note:
        Falls back gracefully on batch failures by providing zero embeddings.
        Processes batches in parallel for maximum throughput.
        Includes timeout protection for large document collections.
    """
    start_time = time.perf_counter()

    logger.info(
        "Starting batch document embedding",
        extra={
            "document_count": len(documents),
            "batch_size": batch_size,
            "batch_count": (len(documents) + batch_size - 1) // batch_size,
        },
    )

    from utils.utils import get_embed_model

    try:
        embed_model = get_embed_model()
    except Exception as e:
        raise handle_embedding_error(
            e,
            operation="embedding_model_initialization",
            document_count=len(documents),
        ) from e

    # Split into batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def embed_batch(batch: list[Document]) -> list[list[float]]:
        """Embed a single batch of documents with error handling."""
        try:
            texts = [doc.text for doc in batch]
            return await asyncio.to_thread(embed_model.embed, texts)
        except Exception as e:
            log_error_with_context(
                e,
                "batch_embedding",
                context={
                    "batch_size": len(batch),
                    "text_lengths": [
                        len(doc.text) for doc in batch[:3]
                    ],  # Sample sizes
                },
            )

            logger.warning(
                f"Batch embedding failed, using placeholder embeddings: {e}",
                extra={"batch_size": len(batch)},
            )

            # Add placeholder embeddings for failed batch
            settings_instance = settings
            return [[0.0] * settings_instance.dense_embedding_dimension] * len(batch)

    # Process batches in parallel
    batch_results = await asyncio.gather(
        *[embed_batch(batch) for batch in batches], return_exceptions=True
    )

    # Combine results with detailed error tracking
    all_embeddings = []
    failed_batches = 0

    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            failed_batches += 1
            log_error_with_context(
                result,
                "batch_embedding_result",
                context={"batch_index": i, "batch_size": len(batches[i])},
            )

            logger.error(
                f"Batch {i} embedding failed: {result}",
                extra={"batch_index": i, "failed_batches": failed_batches},
            )

            # Add placeholder embeddings for failed batch
            failed_batch_size = len(batches[i])
            settings_instance = settings
            all_embeddings.extend(
                [[0.0] * settings_instance.dense_embedding_dimension]
                * failed_batch_size
            )
        else:
            all_embeddings.extend(result)

    # Log performance and results
    duration = time.perf_counter() - start_time
    log_performance(
        "batch_document_embedding",
        duration,
        document_count=len(documents),
        batch_count=len(batches),
        failed_batches=failed_batches,
        embedding_count=len(all_embeddings),
    )

    logger.success(
        f"Embedded {len(all_embeddings)} documents in {len(batches)} parallel batches",
        extra={
            "document_count": len(documents),
            "batch_count": len(batches),
            "failed_batches": failed_batches,
            "success_rate": (len(batches) - failed_batches) / len(batches)
            if batches
            else 0,
            "duration_seconds": round(duration, 2),
        },
    )
    return all_embeddings


@async_with_timeout(timeout_seconds=900)  # 15 minute timeout for large collections
async def process_documents_streaming(
    file_paths: list[str], chunk_size: int = 1024, chunk_overlap: int = 200
) -> AsyncIterator[Document]:
    """Process documents with streaming and chunking with error handling.

    Combines streaming document processing with intelligent chunking to
    handle large document collections efficiently. Provides real-time
    progress updates and memory-conscious processing with comprehensive
    error handling and performance monitoring.

    Args:
        file_paths: List of file paths to process.
        chunk_size: Maximum chunk size for document splitting.
        chunk_overlap: Overlap between chunks for context preservation.

    Yields:
        Processed and chunked Document objects as they become available.

    Raises:
        DocumentLoadingError: If critical streaming operations fail.
        asyncio.TimeoutError: If processing exceeds timeout limit.

    Note:
        Automatically applies semantic chunking to large documents while
        preserving multimodal content. Streams results for immediate processing.
        Includes timeout protection and detailed error tracking.
    """
    start_time = time.perf_counter()
    processed_count = 0
    chunked_count = 0
    error_count = 0
    total_files = len(file_paths)

    logger.info(
        "Starting streaming document processing with chunking",
        extra={
            "file_count": total_files,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    try:
        async for document in stream_document_processing(file_paths):
            try:
                # Apply chunking if document is too large
                if len(document.text) > chunk_size:
                    # Use chunking for large documents
                    from llama_index.core.node_parser import SentenceSplitter

                    splitter = SentenceSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        paragraph_separator="\n\n",
                        secondary_chunking_regex="[^,.;。]+[,.;。]?",
                    )

                    chunks = await asyncio.to_thread(
                        splitter.get_nodes_from_documents, [document]
                    )

                    for chunk in chunks:
                        chunked_count += 1
                        yield chunk

                    logger.debug(
                        f"Chunked document into {len(chunks)} pieces",
                        extra={
                            "original_length": len(document.text),
                            "chunks": len(chunks),
                        },
                    )
                else:
                    # Document is small enough, yield as-is
                    yield document

                processed_count += 1
                if processed_count % 10 == 0:  # Log progress every 10 documents
                    logger.info(
                        f"Processed {processed_count}/{total_files} documents",
                        extra={
                            "processed": processed_count,
                            "total": total_files,
                            "chunked_count": chunked_count,
                            "error_count": error_count,
                        },
                    )

            except Exception as e:
                error_count += 1
                log_error_with_context(
                    e,
                    "streaming_document_chunking",
                    context={
                        "processed_count": processed_count,
                        "document_length": len(document.text)
                        if hasattr(document, "text")
                        else 0,
                        "chunk_size": chunk_size,
                    },
                )

                logger.warning(
                    f"Document chunking failed, skipping document: {e}",
                    extra={
                        "error_count": error_count,
                        "processed_count": processed_count,
                    },
                )
                continue

    except Exception as e:
        log_error_with_context(
            e,
            "streaming_document_processing_main",
            context={
                "processed_count": processed_count,
                "total_files": total_files,
                "chunk_size": chunk_size,
            },
        )

        # Re-raise critical streaming errors
        raise handle_document_error(
            e,
            operation="streaming_document_processing",
            processed_count=processed_count,
            total_files=total_files,
        ) from e

    finally:
        # Log final performance metrics
        duration = time.perf_counter() - start_time
        log_performance(
            "streaming_document_processing_with_chunking",
            duration,
            file_count=total_files,
            processed_count=processed_count,
            chunked_count=chunked_count,
            error_count=error_count,
        )

        logger.success(
            "Streaming processing with chunking completed",
            extra={
                "file_count": total_files,
                "processed_count": processed_count,
                "chunked_count": chunked_count,
                "error_count": error_count,
                "success_rate": (processed_count / total_files) if total_files else 0,
                "duration_seconds": round(duration, 2),
            },
        )
