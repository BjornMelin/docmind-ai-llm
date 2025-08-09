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
import io
import logging
import os
import tempfile
from collections.abc import AsyncIterator
from typing import Any

import torch
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.schema import ImageDocument
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_parse import LlamaParse
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from unstructured.partition.auto import partition
from whisper import load_model as whisper_load

from models import AppSettings
from utils.model_manager import ModelManager

settings = AppSettings()


def extract_images_from_pdf(pdf_path: str) -> list[dict[str, Any]]:
    """Extract images from PDF files using PyMuPDF.

    Extracts all embedded images from a PDF document and converts them to
    base64-encoded PNG format for storage and processing. Handles various
    image formats and color spaces with automatic conversion.

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
        FileNotFoundError: If the PDF file does not exist.
        Exception: If image extraction fails for any reason.

    Note:
        Only processes images in GRAY or RGB color spaces to avoid
        corruption. CMYK and other complex color spaces are skipped.
        Images are converted to PNG format for consistent handling.

    Example:
        >>> images = extract_images_from_pdf("document.pdf")
        >>> for img in images:
        ...     print(f"Page {img['page_number']}: {img['size']} pixels")
        ...     # Use img['image_data'] for base64 image content
    """
    images = []
    try:
        from pathlib import Path

        import fitz  # PyMuPDF

        # Validate file exists and is readable
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

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

        logging.info("Extracted %s images from PDF", len(images))

    except Exception as e:
        logging.error("PDF image extraction failed: %s", e)

    return images


def create_native_multimodal_embeddings(
    text: str,
    images: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create multimodal embeddings using FastEmbed native implementation.

    Generates embeddings for both text and image content using FastEmbed's
    LateInteractionMultimodalEmbedding model. Provides unified embedding
    space for cross-modal similarity search and retrieval.

    Features:
    - FastEmbed native multimodal processing
    - Automatic provider detection (CUDA/CPU)
    - Memory-efficient batch processing
    - Temporary file management for images
    - Graceful fallbacks for missing components

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
    embeddings = {
        "text_embedding": None,
        "image_embeddings": [],
        "combined_embedding": None,
        "provider_used": None,
    }

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
                    # Native FastEmbed image embeddings
                    image_embeddings = list(model.embed_image(image_paths))

                    for i, img_emb in enumerate(image_embeddings):
                        embeddings["image_embeddings"].append(
                            {
                                "embedding": img_emb.flatten().tolist(),
                                "metadata": images[i] if i < len(images) else {},
                            }
                        )

                    # Clean up temporary files
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
            fastembed_model = FastEmbedEmbedding(
                model_name=settings.dense_embedding_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                cache_dir="./embeddings_cache",
            )
            embeddings["text_embedding"] = fastembed_model.get_text_embedding(text)
            embeddings["combined_embedding"] = embeddings["text_embedding"]
            embeddings["provider_used"] = "fastembed_text_only"

    except Exception as e:
        logging.error("Native multimodal embedding creation failed: %s", e)
        # Ultimate fallback to FastEmbed text-only
        try:
            fastembed_model = FastEmbedEmbedding(
                model_name=settings.dense_embedding_model,
                cache_dir="./embeddings_cache",
            )
            embeddings["text_embedding"] = fastembed_model.get_text_embedding(text)
            embeddings["combined_embedding"] = embeddings["text_embedding"]
            embeddings["provider_used"] = "fastembed_fallback"
        except Exception as fallback_e:
            logging.error("All embedding methods failed: %s", fallback_e)
            embeddings["provider_used"] = "failed"

    return embeddings


def load_documents_unstructured(file_path: str) -> list[Document]:
    """Load documents using Unstructured for multimodal parsing.

    Uses Unstructured library to extract text, images, tables, and other elements
    from documents with high-res strategy for best quality. Supports embedded
    image extraction and semantic chunking while preserving document structure.

    Args:
        file_path: Path to the document file to process.

    Returns:
        List of Document objects with multimodal elements and metadata.

    Raises:
        Exception: If document parsing fails, falls back to existing loader.
    """
    logger = logging.getLogger(__name__)
    try:
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
                        # Not base64 data, skip
                        continue

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

        # Apply additional chunking if documents are too large
        if settings.chunk_size > 0:
            documents = chunk_documents_structured(documents)

        logger.info(
            f"Loaded {len(documents)} multimodal elements from {file_path} "
            f"using Unstructured with strategy '{settings.parse_strategy}'"
        )
        return documents

    except Exception as e:
        logger.error(f"Error loading with Unstructured: {e}")
        logger.info("Falling back to existing LlamaParse loader")

        # Fall back to existing loader - create temporary file list for compatibility
        class TempFile:
            def __init__(self, path: str):
                self.name = os.path.basename(path)
                with open(path, "rb") as f:
                    self._content = f.read()

            def getvalue(self):
                return self._content

        temp_files = [TempFile(file_path)]
        return load_documents_llama(
            uploaded_files=temp_files, parse_media=False, enable_multimodal=True
        )


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


def load_documents_llama(
    uploaded_files: list[Any], parse_media: bool = False, enable_multimodal: bool = True
) -> list[Document]:
    """Load documents using LlamaParse with multimodal support.

    Supports standard document formats plus video/audio ingestion and PDF image
    extraction for multimodal processing with enhanced error handling.

    Args:
        uploaded_files: List of uploaded file objects.
        parse_media: Whether to parse video/audio files. Defaults to False.
        enable_multimodal: Whether to enable multimodal processing for PDFs.
            Defaults to True.

    Returns:
        List of loaded Document objects with multimodal embeddings where applicable.

    Raises:
        Exception: If critical document loading operations fail.
    """
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
            logging.error("File not found: %s - %s", file.name, str(e))
        except ValueError as e:
            logging.error("Invalid file format: %s - %s", file.name, str(e))
        except Exception as e:
            logging.error("Unexpected error loading %s: %s", file.name, str(e))
        finally:
            # Ensure temporary file is always cleaned up
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    logging.warning(
                        "Failed to cleanup temp file %s: %s", file_path, cleanup_error
                    )

    logging.info(
        f"Loaded {len(docs)} documents, multimodal processing enabled: "
        f"{enable_multimodal}"
    )
    return docs


# Streaming Document Processing for Async Performance


async def stream_document_processing(file_paths: list[str]) -> AsyncIterator[Document]:
    """Stream document processing for large document sets.

    Processes documents asynchronously and yields results as they
    become available, reducing memory usage and improving responsiveness.
    Provides significant performance improvements for large document collections.

    Args:
        file_paths: List of file paths to process.

    Yields:
        Processed Document objects as they become available.

    Note:
        Uses semaphore to limit concurrent processing and prevent resource exhaustion.
        Falls back gracefully on individual document processing failures.
    """
    semaphore = asyncio.Semaphore(5)  # Limit concurrent processing

    async def process_single_file(file_path: str) -> list[Document]:
        """Process a single file with semaphore limiting."""
        async with semaphore:
            try:
                return await asyncio.to_thread(load_documents_unstructured, file_path)
            except Exception as e:
                logging.error(f"Document processing failed for {file_path}: {e}")
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
                yield doc
        except Exception as e:
            logging.error(f"Document processing failed: {e}")
            # Continue with other documents
            continue


async def batch_embed_documents(
    documents: list[Document], batch_size: int = 32
) -> list[list[float]]:
    """Batch document embedding with parallel processing.

    Embeds documents in parallel batches for optimal performance.
    Uses asyncio.to_thread to prevent blocking the event loop during
    CPU-intensive embedding operations.

    Args:
        documents: Documents to embed.
        batch_size: Size of processing batches for optimal memory usage.

    Returns:
        List of embedding vectors corresponding to input documents.

    Note:
        Falls back gracefully on batch failures by providing zero embeddings.
        Processes batches in parallel for maximum throughput.
    """
    from utils.utils import get_embed_model

    embed_model = get_embed_model()

    # Split into batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]

    async def embed_batch(batch: list[Document]) -> list[list[float]]:
        """Embed a single batch of documents."""
        try:
            texts = [doc.text for doc in batch]
            return await asyncio.to_thread(embed_model.embed, texts)
        except Exception as e:
            logging.error(f"Batch embedding failed: {e}")
            # Add placeholder embeddings for failed batch
            settings_instance = settings
            return [[0.0] * settings_instance.dense_embedding_dimension] * len(batch)

    # Process batches in parallel
    batch_results = await asyncio.gather(
        *[embed_batch(batch) for batch in batches], return_exceptions=True
    )

    # Combine results
    all_embeddings = []
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logging.error(f"Batch {i} embedding failed: {result}")
            # Add placeholder embeddings for failed batch
            failed_batch_size = len(batches[i])
            settings_instance = settings
            all_embeddings.extend(
                [[0.0] * settings_instance.dense_embedding_dimension]
                * failed_batch_size
            )
        else:
            all_embeddings.extend(result)

    logging.info(
        f"Embedded {len(all_embeddings)} documents in {len(batches)} parallel batches"
    )
    return all_embeddings


async def process_documents_streaming(
    file_paths: list[str], chunk_size: int = 1024, chunk_overlap: int = 200
) -> AsyncIterator[Document]:
    """Process documents with streaming and chunking for memory efficiency.

    Combines streaming document processing with intelligent chunking to
    handle large document collections efficiently. Provides real-time
    progress updates and memory-conscious processing.

    Args:
        file_paths: List of file paths to process.
        chunk_size: Maximum chunk size for document splitting.
        chunk_overlap: Overlap between chunks for context preservation.

    Yields:
        Processed and chunked Document objects as they become available.

    Note:
        Automatically applies semantic chunking to large documents while
        preserving multimodal content. Streams results for immediate processing.
    """
    processed_count = 0
    total_files = len(file_paths)

    async for document in stream_document_processing(file_paths):
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
                yield chunk
        else:
            # Document is small enough, yield as-is
            yield document

        processed_count += 1
        if processed_count % 10 == 0:  # Log progress every 10 documents
            logging.info(f"Processed {processed_count}/{total_files} documents")

    logging.info(
        f"Streaming processing completed: {processed_count}/{total_files} documents"
    )
