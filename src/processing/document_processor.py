"""Hybrid Document Processor combining unstructured and LlamaIndex IngestionPipeline.

This module implements a unified document processor that combines
Unstructured.io + LlamaIndex IngestionPipeline:
- Direct unstructured.partition.auto.partition() for document extraction
- LlamaIndex IngestionPipeline for orchestration, caching, and transformations
- Strategy-based processing with comprehensive error handling
- Full API compatibility with ResilientDocumentProcessor

Key Features:
- Unstructured.io parsing with hi_res strategy and multimodal extraction
- LlamaIndex IngestionPipeline for built-in caching and async processing
- Custom UnstructuredTransformation for seamless integration
- Strategy mapping (hi_res, fast, ocr_only) based on file type
- Complete metadata preservation and transformation pipeline
- Async processing with retry logic and error handling
- Performance target: >1 page/second with hi_res strategy
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from unstructured.partition.auto import partition

from src.cache.simple_cache import SimpleCache
from src.models.processing import DocumentElement, ProcessingResult, ProcessingStrategy


class ProcessingError(Exception):
    """Custom exception for document processing errors."""


class UnstructuredTransformation(TransformComponent):
    """Custom LlamaIndex transformation that uses unstructured.io for document parsing.

    This transformation integrates unstructured.io directly into the LlamaIndex
    IngestionPipeline, providing strategy-based parsing with full metadata preservation.
    """

    # Declare Pydantic fields for the strategy and settings
    strategy: ProcessingStrategy = None
    settings: Any = None

    def __init__(self, strategy: ProcessingStrategy, settings: Any | None = None):
        """Initialize UnstructuredTransformation.

        Args:
            strategy: Processing strategy to use for unstructured.io
            settings: DocMind configuration settings
        """
        super().__init__()
        self.strategy = strategy
        self.settings = settings or settings
        logger.debug(
            f"UnstructuredTransformation initialized with strategy: {strategy}"
        )

    def __call__(self, nodes: list[BaseNode], **kwargs) -> list[BaseNode]:
        """Transform Document nodes using unstructured.io parsing.

        Args:
            nodes: List of Document nodes to transform
            **kwargs: Additional transformation arguments

        Returns:
            List of transformed BaseNode objects with unstructured elements

        Raises:
            ProcessingError: If document processing fails
        """
        transformed_nodes = []

        for node in nodes:
            if not isinstance(node, Document):
                # Pass through non-Document nodes unchanged
                transformed_nodes.append(node)
                continue

            try:
                # Get file path from metadata or node content
                file_path = self._extract_file_path(node)
                if not file_path:
                    logger.warning(
                        "No file path found in node, skipping unstructured parsing"
                    )
                    transformed_nodes.append(node)
                    continue

                # Build partition configuration for this strategy
                partition_config = self._build_partition_config(self.strategy)

                # Process document with unstructured.io
                elements = partition(filename=str(file_path), **partition_config)

                # Convert unstructured elements to Document nodes
                element_nodes = self._convert_elements_to_nodes(
                    elements, node, file_path
                )
                transformed_nodes.extend(element_nodes)

                logger.debug(
                    f"Processed {file_path}: {len(elements)} elements -> "
                    f"{len(element_nodes)} nodes"
                )

            except (OSError, ProcessingError, ValueError) as e:
                logger.error(f"UnstructuredTransformation failed for node: {str(e)}")
                # Include original node on error to maintain pipeline flow
                transformed_nodes.append(node)
            except Exception as e:
                logger.error(
                    f"Unexpected error in UnstructuredTransformation: {str(e)}"
                )
                # Include original node on error to maintain pipeline flow
                transformed_nodes.append(node)

        return transformed_nodes

    def _extract_file_path(self, node: Document) -> Path | None:
        """Extract file path from Document node metadata.

        Args:
            node: Document node to extract path from

        Returns:
            Path object if found, None otherwise
        """
        # Check various metadata fields for file path
        metadata = node.metadata or {}

        # Common metadata fields where file path might be stored
        path_fields = ["file_path", "filename", "source", "file_name", "path"]

        for field in path_fields:
            if field in metadata and metadata[field]:
                path = Path(metadata[field])
                if path.exists():
                    return path

        return None

    def _build_partition_config(self, strategy: ProcessingStrategy) -> dict[str, Any]:
        """Build partition configuration for unstructured.io.

        Args:
            strategy: Processing strategy to use

        Returns:
            Configuration dictionary for partition() call
        """
        # Base configuration for all strategies
        config = {
            "strategy": strategy.value,
            "include_metadata": True,
            "include_page_breaks": True,
        }

        # Strategy-specific configuration
        if strategy == ProcessingStrategy.HI_RES:
            # Hi-res strategy for full multimodal extraction
            config.update(
                {
                    "extract_images_in_pdf": True,
                    "extract_image_blocks": True,
                    "infer_table_structure": True,
                    "chunking_strategy": "by_title",
                    "multipage_sections": True,
                    "combine_text_under_n_chars": 500,
                    "new_after_n_chars": 1200,
                    "max_characters": 1500,
                }
            )
        elif strategy == ProcessingStrategy.FAST:
            # Fast strategy for quick text extraction
            config.update(
                {
                    "extract_images_in_pdf": False,
                    "infer_table_structure": False,
                    "chunking_strategy": "basic",
                    "max_characters": 1000,
                }
            )
        elif strategy == ProcessingStrategy.OCR_ONLY:
            # OCR strategy for image-based documents
            config.update(
                {
                    "extract_images_in_pdf": True,
                    "extract_image_blocks": True,
                    "infer_table_structure": False,
                    "ocr_languages": ["eng"],
                }
            )

        return config

    def _convert_elements_to_nodes(
        self, elements: list[Any], original_node: Document, file_path: Path
    ) -> list[Document]:
        """Convert unstructured elements to Document nodes.

        Args:
            elements: Raw elements from unstructured.partition
            original_node: Original Document node for metadata inheritance
            file_path: Path to the source document

        Returns:
            List of Document nodes with unstructured element data
        """
        nodes = []

        for i, element in enumerate(elements):
            # Extract metadata dictionary from unstructured element
            element_metadata = {}
            if hasattr(element, "metadata") and element.metadata:
                element_metadata = {
                    "page_number": getattr(element.metadata, "page_number", None),
                    "element_id": getattr(element.metadata, "element_id", None),
                    "parent_id": getattr(element.metadata, "parent_id", None),
                    "filename": getattr(element.metadata, "filename", None),
                    "coordinates": getattr(element.metadata, "coordinates", None),
                    "text_as_html": getattr(element.metadata, "text_as_html", None),
                    "image_path": getattr(element.metadata, "image_path", None),
                }
                # Remove None values
                element_metadata = {
                    k: v for k, v in element_metadata.items() if v is not None
                }

            # Combine original node metadata with element metadata
            combined_metadata = {
                **(original_node.metadata or {}),
                **element_metadata,
                "element_index": i,
                "element_category": str(element.category)
                if hasattr(element, "category")
                else "Unknown",
                "processing_strategy": self.strategy.value,
                "source_file": str(file_path),
            }

            # Create Document node for this element
            element_node = Document(
                text=str(element.text) if element.text else "",
                metadata=combined_metadata,
                # Preserve original node's excluded metadata keys and relationships
                excluded_embed_metadata_keys=original_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=original_node.excluded_llm_metadata_keys,
                metadata_seperator=original_node.metadata_seperator,
                metadata_template=original_node.metadata_template,
                text_template=original_node.text_template,
            )

            nodes.append(element_node)

        return nodes


class DocumentProcessor:
    """Hybrid document processor combining unstructured + LlamaIndex IngestionPipeline.

    This processor provides the best of both worlds:
    - Direct unstructured.io integration for comprehensive document parsing
    - LlamaIndex IngestionPipeline for built-in caching, async, and transformations
    - Strategy-based processing with bulletproof error handling
    - Full API compatibility with ResilientDocumentProcessor
    """

    def __init__(self, settings: Any | None = None):
        """Initialize DocumentProcessor.

        Args:
            settings: DocMind configuration settings. Uses settings if None.
        """
        self.settings = settings or settings
        self._config_override = {}

        # Strategy mapping based on file extensions
        self.strategy_map = {
            ".pdf": ProcessingStrategy.HI_RES,
            ".docx": ProcessingStrategy.HI_RES,
            ".doc": ProcessingStrategy.HI_RES,
            ".pptx": ProcessingStrategy.HI_RES,
            ".ppt": ProcessingStrategy.HI_RES,
            ".html": ProcessingStrategy.FAST,
            ".htm": ProcessingStrategy.FAST,
            ".txt": ProcessingStrategy.FAST,
            ".md": ProcessingStrategy.FAST,
            ".rtf": ProcessingStrategy.FAST,
            ".jpg": ProcessingStrategy.OCR_ONLY,
            ".jpeg": ProcessingStrategy.OCR_ONLY,
            ".png": ProcessingStrategy.OCR_ONLY,
            ".tiff": ProcessingStrategy.OCR_ONLY,
            ".bmp": ProcessingStrategy.OCR_ONLY,
        }

        # Initialize cache for LlamaIndex IngestionPipeline
        self.cache = IngestionCache(
            collection="docmind_processing",
            cache_dir=str(getattr(self.settings, "cache_dir", "./cache")),
        )

        # Document store for document management and deduplication
        self.docstore = SimpleDocumentStore()

        # Initialize SimpleCache for compatibility
        self.simple_cache = SimpleCache(
            cache_dir=str(getattr(self.settings, "cache_dir", "./cache"))
        )

        logger.info(
            "DocumentProcessor initialized with strategy mapping for "
            f"{len(self.strategy_map)} file types"
        )

    def _get_strategy_for_file(self, file_path: str | Path) -> ProcessingStrategy:
        """Determine processing strategy based on file extension.

        Args:
            file_path: Path to the document file

        Returns:
            ProcessingStrategy enum value

        Raises:
            ValueError: If file extension is not supported
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self.strategy_map:
            supported_extensions = ", ".join(sorted(self.strategy_map.keys()))
            raise ValueError(
                f"Unsupported file format '{extension}'. "
                f"Supported formats: {supported_extensions}"
            )

        strategy = self.strategy_map[extension]
        logger.debug(f"Selected strategy '{strategy}' for file: {file_path}")
        return strategy

    def get_strategy_for_file(self, file_path: str | Path) -> ProcessingStrategy:
        """Public method to determine processing strategy based on file extension.

        Args:
            file_path: Path to the document file

        Returns:
            ProcessingStrategy enum value

        Raises:
            ValueError: If file extension is not supported
        """
        return self._get_strategy_for_file(file_path)

    def _calculate_document_hash(self, file_path: str | Path) -> str:
        """Calculate unique hash for document caching.

        Args:
            file_path: Path to the document file

        Returns:
            SHA-256 hash string of file content and metadata
        """
        file_path = Path(file_path)

        # Hash file content + metadata for cache key
        hasher = hashlib.sha256()

        # Include file content
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        # Include file metadata
        stat = file_path.stat()
        metadata = f"{file_path.name}:{stat.st_size}:{stat.st_mtime}".encode()
        hasher.update(metadata)

        return hasher.hexdigest()

    def _create_pipeline(self, strategy: ProcessingStrategy) -> IngestionPipeline:
        """Create LlamaIndex IngestionPipeline with UnstructuredTransformation.

        Args:
            strategy: Processing strategy to use

        Returns:
            Configured IngestionPipeline instance
        """
        # Create transformations pipeline
        transformations = [
            # First transformation: parse document with unstructured
            UnstructuredTransformation(strategy, self.settings),
            # Second transformation: split into semantic chunks
            SentenceSplitter(
                chunk_size=self.settings.processing.chunk_size,
                chunk_overlap=self.settings.processing.chunk_overlap,
                include_metadata=True,
                include_prev_next_rel=True,
            ),
        ]

        # Create pipeline with all LlamaIndex benefits
        pipeline = IngestionPipeline(
            transformations=transformations,
            cache=self.cache,  # Built-in caching
            docstore=self.docstore,  # Document deduplication
        )

        return pipeline

    @retry(
        retry=retry_if_exception_type((IOError, OSError, ProcessingError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def process_document_async(
        self,
        file_path: str | Path,
        config_override: dict[str, Any] | None = None,
    ) -> ProcessingResult:
        """Process document asynchronously with hybrid approach.

        Args:
            file_path: Path to the document file
            config_override: Optional configuration overrides

        Returns:
            ProcessingResult with extracted elements and metadata

        Raises:
            ProcessingError: If document processing fails
            ValueError: If file format is unsupported
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.info(f"Processing document with hybrid processor: {file_path}")

        # Validate file exists and size
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        max_size = getattr(self.settings, "max_document_size_mb", 100)
        if file_size_mb > max_size:
            raise ProcessingError(
                f"Document size ({file_size_mb:.1f}MB) exceeds limit ({max_size}MB)"
            )

        try:
            # Determine processing strategy
            strategy = self._get_strategy_for_file(file_path)

            # Check cache first using SimpleCache for compatibility
            cached_result = await self.simple_cache.get_document(str(file_path))
            if cached_result:
                logger.info(f"Retrieved cached result for: {file_path}")
                return cached_result

            # Create pipeline for this strategy
            pipeline = self._create_pipeline(strategy)

            # Create Document object for LlamaIndex processing
            document = Document(
                text="",  # Will be populated by UnstructuredTransformation
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "source": str(file_path),
                    "strategy": strategy.value,
                },
            )

            # Process document through pipeline (async operation)
            nodes = await asyncio.to_thread(
                pipeline.run, documents=[document], show_progress=False
            )

            # Convert LlamaIndex nodes to DocumentElements for compatibility
            processed_elements = self._convert_nodes_to_elements(nodes)

            processing_time = time.time() - start_time
            document_hash = self._calculate_document_hash(file_path)

            result = ProcessingResult(
                elements=processed_elements,
                processing_time=processing_time,
                strategy_used=strategy,
                metadata={
                    "file_path": str(file_path),
                    "file_size_mb": file_size_mb,
                    "element_count": len(processed_elements),
                    "node_count": len(nodes),
                    "pipeline_config": {
                        "strategy": strategy.value,
                        "transformations": len(pipeline.transformations),
                        "cache_enabled": True,
                        "docstore_enabled": True,
                    },
                    "cache_stats": {
                        "hits": getattr(pipeline.cache, "hits", 0),
                        "misses": getattr(pipeline.cache, "misses", 0),
                    },
                },
                document_hash=document_hash,
            )

            # Store in cache for future use
            await self.simple_cache.store_document(str(file_path), result)

            logger.info(
                f"Successfully processed {file_path}: {len(processed_elements)} "
                f"elements from {len(nodes)} nodes in {processing_time:.2f}s "
                f"(strategy: {strategy})"
            )

            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"Failed to process {file_path} after {processing_time:.2f}s: {str(e)}"
            )

            if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                raise ProcessingError(f"Document appears to be corrupted: {e}") from e

            raise ProcessingError(f"Document processing failed: {e}") from e

    def _convert_nodes_to_elements(
        self, nodes: list[BaseNode]
    ) -> list[DocumentElement]:
        """Convert LlamaIndex nodes to DocumentElement objects.

        Args:
            nodes: List of processed LlamaIndex nodes

        Returns:
            List of DocumentElement objects for API compatibility
        """
        elements = []

        for node in nodes:
            # Extract metadata from node
            metadata = {}
            if hasattr(node, "metadata") and node.metadata:
                # Preserve all metadata from the node
                metadata = dict(node.metadata)

            # Create DocumentElement for compatibility with existing API
            element = DocumentElement(
                text=node.get_content()
                if hasattr(node, "get_content")
                else str(node.text),
                category=metadata.get("element_category", "Text"),
                metadata=metadata,
            )

            elements.append(element)

        return elements

    def override_config(self, config: dict[str, Any]) -> None:
        """Override configuration for processing operations.

        Args:
            config: Configuration dictionary to apply
        """
        # Store configuration for use in pipeline creation
        self._config_override.update(config)
        logger.info(f"Configuration override applied: {config}")

    async def clear_cache(self) -> bool:
        """Clear processing cache.

        Returns:
            True if cache was cleared successfully
        """
        try:
            # Clear LlamaIndex cache (if possible)
            if hasattr(self.cache, "clear"):
                self.cache.clear()
            elif hasattr(self.cache, "delete_all"):
                self.cache.delete_all()
            else:
                logger.warning("LlamaIndex cache does not support clearing")

            # Clear SimpleCache
            await self.simple_cache.clear_cache()

            logger.info("Processing cache cleared successfully")
            return True
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error clearing cache: {e}")
            return False

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        try:
            simple_stats = await self.simple_cache.get_cache_stats()

            return {
                "processor_type": "hybrid",
                "simple_cache": simple_stats,
                "llamaindex_cache": {
                    "cache_enabled": True,
                    "docstore_enabled": True,
                    "collection": self.cache.collection,
                },
                "strategy_mappings": len(self.strategy_map),
            }
        except (OSError, RuntimeError) as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e), "processor_type": "hybrid"}
        except Exception as e:
            logger.error(f"Unexpected error getting cache stats: {e}")
            return {"error": str(e), "processor_type": "hybrid"}
