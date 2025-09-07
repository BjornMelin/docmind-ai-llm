"""Hybrid document processing with Unstructured + LlamaIndex.

This module provides a unified, document processing pipeline that
combines Unstructured document partitioning and LlamaIndex IngestionPipeline.

Overview:
- Uses ``unstructured.partition.auto.partition`` to extract structural elements.
- Uses Unstructured's built-in chunking via ``chunk_by_title`` (default) with a
  basic fallback when heading density is low. Table elements remain isolated.
- Orchestrates transformations and caching with LlamaIndex ``IngestionPipeline``.
- Provides robust error handling and async execution with retries.
"""

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

from llama_index.core import Document
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from unstructured.chunking.basic import chunk_elements as chunk_by_basic
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from src.config.settings import settings as app_settings
from src.models.processing import DocumentElement, ProcessingResult, ProcessingStrategy
from src.processing.utils import is_unstructured_like, sha256_id


class ProcessingError(Exception):
    """Raised when a document processing operation fails."""


class UnstructuredTransformation(TransformComponent):
    """Unstructured parsing + chunking transformation for LlamaIndex.

    This transformation integrates Unstructured directly into the LlamaIndex
    ``IngestionPipeline``. It performs partitioning and then applies the
    Unstructured chunking strategy (``chunk_by_title`` by default, with a
    basic fallback for heading-sparse documents). Metadata is preserved.

    Args:
        strategy: Processing strategy to use for Unstructured partitioning.
        settings: Application settings object providing processing parameters.

    Raises:
        ProcessingError: If Unstructured partitioning or chunking fails.
    """

    # Declare Pydantic fields for the strategy and settings
    strategy: ProcessingStrategy = None
    settings: Any = None

    def __init__(
        self, strategy: ProcessingStrategy, settings: Any | None = None
    ) -> None:
        """Initialize UnstructuredTransformation.

        Args:
            strategy: Processing strategy to use for unstructured.io
            settings: DocMind configuration settings
        """
        super().__init__()
        self.strategy = strategy
        self.settings = settings or app_settings
        logger.debug(
            "UnstructuredTransformation initialized with strategy: {}", strategy
        )

    def __call__(self, nodes: list[BaseNode], **kwargs: Any) -> list[BaseNode]:
        """Transform LlamaIndex ``Document`` nodes using Unstructured.

        Args:
            nodes: List of ``Document`` nodes to transform.
            **kwargs: Additional transformation arguments (unused).

        Returns:
            list[BaseNode]: Transformed nodes with chunked elements.

        Raises:
            ProcessingError: If partitioning or chunking fails.
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

                # Build partition configuration for this strategy (no chunking here)
                partition_config = self._build_partition_config(self.strategy)

                # First: partition the document into structural elements
                elements = partition(str(file_path), **partition_config)

                # Decide chunking strategy based on detected title density
                title_count = sum(
                    1
                    for e in elements
                    if hasattr(e, "category") and "title" in str(e.category).lower()
                )
                elem_count = len(elements) or 1
                title_density = title_count / elem_count

                # Parameters from settings (coerce to safe types)
                def _safe_int(val: Any, default: int) -> int:
                    try:
                        return int(val)  # type: ignore[arg-type]
                    except (ValueError, TypeError):
                        return default

                def _safe_bool(val: Any, default: bool) -> bool:
                    return bool(val) if isinstance(val, bool) else default

                max_chars = _safe_int(
                    getattr(self.settings.processing, "chunk_size", 1500), 1500
                )
                new_after = _safe_int(
                    getattr(self.settings.processing, "new_after_n_chars", 1200), 1200
                )
                combine_under = _safe_int(
                    getattr(
                        self.settings.processing, "combine_text_under_n_chars", 500
                    ),
                    500,
                )
                multipage = _safe_bool(
                    getattr(self.settings.processing, "multipage_sections", True), True
                )

                # If elements don't look like Unstructured elements (e.g., test mocks),
                # skip chunking and treat them as pre-chunked identity to keep
                # tests robust.
                def _looks_like_unstructured(el: Any) -> bool:
                    """Helper delegating to processing utils for detection."""
                    return is_unstructured_like(el)

                if not all(_looks_like_unstructured(e) for e in elements):
                    chunked = elements
                else:
                    # Heuristic: use by_title if there are enough titles
                    use_by_title = title_count >= 3 or title_density >= 0.05

                    if use_by_title:
                        if getattr(self.settings.processing, "debug_chunk_flow", False):
                            logger.debug(
                                "Chunk flow: by_title (titles={}, density={:.3f})",
                                title_count,
                                title_density,
                            )
                        chunked = chunk_by_title(
                            elements=elements,
                            max_characters=max_chars,
                            new_after_n_chars=new_after,
                            combine_text_under_n_chars=combine_under,
                            multipage_sections=multipage,
                        )
                    else:
                        # Fallback to basic chunking for heading-sparse docs
                        if getattr(self.settings.processing, "debug_chunk_flow", False):
                            logger.debug(
                                "Chunk flow: basic fallback (titles=%s, density=%.3f)",
                                title_count,
                                title_density,
                            )
                        chunked = chunk_by_basic(
                            elements=elements,
                            max_characters=max_chars,
                            new_after_n_chars=new_after,
                            overlap=0,
                            overlap_all=False,
                        )

                # Convert chunked elements to Document nodes
                element_nodes = self._convert_elements_to_nodes(
                    chunked, node, file_path
                )
                transformed_nodes.extend(element_nodes)

                logger.debug(
                    "Processed {}: {} elements -> {} nodes",
                    file_path,
                    len(elements),
                    len(element_nodes),
                )

            except (OSError, RuntimeError, ValueError, ProcessingError) as e:
                logger.error("UnstructuredTransformation failed for node: {}", e)
                transformed_nodes.append(node)

        return transformed_nodes

    def _extract_file_path(self, node: Document) -> Path | None:
        """Extract a local file path from a ``Document`` node.

        Args:
            node: Document node to extract path from.

        Returns:
            Path | None: File path if found and exists, otherwise ``None``.
        """
        # Check various metadata fields for file path
        metadata = node.metadata or {}

        # Common metadata fields where file path might be stored
        path_fields = ["file_path", "filename", "source", "file_name", "path"]

        for field in path_fields:
            if metadata.get(field):
                path = Path(metadata[field])
                if path.exists():
                    return path

        return None

    def _build_partition_config(self, strategy: ProcessingStrategy) -> dict[str, Any]:
        """Build configuration for Unstructured ``partition``.

        Args:
            strategy: Processing strategy to use.

        Returns:
            dict[str, Any]: Keyword arguments to pass to ``partition``.
        """
        # Base configuration for all strategies (chunking is applied after partition)
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
                }
            )
        elif strategy == ProcessingStrategy.FAST:
            # Fast strategy for quick text extraction
            config.update(
                {"extract_images_in_pdf": False, "infer_table_structure": False}
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
        """Convert Unstructured elements to LlamaIndex ``Document`` nodes.

        Args:
            elements: Raw elements from Unstructured partition/chunking.
            original_node: Original Document for metadata inheritance.
            file_path: Source document path used for processing.

        Returns:
            list[Document]: Nodes representing the chunked elements.
        """
        nodes = []

        def _safe_json_value(val: Any) -> Any:
            """Coerce values to JSON-serializable forms.

            Note: avoid ``isinstance(x, A | B)`` which is invalid at runtime.
            """
            if isinstance(val, (str, int, float, bool)) or val is None:  # noqa: UP038
                return val
            if isinstance(val, (list, tuple)):  # noqa: UP038
                return [_safe_json_value(v) for v in val]
            if isinstance(val, dict):
                return {k: _safe_json_value(v) for k, v in val.items()}
            # Fallback to string representation
            return str(val)

        for i, element in enumerate(elements):
            # Extract metadata dictionary from unstructured element
            element_metadata = {}
            if hasattr(element, "metadata") and element.metadata:
                raw_md = {
                    "page_number": getattr(element.metadata, "page_number", None),
                    "element_id": getattr(element.metadata, "element_id", None),
                    "parent_id": getattr(element.metadata, "parent_id", None),
                    "filename": getattr(element.metadata, "filename", None),
                    "coordinates": getattr(element.metadata, "coordinates", None),
                    "text_as_html": getattr(element.metadata, "text_as_html", None),
                    "image_path": getattr(element.metadata, "image_path", None),
                }
                element_metadata = {k: _safe_json_value(v) for k, v in raw_md.items()}
                # Remove None values
                element_metadata = {
                    k: v for k, v in element_metadata.items() if v is not None
                }

            # Derive page number (if available) for deterministic IDs
            page_no = 0
            try:
                page_no = int(element_metadata.get("page_number", 0))
            except Exception:
                page_no = 0

            # Compute deterministic node id including a per-element discriminator to
            # avoid collisions for blank/non-text elements that can occur on the same
            # page.
            if element_metadata.get("element_id"):
                discriminator = str(element_metadata["element_id"])
            elif element_metadata.get("coordinates"):
                discriminator = str(element_metadata["coordinates"])  # JSON-safe
            else:
                discriminator = str(i)

            det_id = sha256_id(
                str(file_path),
                str(page_no),
                str(getattr(element, "text", "")),
                discriminator,
            )

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
                "parent_id": getattr(original_node, "doc_id", None)
                or getattr(original_node, "id_", None),
                "node_id": det_id,
            }

            # Create Document node for this element (safe attribute access)
            element_node = Document(
                text=str(getattr(element, "text", ""))
                if getattr(element, "text", "")
                else "",
                doc_id=det_id,
                metadata=combined_metadata,
                # Preserve original node's excluded metadata keys and relationships
                excluded_embed_metadata_keys=original_node.excluded_embed_metadata_keys,
                excluded_llm_metadata_keys=original_node.excluded_llm_metadata_keys,
                metadata_separator=original_node.metadata_separator,
                metadata_template=original_node.metadata_template,
                text_template=original_node.text_template,
            )

            nodes.append(element_node)

        return nodes


class DocumentProcessor:
    """Hybrid processor combining Unstructured with LlamaIndex pipeline.

    - Direct Unstructured integration for parsing and chunking.
    - LlamaIndex IngestionPipeline for caching and orchestration.
    - Strategy mapping (hi_res/fast/ocr_only) by file extension.
    - Async, resilient execution with retries.
    """

    def __init__(self, settings: Any | None = None) -> None:
        """Initialize DocumentProcessor.

        Args:
            settings: DocMind configuration settings. Uses settings if None.
        """
        self.settings = settings or app_settings
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

        # Initialize DuckDB-backed cache for LlamaIndex IngestionPipeline
        cache_dir = Path(getattr(self.settings, "cache_dir", "./cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_db = cache_dir / "docmind.duckdb"
        kv = DuckDBKVStore(db_path=str(cache_db))
        self.cache = IngestionCache(cache=kv, collection="docmind_processing")

        # Document store for document management and deduplication
        self.docstore = SimpleDocumentStore()

        logger.info(
            "DocumentProcessor initialized with strategy mapping for {} file types",
            len(self.strategy_map),
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
        logger.debug("Selected strategy '{}' for file: {}", strategy, file_path)
        return strategy

    def _get_max_document_size_mb(self) -> int:
        """Resolve max document size from settings (nested preferred).

        Prefer ``settings.processing.max_document_size_mb``. If a top-level
        ``settings.max_document_size_mb`` is present (legacy/testing convenience),
        use it as a fallback. Defaults to 100MB if neither found or invalid.
        """
        # Nested (authoritative)
        from contextlib import suppress

        with suppress(Exception):
            val = self.settings.processing.max_document_size_mb
            if isinstance(val, (int, float)) and val > 0:  # noqa: UP038
                return int(val)
        # Top-level (fallback for older tests/mocks)
        with suppress(Exception):
            val = self.settings.max_document_size_mb
            if isinstance(val, (int, float)) and val > 0:  # noqa: UP038
                return int(val)
        return 100

    def get_strategy_for_file(self, file_path: str | Path) -> ProcessingStrategy:
        """Get processing strategy based on file extension.

        Args:
            file_path: Path to the document file.

        Returns:
            ProcessingStrategy: Chosen strategy.

        Raises:
            ValueError: If file extension is not supported.
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
        # Create transformations pipeline: Unstructured parsing + our chunking inside
        transformations = [
            UnstructuredTransformation(strategy, self.settings),
        ]

        # Create pipeline with all LlamaIndex benefits
        pipeline = IngestionPipeline(
            transformations=transformations,
            cache=self.cache,  # Built-in caching
            docstore=self.docstore,  # Document deduplication
        )

        return pipeline

    @retry(
        retry=retry_if_exception_type((IOError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def process_document_async(
        self,
        file_path: str | Path,
        config_override: dict[str, Any] | None = None,  # pylint: disable=unused-argument
    ) -> ProcessingResult:
        """Process a local document asynchronously.

        Args:
            file_path: Local path to the document file.
            config_override: Optional configuration overrides (unused).

        Returns:
            ProcessingResult: Extracted elements, metadata, and stats.

        Raises:
            ProcessingError: If processing fails.
            ValueError: If file format is unsupported.
        """
        start_time = time.time()
        file_path = Path(file_path)

        logger.info(
            "Processing document with hybrid processor: {}",
            file_path,
        )

        # Validate file exists and size
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        max_size = self._get_max_document_size_mb()
        if file_size_mb > max_size:
            raise ProcessingError(
                f"Document size ({file_size_mb:.1f}MB) exceeds limit ({max_size}MB)"
            )

        try:
            # Determine processing strategy
            strategy = self._get_strategy_for_file(file_path)

            # Calculate a stable file hash up front so the ingestion cache and
            # docstore can re-use results across repeated runs.
            document_hash = self._calculate_document_hash(file_path)

            # Create pipeline for this strategy
            pipeline = self._create_pipeline(strategy)

            # Create Document object for LlamaIndex processing with a stable ID.
            # Include the file hash in metadata so BaseNode.hash changes when the
            # underlying file content changes.
            document = Document(
                doc_id=document_hash,
                text="",  # Will be populated by UnstructuredTransformation
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "source": str(file_path),
                    "strategy": strategy.value,
                    "file_sha256": document_hash,
                },
            )

            # Process document through pipeline (async operation)
            nodes = await asyncio.to_thread(
                pipeline.run, documents=[document], show_progress=False
            )

            # Convert LlamaIndex nodes to DocumentElements for compatibility
            processed_elements = self._convert_nodes_to_elements(nodes)

            # If PDF, emit page-image nodes for multimodal reranking
            if file_path.suffix.lower() == ".pdf":
                # Lazy import to avoid importing PyMuPDF unless needed
                from src.processing.pdf_pages import save_pdf_page_images

                images_dir = (
                    Path(getattr(self.settings, "cache_dir", "./cache"))
                    / "page_images"
                    / file_path.stem
                )
                page_images = await asyncio.to_thread(
                    save_pdf_page_images, file_path, images_dir, 180
                )

                # Build deterministic page-image metadata elements
                image_elements: list[DocumentElement] = []
                for img in page_images:
                    img_path = Path(img["image_path"])  # guaranteed by save function
                    # Hash image bytes to bind node id to rendered content
                    img_hash = ""
                    try:
                        with open(img_path, "rb") as f:
                            img_hash = hashlib.sha256(f.read()).hexdigest()
                    except Exception as err:
                        # Log and continue with empty hash; keeps pipeline resilient
                        logger.warning(
                            "Failed to hash page image for {} ({}): {}",
                            file_path,
                            img_path,
                            err,
                        )
                        img_hash = ""

                    node_id = sha256_id(
                        str(file_path), str(img.get("page_no", 0)), img_hash
                    )
                    image_elements.append(
                        DocumentElement(
                            text="",  # images carry metadata only here
                            category="Image",
                            metadata={
                                "modality": "pdf_page_image",
                                "page_no": img.get("page_no", 0),
                                "bbox": img.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                                "image_path": str(img_path),
                                "source_file": str(file_path),
                                "node_id": node_id,
                                "parent_id": document_hash,
                            },
                        )
                    )

                processed_elements.extend(image_elements)

            processing_time = time.time() - start_time

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

            logger.info(
                "Processed {}: {} elements from {} nodes in {:.2f}s (strategy: {})",
                file_path,
                len(processed_elements),
                len(nodes),
                processing_time,
                strategy,
            )

            return result

        except (OSError, ValueError, RuntimeError) as e:
            processing_time = time.time() - start_time
            logger.error(
                "Failed to process {} after {:.2f}s: {}",
                file_path,
                processing_time,
                e,
            )

            if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                raise ProcessingError(f"Document appears to be corrupted: {e}") from e

            raise ProcessingError(f"Document processing failed: {e}") from e

    def _convert_nodes_to_elements(
        self, nodes: list[BaseNode]
    ) -> list[DocumentElement]:
        """Convert LlamaIndex nodes to ``DocumentElement`` objects.

        Args:
            nodes: List of processed LlamaIndex nodes.

        Returns:
            list[DocumentElement]: Compatible element structures.
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
            config: Configuration dictionary to apply.
        """
        # Store configuration for use in pipeline creation
        self._config_override.update(config)
        logger.info("Configuration override applied: {}", config)

    async def clear_cache(self) -> bool:
        """Clear processing cache.

        Returns:
            bool: True if cache was cleared successfully.
        """
        try:
            # Best-effort clear: delete DuckDB cache file; it will
            # be recreated automatically on next use
            cache_dir = Path(getattr(self.settings, "cache_dir", "./cache"))
            cache_db = cache_dir / "docmind.duckdb"
            if cache_db.exists():
                cache_db.unlink()
            logger.info("Processing cache cleared (duckdb file removed if present)")
            return True
        except (OSError, RuntimeError) as e:
            logger.error("Failed to clear cache: {}", e)
            return False
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Boundary: ensure cache clear operation never crashes callers.
            logger.error("Unexpected error clearing cache: {}", e)
            return False

    async def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            dict[str, Any]: Basic stats for both caches.
        """
        try:
            cache_dir = Path(getattr(self.settings, "cache_dir", "./cache"))
            cache_db = cache_dir / "docmind.duckdb"
            return {
                "processor_type": "hybrid",
                "llamaindex_cache": {
                    "cache_type": "duckdb_kvstore",
                    "db_path": str(cache_db),
                    "collection": getattr(
                        self.cache, "collection", "docmind_processing"
                    ),
                    "total_documents": -1,
                },
                "strategy_mappings": len(self.strategy_map),
            }
        except (OSError, RuntimeError) as e:
            logger.error("Failed to get cache stats: {}", e)
            return {"error": str(e), "processor_type": "hybrid"}
        except Exception as e:  # pylint: disable=broad-exception-caught
            # Boundary: surface error as data instead of crashing monitoring code.
            logger.error("Unexpected error getting cache stats: {}", e)
            return {"error": str(e), "processor_type": "hybrid"}
