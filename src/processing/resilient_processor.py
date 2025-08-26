"""ResilientDocumentProcessor with direct Unstructured.io integration.

This module implements ADR-009 compliant document processing with direct
Unstructured.io integration using hi_res strategy, multimodal extraction,
and bulletproof error handling.

Key Features:
- Direct unstructured.partition.auto.partition() calls
- Strategy mapping (hi_res, fast, ocr_only) based on file type
- Multimodal extraction: tables, images, OCR text
- Complete metadata preservation from unstructured.io
- Async processing with Tenacity retry patterns
- Performance target: >1 page/second with hi_res strategy
"""

import asyncio
import hashlib
import time
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from unstructured.partition.auto import partition

from src.config.settings import app_settings
from src.processing.models import DocumentElement, ProcessingResult


class ProcessingStrategy(str, Enum):
    """Processing strategies for different document types."""

    HI_RES = "hi_res"
    FAST = "fast"
    OCR_ONLY = "ocr_only"


class ProcessingError(Exception):
    """Custom exception for document processing errors."""

    pass


class ResilientDocumentProcessor:
    """Direct Unstructured.io document processor with bulletproof error handling.

    This processor implements ADR-009 requirements for direct unstructured.io
    integration with no wrapper abstractions. It provides:

    - Direct partition() calls with strategy mapping
    - Multimodal extraction (tables, images, OCR)
    - Complete metadata preservation
    - Async processing with retry logic
    - Performance optimization for >1 page/second target
    """

    def __init__(self, settings: Any | None = None):
        """Initialize ResilientDocumentProcessor.

        Args:
            settings: DocMind configuration settings. Uses app_settings if None.
        """
        self.settings = settings or app_settings

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

        logger.info(
            "ResilientDocumentProcessor initialized with strategy mapping for "
            "{} file types",
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
        logger.debug(f"Selected strategy '{strategy}' for file: {file_path}")
        return strategy

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

    def _validate_config(self, config: dict[str, Any]) -> bool:
        """Validate processing configuration parameters.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        valid_strategies = {s.value for s in ProcessingStrategy}

        if "strategy" in config and config["strategy"] not in valid_strategies:
            raise ValueError(
                f"Invalid strategy '{config['strategy']}'. "
                f"Valid strategies: {valid_strategies}"
            )

        # Validate other parameters
        if "max_characters" in config and (
            not isinstance(config["max_characters"], int)
            or config["max_characters"] <= 0
        ):
            raise ValueError("max_characters must be a positive integer")

        if "combine_text_under_n_chars" in config and (
            not isinstance(config["combine_text_under_n_chars"], int)
            or config["combine_text_under_n_chars"] < 0
        ):
            raise ValueError(
                "combine_text_under_n_chars must be a non-negative integer"
            )

        return True

    def _convert_unstructured_element(self, element: Any) -> DocumentElement:
        """Convert unstructured.io element to DocumentElement.

        Args:
            element: Raw element from unstructured.partition

        Returns:
            DocumentElement with all metadata preserved
        """
        # Extract metadata dictionary
        metadata = {}
        if hasattr(element, "metadata") and element.metadata:
            metadata = {
                "page_number": getattr(element.metadata, "page_number", None),
                "element_id": getattr(element.metadata, "element_id", None),
                "parent_id": getattr(element.metadata, "parent_id", None),
                "filename": getattr(element.metadata, "filename", None),
                "coordinates": getattr(element.metadata, "coordinates", None),
                "text_as_html": getattr(element.metadata, "text_as_html", None),
                "image_path": getattr(element.metadata, "image_path", None),
            }

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

        return DocumentElement(
            text=str(element.text) if element.text else "",
            category=str(element.category)
            if hasattr(element, "category")
            else "Unknown",
            metadata=metadata,
        )

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
        """Process document asynchronously with direct unstructured.io integration.

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

        logger.info(f"Processing document: {file_path}")

        # Validate file exists and size
        if not file_path.exists():
            raise ProcessingError(f"File not found: {file_path}")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.settings.max_document_size_mb:
            raise ProcessingError(
                f"Document size ({file_size_mb:.1f}MB) exceeds limit "
                f"({self.settings.max_document_size_mb}MB)"
            )

        try:
            # Determine processing strategy
            strategy = self._get_strategy_for_file(file_path)

            # Build partition configuration
            partition_config = self._build_partition_config(strategy, config_override)

            # Validate configuration
            if config_override:
                self._validate_config(config_override)
                partition_config.update(config_override)

            # Process document in thread pool to avoid blocking
            elements = await asyncio.to_thread(
                self._partition_document_sync, str(file_path), partition_config
            )

            # Convert unstructured elements to DocumentElements
            processed_elements = [
                self._convert_unstructured_element(elem) for elem in elements
            ]

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
                    "configuration": partition_config,
                },
                document_hash=document_hash,
            )

            logger.info(
                f"Successfully processed {file_path}: {len(processed_elements)} "
                f"elements in {processing_time:.2f}s (strategy: {strategy})"
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

    def _build_partition_config(
        self,
        strategy: ProcessingStrategy,
        config_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build partition configuration for unstructured.io.

        Args:
            strategy: Processing strategy to use
            config_override: Optional configuration overrides

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

        # Apply overrides
        if config_override:
            config.update(config_override)

        return config

    def _partition_document_sync(
        self, file_path: str, config: dict[str, Any]
    ) -> list[Any]:
        """Synchronous document partitioning with direct unstructured.io call.

        Args:
            file_path: Path to the document file
            config: Partition configuration

        Returns:
            List of unstructured.io elements

        Raises:
            ProcessingError: If partitioning fails
        """
        try:
            logger.debug(f"Calling partition() with config: {config}")

            # Direct unstructured.partition.auto.partition() call
            elements = partition(filename=file_path, **config)

            logger.debug(f"Partition returned {len(elements)} elements")
            return elements

        except Exception as e:
            logger.error(f"Partition failed for {file_path}: {str(e)}")
            raise ProcessingError(f"Unstructured.io partition failed: {e}") from e

    def override_config(self, config: dict[str, Any]) -> None:
        """Override configuration for processing operations.

        Args:
            config: Configuration dictionary to apply

        Raises:
            ValueError: If configuration is invalid
        """
        self._validate_config(config)
        self._config_override = config
        logger.info(f"Configuration override applied: {config}")

    def _apply_config_override(self, base_config: dict[str, Any]) -> dict[str, Any]:
        """Apply configuration overrides to base configuration.

        Args:
            base_config: Base configuration dictionary

        Returns:
            Configuration with overrides applied
        """
        if not hasattr(self, "_config_override"):
            return base_config

        config = base_config.copy()
        config.update(self._config_override)
        return config


# Factory function for easy instantiation
def create_resilient_processor(
    settings: Any | None = None,
) -> ResilientDocumentProcessor:
    """Factory function to create ResilientDocumentProcessor instance.

    Args:
        settings: Optional DocMind settings. Uses app_settings if None.

    Returns:
        Configured ResilientDocumentProcessor instance
    """
    return ResilientDocumentProcessor(settings)
