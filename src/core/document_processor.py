"""Document processing module for DocMind AI.

Handles document ingestion, parsing, chunking, and embedding generation.
"""

from pathlib import Path
from typing import Any

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger

from src.config.settings import Settings


class DocumentProcessor:
    """Processes documents for ingestion into the vector store."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the document processor.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap

        # Initialize text splitter
        self.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def process_document(self, file_path: Path) -> dict[str, Any]:
        """Process a single document synchronously.

        Args:
            file_path: Path to the document.

        Returns:
            Processing results metadata.
        """
        try:
            # Load document
            documents = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()

            # Parse into nodes
            nodes = self.text_splitter.get_nodes_from_documents(documents)

            logger.info(f"Processed document {file_path}: {len(nodes)} chunks created")

            return {
                "status": "success",
                "file": str(file_path),
                "chunks": len(nodes),
                "documents": len(documents),
            }

        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return {
                "status": "error",
                "file": str(file_path),
                "error": str(e),
            }

    async def aprocess_document(self, file_path: Path) -> dict[str, Any]:
        """Process a document asynchronously.

        Args:
            file_path: Path to the document.

        Returns:
            Processing results metadata.
        """
        # For now, delegate to sync version
        # In production, this would use async I/O
        return self.process_document(file_path)
