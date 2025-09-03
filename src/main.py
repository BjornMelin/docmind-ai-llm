"""Main application entry point for DocMind AI with Multi-Agent Coordination.

This module provides the primary application interface with full multi-agent
orchestration, document processing, and retrieval capabilities.
"""

import asyncio
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import AgentResponse
from src.config.settings import DocMindSettings, settings
from src.processing.document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Constants
AGENT_TIMEOUT_DIVISOR = 1000.0
BASIC_VALIDATION_SCORE = 0.8


class DocMindApplication:
    """Main application class for DocMind AI with multi-agent coordination."""

    def __init__(
        self,
        app_settings: DocMindSettings | None = None,
        enable_multi_agent: bool = True,
    ) -> None:
        """Initialize the DocMind application.

        Args:
            app_settings: Application settings. Defaults to unified settings.
            enable_multi_agent: Enable multi-agent coordination system.
        """
        self.settings = app_settings or settings
        self.enable_multi_agent = enable_multi_agent

        # Initialize components
        self._initialize_components()

        logger.info("DocMind AI initialized (multi-agent: %s)", self.enable_multi_agent)

    def _initialize_components(self) -> None:
        """Initialize application components."""
        # Document processor for ingestion
        self.document_processor = DocumentProcessor(settings=self.settings)

        # Initialize retrieval components (handled through agents)
        # Retrieval is managed through the multi-agent coordinator

        # Multi-agent coordinator if enabled
        if self.enable_multi_agent:
            self.agent_coordinator = MultiAgentCoordinator(
                model_path=self.settings.vllm.model,
                max_context_length=self.settings.vllm.context_window,
                backend="vllm",
                enable_fallback=self.settings.agents.enable_fallback_rag,
                max_agent_timeout=self.settings.agents.decision_timeout
                / AGENT_TIMEOUT_DIVISOR,
            )
        else:
            self.agent_coordinator = None

    async def process_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        use_multi_agent: bool | None = None,
    ) -> AgentResponse:
        """Process a user query through the appropriate pipeline.

        Args:
            query: User's natural language query.
            context: Optional conversation context.
            use_multi_agent: Override for multi-agent usage.

        Returns:
            Agent response with answer and metadata.
        """
        try:
            # Determine whether to use multi-agent
            use_agents = (
                use_multi_agent
                if use_multi_agent is not None
                else self.enable_multi_agent
            )

            if use_agents and self.agent_coordinator:
                # Use multi-agent coordination
                logger.info("Processing query with multi-agent system")
                response = self.agent_coordinator.process_query(
                    query=query,
                    context=context,
                )
            else:
                # Use basic RAG pipeline
                logger.info("Processing query with basic RAG pipeline")
                response = await self._process_basic_rag(query, context)

            return response

        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.error("Error processing query: %s", e)
            # Return error response
            return AgentResponse(
                content=f"I encountered an error processing your query: {e!s}",
                sources=[],
                metadata={"error": str(e), "fallback": True},
                validation_score=0.0,
                processing_time=0.0,
            )

    async def _process_basic_rag(
        self,
        _query: str,
        _context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Process query through basic RAG pipeline.

        Args:
            query: User query.
            _context: Optional context (unused in basic implementation).

        Returns:
            Basic RAG response.
        """
        # Basic RAG is deprecated - multi-agent mode should be used instead
        logger.warning(
            "Basic RAG pipeline is deprecated. Use multi-agent mode "
            "for full functionality."
        )
        content = (
            "Basic RAG pipeline is no longer supported. "
            "Please enable multi-agent mode for document analysis "
            "and retrieval capabilities."
        )

        return AgentResponse(
            content=content,
            sources=[],  # No sources in placeholder implementation
            metadata={"pipeline": "basic_rag"},
            validation_score=BASIC_VALIDATION_SCORE,  # Basic confidence
            processing_time=0.0,  # Would be measured
        )

    async def ingest_document(
        self,
        file_path: Path,
        process_async: bool = True,
    ) -> dict[str, Any]:
        """Ingest a document into the system.

        Args:
            file_path: Path to document file.
            process_async: Process asynchronously.

        Returns:
            Ingestion results metadata.
        """
        try:
            logger.info("Ingesting document: %s", file_path)

            if process_async:
                result = await self.document_processor.process_document_async(file_path)
            else:
                result = asyncio.run(
                    self.document_processor.process_document_async(file_path)
                )

            logger.info("Document ingested successfully: %s", file_path)
            return result

        except (ValueError, TypeError, OSError, FileNotFoundError) as e:
            logger.error("Error ingesting document %s: %s", file_path, e)
            return {"error": str(e), "status": "failed"}

    def shutdown(self) -> None:
        """Shutdown application and cleanup resources."""
        logger.info("Shutting down DocMind AI application")

        # Cleanup agent coordinator if present
        if self.agent_coordinator:
            # Agent coordinator handles its own cleanup
            pass

        logger.info("Shutdown complete")


async def main() -> None:
    """Main entry point for the application."""
    # Initialize application
    app = DocMindApplication(enable_multi_agent=True)

    # Example usage
    query = "What are the key features of the new product?"
    response = await app.process_query(query)

    logger.info("Response: %s", response.content)
    logger.info("Processing time: %.2fs", response.processing_time)
    logger.info("Validation score: %.2f", response.validation_score)

    # Shutdown
    app.shutdown()


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
