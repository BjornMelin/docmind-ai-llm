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
from src.config.app_settings import DocMindSettings, app_settings
from src.processing.document_processor import DocumentProcessor

# Load environment variables
load_dotenv()

# Constants
AGENT_TIMEOUT_DIVISOR = 1000.0
BASIC_VALIDATION_SCORE = 0.8
DOCUMENT_TEXT_SLICE_SHORT = 500
DOCUMENT_TEXT_SLICE_LONG = 1000


class DocMindApplication:
    """Main application class for DocMind AI with multi-agent coordination."""

    def __init__(
        self,
        settings: DocMindSettings | None = None,
        enable_multi_agent: bool = True,
    ) -> None:
        """Initialize the DocMind application.

        Args:
            settings: Application settings. Defaults to loading from environment.
            enable_multi_agent: Enable multi-agent coordination system.
        """
        self.settings = settings or app_settings
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
                model_path=self.settings.model_name,
                max_context_length=self.settings.context_window_size,
                backend="vllm",
                enable_fallback=self.settings.enable_fallback_rag,
                max_agent_timeout=self.settings.agent_decision_timeout
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
                content=f"I encountered an error processing your query: {str(e)}",
                sources=[],
                metadata={"error": str(e), "fallback": True},
                validation_score=0.0,
                processing_time=0.0,
            )

    async def _process_basic_rag(
        self,
        query: str,
        _context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Process query through basic RAG pipeline.

        Args:
            query: User query.
            _context: Optional context (unused in basic implementation).

        Returns:
            Basic RAG response.
        """
        # Implementation of basic RAG pipeline using retrieval components
        # Note: This is a simplified implementation placeholder
        try:
            # Import would be done here, but keeping it safe for now
            # TODO: Properly integrate with retrieval system once API is stable
            logger.info("Basic RAG pipeline requested for query: %s", query)

            # Generate a basic informative response
            content = (
                f"Basic RAG pipeline received query: '{query}'. "
                "This functionality requires document indexing and retrieval setup. "
                "For full functionality, please use multi-agent mode which includes "
                "comprehensive document analysis and retrieval capabilities."
            )

        except (ImportError, ValueError, RuntimeError) as e:
            logger.warning("Basic RAG pipeline setup encountered issues: %s", e)
            content = (
                f"Basic RAG pipeline encountered an error for query: '{query}'. "
                "Please use multi-agent mode or check your configuration."
            )

        return AgentResponse(
            content=content,
            sources=[],  # No sources in placeholder implementation
            metadata={"pipeline": "basic_rag"},
            validation_score=BASIC_VALIDATION_SCORE,  # Basic confidence
            processing_time=0.0,  # Would be measured
        )

    def _generate_basic_response(
        self,
        _query: str,
        results: list[Any],
    ) -> str:
        """Generate basic response from retrieval results.

        Args:
            _query: Original query (unused in basic implementation).
            results: Retrieved documents.

        Returns:
            Generated response text.
        """
        if not results:
            return "I couldn't find relevant information to answer your query."

        # Format context from results
        context_text = "\n\n".join(
            [
                f"Source {i + 1}: {doc.text[:DOCUMENT_TEXT_SLICE_SHORT]}"
                for i, doc in enumerate(results[:3])
            ]
        )

        # Note: In production, this would use the actual LLM
        response = (
            f"Based on the retrieved information:\n\n"
            f"{context_text[:DOCUMENT_TEXT_SLICE_LONG]}..."
        )

        return response

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

        # Cleanup components
        if self.agent_coordinator:
            # Any cleanup needed for agents
            pass

        # Close connections handled by agent coordinator
        # No explicit cleanup needed for current implementation

        logger.info("Shutdown complete")


async def main() -> None:
    """Main entry point for the application."""
    # Initialize application
    app = DocMindApplication(enable_multi_agent=True)

    # Example usage
    query = "What are the key features of the new product?"
    response = await app.process_query(query)

    print(f"Response: {response.content}")
    print(f"Processing time: {response.processing_time:.2f}s")
    print(f"Validation score: {response.validation_score:.2f}")

    # Shutdown
    app.shutdown()


if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
