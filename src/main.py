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
from src.config.settings import Settings
from src.core.document_processor import DocumentProcessor
from src.core.retrieval_engine import RetrievalEngine
from src.models.schemas import AgentResponse

# Load environment variables
load_dotenv()


class DocMindApplication:
    """Main application class for DocMind AI with multi-agent coordination."""

    def __init__(
        self,
        settings: Settings | None = None,
        enable_multi_agent: bool = True,
    ) -> None:
        """Initialize the DocMind application.

        Args:
            settings: Application settings. Defaults to loading from environment.
            enable_multi_agent: Enable multi-agent coordination system.
        """
        self.settings = settings or Settings()
        self.enable_multi_agent = enable_multi_agent

        # Initialize components
        self._initialize_components()

        logger.info(f"DocMind AI initialized (multi-agent: {self.enable_multi_agent})")

    def _initialize_components(self) -> None:
        """Initialize application components."""
        # Document processor for ingestion
        self.document_processor = DocumentProcessor(settings=self.settings)

        # Retrieval engine for search
        self.retrieval_engine = RetrievalEngine(settings=self.settings)

        # Multi-agent coordinator if enabled
        if self.enable_multi_agent:
            self.agent_coordinator = MultiAgentCoordinator(
                llm_backend=self.settings.llm_backend,
                model_name=self.settings.model_name,
                base_url=self.settings.llm_base_url,
                enable_fallback=self.settings.enable_fallback_rag,
                agent_timeout=self.settings.agent_decision_timeout / 1000.0,
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
                response = await self.agent_coordinator.aprocess_query(
                    query=query,
                    context=context,
                )
            else:
                # Use basic RAG pipeline
                logger.info("Processing query with basic RAG pipeline")
                response = await self._process_basic_rag(query, context)

            return response

        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"Error processing query: {e}")
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
        # Execute retrieval
        results = await self.retrieval_engine.asearch(
            query=query,
            top_k=self.settings.top_k,
            use_reranking=self.settings.use_reranking,
        )

        # Generate response
        # Note: This would integrate with existing LLM infrastructure
        content = self._generate_basic_response(query, results)

        return AgentResponse(
            content=content,
            sources=results[:3],  # Top 3 sources
            metadata={"pipeline": "basic_rag"},
            validation_score=0.8,  # Basic confidence
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
            [f"Source {i + 1}: {doc.text[:500]}" for i, doc in enumerate(results[:3])]
        )

        # Note: In production, this would use the actual LLM
        response = f"Based on the retrieved information:\n\n{context_text[:1000]}..."

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
            logger.info(f"Ingesting document: {file_path}")

            if process_async:
                result = await self.document_processor.aprocess_document(file_path)
            else:
                result = self.document_processor.process_document(file_path)

            logger.info(f"Document ingested successfully: {file_path}")
            return result

        except (ValueError, TypeError, OSError, FileNotFoundError) as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {"error": str(e), "status": "failed"}

    def shutdown(self) -> None:
        """Shutdown application and cleanup resources."""
        logger.info("Shutting down DocMind AI application")

        # Cleanup components
        if self.agent_coordinator:
            # Any cleanup needed for agents
            pass

        # Close vector store connections
        if hasattr(self.retrieval_engine, "close"):
            self.retrieval_engine.close()

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
