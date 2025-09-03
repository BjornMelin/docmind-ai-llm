"""Retrieval expert agent for multi-agent coordination system.

This module implements the RetrievalAgent that executes document retrieval
using multiple strategies and optimizations. The agent handles vector search,
hybrid search, GraphRAG, and DSPy query optimization for optimal results.

Features:
- Multi-strategy retrieval (vector/hybrid/graphrag)
- DSPy query optimization and rewriting
- GraphRAG for entity relationship queries
- Fallback mechanisms and error handling
- Performance monitoring under 150ms
- Integration with existing tool factory

Example:
    Using the retrieval agent::

        from agents.retrieval import RetrievalAgent

        retrieval_agent = RetrievalAgent(llm, tools_data)
        result = retrieval_agent.retrieve_documents(
            "machine learning algorithms",
            strategy="hybrid",
            use_dspy=True
        )
        # len(result.documents) contains the document count
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.tools import retrieve_documents

# Constants
RECURSION_LIMIT = 3
PERFORMANCE_TARGET_MS = 150  # milliseconds
BASE_CONFIDENCE = 0.5
CONFIDENCE_HIGH_DOC_COUNT = 0.3
CONFIDENCE_MEDIUM_DOC_COUNT = 0.2
CONFIDENCE_LOW_DOC_COUNT = 0.1
CONFIDENCE_STRATEGY_BONUS = 0.1
CONFIDENCE_QUALITY_BONUS = 0.05


class RetrievalResult(BaseModel):
    """Document retrieval result with metadata."""

    documents: list[dict[str, Any]] = Field(
        default_factory=list, description="Retrieved documents"
    )
    strategy_used: str = Field(description="Actual retrieval strategy used")
    query_original: str = Field(description="Original search query")
    query_optimized: str = Field(description="Optimized query after DSPy")
    document_count: int = Field(description="Number of documents retrieved")
    processing_time_ms: float = Field(description="Time taken for retrieval")
    dspy_used: bool = Field(
        default=False, description="Whether DSPy optimization was used"
    )
    graphrag_used: bool = Field(default=False, description="Whether GraphRAG was used")
    confidence_score: float = Field(
        default=0.0, description="Confidence in retrieval quality"
    )
    reasoning: str = Field(default="", description="Explanation of retrieval decisions")


class RetrievalAgent:
    """Specialized agent for document retrieval with multi-strategy support.

    Executes document retrieval using optimal strategies based on query
    characteristics and available tools. Supports vector search, hybrid search,
    and GraphRAG with DSPy optimization for improved query performance.

    Retrieval Strategies:
    - Vector search: Direct semantic similarity using dense embeddings
    - Hybrid search: Combined dense + sparse embeddings with fusion
    - GraphRAG: Entity relationship queries using knowledge graphs
    - Fallback: Automatic strategy degradation on failures
    """

    def __init__(self, llm: Any, tools_data: dict[str, Any]):
        """Initialize retrieval agent.

        Args:
            llm: Language model for retrieval decisions
            tools_data: Dictionary containing indexes and retrieval tools
        """
        self.llm = llm
        self.tools_data = tools_data
        self.total_retrievals = 0
        self.retrieval_times = []
        self.strategy_usage = {"vector": 0, "hybrid": 0, "graphrag": 0, "fallback": 0}

        # Create LangGraph agent
        self.agent = create_react_agent(
            model=self.llm,
            tools=[retrieve_documents],
        )

        logger.info("RetrievalAgent initialized")

    def retrieve_documents(
        self,
        query: str,
        strategy: str = "hybrid",
        use_dspy: bool = True,
        use_graphrag: bool = False,
        _context: ChatMemoryBuffer | None = None,
        **_kwargs: Any,
    ) -> RetrievalResult:
        """Execute document retrieval using specified strategy.

        Retrieves relevant documents using the specified strategy with
        optional optimizations. Automatically handles fallbacks and
        provides detailed metadata about the retrieval process.

        Args:
            query: Search query for document retrieval
            strategy: Retrieval strategy ("vector", "hybrid", "graphrag")
            use_dspy: Whether to use DSPy query optimization
            use_graphrag: Whether to use GraphRAG for relationships
            _context: Optional conversation context (not yet implemented)
            **_kwargs: Additional parameters for retrieval (not yet implemented)

        Returns:
            RetrievalResult with documents and comprehensive metadata

        Example:
            >>> result = agent.retrieve_documents("AI ethics", "hybrid")
            >>> result.document_count, result.strategy_used
        """
        start_time = time.perf_counter()
        self.total_retrievals += 1

        try:
            # Prepare agent input
            messages = [
                HumanMessage(
                    content=f"Retrieve documents for: {query} using {strategy} strategy"
                )
            ]

            # Include tools data and context in state
            # Note: state not used directly but kept for future context handling

            # Execute retrieval through agent
            result = self.agent.invoke(
                {"messages": messages},
                {
                    "recursion_limit": RECURSION_LIMIT
                },  # Limit iterations for performance
            )

            # Parse agent response
            retrieval_data = self._parse_agent_response(result, query, strategy)

            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            self.retrieval_times.append(processing_time)

            # Update strategy usage statistics
            strategy_used = retrieval_data.get("strategy_used", strategy)
            if strategy_used in self.strategy_usage:
                self.strategy_usage[strategy_used] += 1
            else:
                self.strategy_usage["fallback"] += 1

            # Build comprehensive result
            result_data = self._build_retrieval_result(
                retrieval_data,
                query,
                requested_strategy=strategy,
                use_dspy=use_dspy,
                use_graphrag=use_graphrag,
                processing_time=processing_time,
            )

            retrieval_result = RetrievalResult(**result_data)

            logger.info(
                "Retrieved %d documents via %s (%.1fms)",
                retrieval_result.document_count,
                retrieval_result.strategy_used,
                processing_time * 1000,
            )

            return retrieval_result

        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            logger.error("Document retrieval failed: %s", e)

            # Fallback retrieval
            processing_time = time.perf_counter() - start_time
            return self._fallback_retrieval(query, strategy, processing_time, str(e))

    def _parse_agent_response(
        self, result: dict, query: str, strategy: str
    ) -> dict[str, Any]:
        """Parse agent response to extract retrieval data."""
        try:
            # Get the final message from agent
            messages = result.get("messages", [])
            if not messages:
                raise ValueError("No messages in agent response")

            last_message = messages[-1]
            content = getattr(last_message, "content", str(last_message))

            # Try to parse JSON from agent response
            try:
                retrieval_data = json.loads(content)
                if isinstance(retrieval_data, dict) and "documents" in retrieval_data:
                    return retrieval_data
            except json.JSONDecodeError:
                pass

            # If JSON parsing failed, use fallback retrieval
            logger.warning("Could not parse JSON from agent, using fallback retrieval")
            return self._execute_fallback_retrieval(query, strategy)

        except (RuntimeError, ValueError, AttributeError) as e:
            logger.error("Failed to parse agent response: %s", e)
            return self._execute_fallback_retrieval(query, strategy)

    def _execute_fallback_retrieval(self, query: str, strategy: str) -> dict[str, Any]:
        """Execute fallback retrieval when agent fails."""
        try:
            # Use tool factory directly for fallback
            from src.agents.tool_factory import ToolFactory

            vector_index = self.tools_data.get("vector")
            kg_index = self.tools_data.get("kg")
            retriever = self.tools_data.get("retriever")

            if not vector_index:
                return {
                    "documents": [],
                    "error": "No vector index available",
                    "strategy_used": "none",
                    "query_original": query,
                    "query_optimized": query,
                }

            # Select fallback strategy
            if strategy == "graphrag" and kg_index:
                tool = ToolFactory.create_kg_search_tool(kg_index)
                strategy_used = "graphrag_fallback"
            elif strategy == "hybrid" and retriever:
                tool = ToolFactory.create_hybrid_search_tool(retriever)
                strategy_used = "hybrid_fallback"
            else:
                tool = ToolFactory.create_vector_search_tool(vector_index)
                strategy_used = "vector_fallback"

            # Execute search
            search_result = tool.call(query)

            # Parse tool result
            documents = self._parse_tool_result(search_result)

            return {
                "documents": documents,
                "strategy_used": strategy_used,
                "query_original": query,
                "query_optimized": query,
                "document_count": len(documents),
                "dspy_used": False,
                "graphrag_used": strategy == "graphrag",
            }

        except (OSError, RuntimeError, ValueError, AttributeError) as e:
            logger.error("Fallback retrieval also failed: %s", e)
            return {
                "documents": [],
                "error": str(e),
                "strategy_used": "failed",
                "query_original": query,
                "query_optimized": query,
            }

    def _parse_tool_result(self, result: Any) -> list[dict[str, Any]]:
        """Parse tool result to extract document list."""
        if isinstance(result, str):
            # Tool returned text response — create a minimal document entry
            return [
                {
                    "content": result,
                    "metadata": {"source": "tool_response"},
                    "score": 1.0,
                }
            ]
        if hasattr(result, "response"):
            # LlamaIndex response object
            documents = []
            if hasattr(result, "source_nodes"):
                for node in result.source_nodes:
                    documents.append(
                        {
                            "content": node.text,
                            "metadata": node.metadata,
                            "score": getattr(node, "score", 1.0),
                        }
                    )
            else:
                documents.append(
                    {
                        "content": result.response,
                        "metadata": {"source": "response"},
                        "score": 1.0,
                    }
                )
            return documents
        if isinstance(result, list):
            # List of documents
            documents = []
            for item in result:
                if hasattr(item, "text") and hasattr(item, "metadata"):
                    documents.append(
                        {
                            "content": item.text,
                            "metadata": item.metadata,
                            "score": getattr(item, "score", 1.0),
                        }
                    )
                elif isinstance(item, dict):
                    documents.append(item)
            return documents
        # Fallback - convert to string
        return [
            {
                "content": str(result),
                "metadata": {"source": "unknown"},
                "score": 1.0,
            }
        ]

    def _build_retrieval_result(
        self,
        retrieval_data: dict,
        original_query: str,
        *,
        requested_strategy: str,
        use_dspy: bool,
        use_graphrag: bool,
        processing_time: float,
    ) -> dict[str, Any]:
        """Build comprehensive retrieval result."""
        documents = retrieval_data.get("documents", [])
        strategy_used = retrieval_data.get("strategy_used", requested_strategy)

        # Calculate confidence score based on retrieval quality
        confidence = self._calculate_confidence_score(
            documents, strategy_used, retrieval_data
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            retrieval_data, requested_strategy, strategy_used, len(documents)
        )

        return {
            "documents": documents,
            "strategy_used": strategy_used,
            "query_original": original_query,
            "query_optimized": retrieval_data.get("query_optimized", original_query),
            "document_count": len(documents),
            "processing_time_ms": round(processing_time * 1000, 2),
            "dspy_used": use_dspy and retrieval_data.get("dspy_used", False),
            "graphrag_used": use_graphrag
            and retrieval_data.get("graphrag_used", False),
            "confidence_score": confidence,
            "reasoning": reasoning,
        }

    def _calculate_confidence_score(
        self, documents: list, strategy_used: str, retrieval_data: dict
    ) -> float:
        """Calculate confidence score for retrieval quality."""
        confidence = BASE_CONFIDENCE  # Base confidence

        # Document count factor
        doc_count = len(documents)
        if doc_count >= 5:
            confidence += CONFIDENCE_HIGH_DOC_COUNT
        elif doc_count >= 2:
            confidence += CONFIDENCE_MEDIUM_DOC_COUNT
        elif doc_count >= 1:
            confidence += CONFIDENCE_LOW_DOC_COUNT

        # Strategy factor
        if "fallback" not in strategy_used:
            confidence += CONFIDENCE_STRATEGY_BONUS
        if strategy_used in ["hybrid_fusion", "graphrag"]:
            confidence += CONFIDENCE_STRATEGY_BONUS

        # Quality indicators
        if retrieval_data.get("dspy_used"):
            confidence += CONFIDENCE_QUALITY_BONUS
        if "error" not in retrieval_data:
            confidence += CONFIDENCE_QUALITY_BONUS

        return min(confidence, 1.0)

    def _generate_reasoning(
        self, retrieval_data: dict, requested: str, used: str, doc_count: int
    ) -> str:
        """Generate human-readable reasoning for retrieval decisions."""
        reasons = []

        # Strategy reasoning
        if used == requested:
            reasons.append(f"Used requested {used} strategy")
        else:
            reasons.append(f"Fallback from {requested} to {used} strategy")

        # Results reasoning
        if doc_count > 0:
            reasons.append(f"Retrieved {doc_count} relevant documents")
        else:
            reasons.append("No documents found matching query")

        # Optimization reasoning
        if retrieval_data.get("dspy_used"):
            reasons.append("Query optimized with DSPy")
        if retrieval_data.get("graphrag_used"):
            reasons.append("Used GraphRAG for entity relationships")

        # Error reasoning
        if "error" in retrieval_data:
            reasons.append(f"Encountered issue: {retrieval_data['error']}")

        return "; ".join(reasons)

    def _fallback_retrieval(
        self, query: str, strategy: str, processing_time: float, error: str
    ) -> RetrievalResult:
        """Create fallback result when retrieval fails completely."""
        self.strategy_usage["fallback"] += 1

        return RetrievalResult(
            documents=[],
            strategy_used=f"{strategy}_failed",
            query_original=query,
            query_optimized=query,
            document_count=0,
            processing_time_ms=round(processing_time * 1000, 2),
            dspy_used=False,
            graphrag_used=False,
            confidence_score=0.0,
            reasoning=f"Retrieval failed: {error}",
        )

    def get_performance_stats(self) -> dict[str, Any]:
        """Get retrieval performance statistics.

        Returns:
            Dictionary with retrieval performance metrics
        """
        if not self.retrieval_times:
            return {
                "total_retrievals": self.total_retrievals,
                "avg_retrieval_time_ms": 0.0,
                "max_retrieval_time_ms": 0.0,
                "min_retrieval_time_ms": 0.0,
                "strategy_usage": self.strategy_usage,
            }

        avg_time = sum(self.retrieval_times) / len(self.retrieval_times)
        max_time = max(self.retrieval_times)
        min_time = min(self.retrieval_times)

        return {
            "total_retrievals": self.total_retrievals,
            "avg_retrieval_time_ms": round(avg_time * 1000, 2),
            "max_retrieval_time_ms": round(max_time * 1000, 2),
            "min_retrieval_time_ms": round(min_time * 1000, 2),
            "performance_target_met": avg_time
            < (PERFORMANCE_TARGET_MS / 1000),  # 150ms target
            "strategy_usage": self.strategy_usage,
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_retrievals = 0
        self.retrieval_times = []
        self.strategy_usage = {"vector": 0, "hybrid": 0, "graphrag": 0, "fallback": 0}
        logger.info("Retrieval performance stats reset")


# Strategy optimization utilities
def optimize_query_for_strategy(query: str, strategy: str) -> str:
    """Optimize query based on retrieval strategy.

    Args:
        query: Original query to optimize
        strategy: Target retrieval strategy

    Returns:
        Optimized query string
    """
    if strategy == "graphrag":
        # Add entity relationship terms
        if not any(
            word in query.lower() for word in ["relationship", "connect", "link"]
        ):
            return f"Find relationships and connections for: {query}"
    elif strategy == "hybrid":
        # Enhance for both semantic and keyword matching
        if len(query.split()) < 3:
            return f"Find comprehensive information about {query}"
    elif strategy == "vector" and not query.endswith("?"):
        # Optimize for semantic similarity
        return f"What is {query}?"

    return query


def select_optimal_strategy(query: str, available_tools: dict[str, Any]) -> str:
    """Select optimal retrieval strategy based on query and available tools.

    Args:
        query: Query to analyze
        available_tools: Dictionary of available retrieval tools

    Returns:
        Optimal strategy: "vector", "hybrid", or "graphrag"
    """
    query_lower = query.lower()

    # GraphRAG for relationship queries
    if any(
        word in query_lower for word in ["relationship", "connect", "link", "network"]
    ) and available_tools.get("kg"):
        return "graphrag"

    # Hybrid for complex queries
    if len(query.split()) > 10 or any(
        word in query_lower for word in ["compare", "analyze", "comprehensive"]
    ):
        return "hybrid"

    # Vector for simple queries
    return "vector"
