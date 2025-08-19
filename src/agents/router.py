"""Query routing agent for multi-agent coordination system.

This module implements the RouterAgent that analyzes incoming queries to
determine optimal processing strategies. The agent evaluates query complexity,
identifies intent patterns, and routes queries to appropriate retrieval
strategies for optimal results.

Features:
- Query complexity analysis (simple/medium/complex)
- Intent pattern recognition and classification
- Strategy selection (vector/hybrid/graphrag)
- Context dependency detection
- Confidence scoring for routing decisions
- Performance monitoring under 50ms

Example:
    Using the router agent::

        from agents.router import RouterAgent

        router = RouterAgent(llm)
        decision = router.route_query("Compare AI vs ML techniques")
        print(f"Strategy: {decision.strategy}")
        print(f"Complexity: {decision.complexity}")
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.tools import route_query


class RoutingDecision(BaseModel):
    """Query routing decision with metadata."""

    strategy: str = Field(description="Retrieval strategy (vector/hybrid/graphrag)")
    complexity: str = Field(description="Query complexity (simple/medium/complex)")
    needs_planning: bool = Field(description="Whether query needs decomposition")
    confidence: float = Field(description="Confidence in routing decision (0-1)")
    processing_time_ms: float = Field(description="Time taken for routing decision")
    context_dependent: bool = Field(
        default=False, description="Requires conversation context"
    )
    reasoning: str = Field(default="", description="Explanation of routing decision")


class RouterAgent:
    """Specialized agent for query analysis and routing decisions.

    Analyzes incoming queries to determine the optimal processing strategy
    based on complexity, intent, and context requirements. Routes queries
    to vector search, hybrid search, or GraphRAG based on characteristics.

    Decision Criteria:
    - Simple queries: Direct factual questions -> vector search
    - Medium queries: Multi-step questions -> hybrid search
    - Complex queries: Comparison/analysis -> hybrid + planning
    - Relationship queries: Entity connections -> GraphRAG
    """

    def __init__(self, llm: Any):
        """Initialize router agent.

        Args:
            llm: Language model for routing decisions
        """
        self.llm = llm
        self.total_routes = 0
        self.routing_times = []

        # Create LangGraph agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=[route_query],
        )

        logger.info("RouterAgent initialized")

    def route_query(
        self, query: str, context: ChatMemoryBuffer | None = None, **kwargs
    ) -> RoutingDecision:
        """Analyze query and determine optimal processing strategy.

        Evaluates query characteristics including complexity, intent patterns,
        and context dependencies to select the most appropriate retrieval
        strategy and processing approach.

        Args:
            query: User query to analyze and route
            context: Optional conversation context for dependency analysis
            **kwargs: Additional parameters for routing logic

        Returns:
            RoutingDecision with strategy, complexity, and metadata

        Example:
            >>> decision = router.route_query("What is machine learning?")
            >>> print(decision.strategy)  # "vector"
            >>> print(decision.complexity)  # "simple"
        """
        start_time = time.perf_counter()
        self.total_routes += 1

        try:
            # Prepare agent input
            messages = [HumanMessage(content=f"Route this query: {query}")]

            # Include context information if available
            # Note: state not used directly but kept for future context handling

            # Execute routing through agent
            result = self.agent.invoke(
                {"messages": messages},
                {"recursion_limit": 3},  # Limit iterations for performance
            )

            # Parse agent response
            decision_data = self._parse_agent_response(result, query)

            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            self.routing_times.append(processing_time)

            # Add timing and reasoning
            decision_data["processing_time_ms"] = round(processing_time * 1000, 2)
            decision_data["reasoning"] = self._generate_reasoning(decision_data, query)

            decision = RoutingDecision(**decision_data)

            logger.info(
                f"Query routed: {decision.complexity} -> {decision.strategy} "
                f"({processing_time * 1000:.1f}ms)"
            )

            return decision

        except Exception as e:
            logger.error(f"Query routing failed: {e}")

            # Fallback routing decision
            processing_time = time.perf_counter() - start_time
            return RoutingDecision(
                strategy="vector",  # Safe fallback
                complexity="simple",
                needs_planning=False,
                confidence=0.5,  # Low confidence for fallback
                processing_time_ms=round(processing_time * 1000, 2),
                reasoning=f"Fallback routing due to error: {str(e)}",
            )

    def _parse_agent_response(
        self, result: dict[str, Any], query: str
    ) -> dict[str, Any]:
        """Parse agent response to extract routing decision."""
        try:
            # Get the final message from agent
            messages = result.get("messages", [])
            if not messages:
                raise ValueError("No messages in agent response")

            last_message = messages[-1]
            content = getattr(last_message, "content", str(last_message))

            # Try to parse JSON from agent response
            try:
                decision_data = json.loads(content)
                if isinstance(decision_data, dict):
                    return decision_data
            except json.JSONDecodeError:
                pass

            # If JSON parsing failed, extract from text using fallback logic
            logger.warning("Could not parse JSON from agent, using fallback routing")
            return self._fallback_routing(query)

        except Exception as e:
            logger.error(f"Failed to parse agent response: {e}")
            return self._fallback_routing(query)

    def _fallback_routing(self, query: str) -> dict[str, Any]:
        """Fallback routing logic when agent fails."""
        query_lower = query.lower().strip()
        word_count = len(query.split())

        # Simple heuristic-based routing
        complexity = "simple"
        strategy = "vector"
        needs_planning = False
        confidence = 0.6  # Lower confidence for fallback

        # Check for complex query patterns
        complex_patterns = [
            "compare",
            "contrast",
            "vs",
            "versus",
            "analyze",
            "breakdown",
            "explain how",
            "relationship between",
        ]

        if (
            any(pattern in query_lower for pattern in complex_patterns)
            or word_count > 20
        ):
            complexity = "complex"
            strategy = "hybrid"
            needs_planning = True
            confidence = 0.7
        elif word_count > 10:
            complexity = "medium"
            strategy = "hybrid"
            confidence = 0.65

        # Check for GraphRAG indicators
        graph_patterns = ["connect", "relationship", "network", "link"]
        if any(pattern in query_lower for pattern in graph_patterns):
            strategy = "graphrag"
            confidence = 0.8

        return {
            "strategy": strategy,
            "complexity": complexity,
            "needs_planning": needs_planning,
            "confidence": confidence,
            "word_count": word_count,
            "context_dependent": any(
                indicator in query_lower
                for indicator in ["this", "that", "it", "they", "them"]
            ),
        }

    def _generate_reasoning(self, decision_data: dict, query: str) -> str:
        """Generate human-readable reasoning for routing decision."""
        strategy = decision_data.get("strategy", "vector")
        complexity = decision_data.get("complexity", "simple")
        word_count = decision_data.get("word_count", len(query.split()))

        reasons = []

        # Complexity reasoning
        if complexity == "complex":
            reasons.append(
                f"Complex query ({word_count} words) requiring decomposition"
            )
        elif complexity == "medium":
            reasons.append(f"Medium complexity query ({word_count} words)")
        else:
            reasons.append(f"Simple query ({word_count} words)")

        # Strategy reasoning
        if strategy == "graphrag":
            reasons.append("Contains relationship/connection terms → GraphRAG")
        elif strategy == "hybrid":
            reasons.append("Multi-faceted query → Hybrid search")
        else:
            reasons.append("Direct factual question → Vector search")

        # Planning reasoning
        if decision_data.get("needs_planning"):
            reasons.append("Requires sub-task decomposition")

        # Context reasoning
        if decision_data.get("context_dependent"):
            reasons.append("References conversation context")

        return "; ".join(reasons)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get routing performance statistics.

        Returns:
            Dictionary with routing performance metrics
        """
        if not self.routing_times:
            return {
                "total_routes": self.total_routes,
                "avg_routing_time_ms": 0.0,
                "max_routing_time_ms": 0.0,
                "min_routing_time_ms": 0.0,
            }

        avg_time = sum(self.routing_times) / len(self.routing_times)
        max_time = max(self.routing_times)
        min_time = min(self.routing_times)

        return {
            "total_routes": self.total_routes,
            "avg_routing_time_ms": round(avg_time * 1000, 2),
            "max_routing_time_ms": round(max_time * 1000, 2),
            "min_routing_time_ms": round(min_time * 1000, 2),
            "performance_target_met": avg_time < 0.05,  # 50ms target
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_routes = 0
        self.routing_times = []
        logger.info("Router performance stats reset")


# Factory function for backward compatibility
def create_router_agent(llm: Any) -> RouterAgent:
    """Create router agent instance.

    Args:
        llm: Language model for routing decisions

    Returns:
        Configured RouterAgent instance
    """
    return RouterAgent(llm)


# Query complexity analysis utilities
def analyze_query_complexity(query: str) -> str:
    """Analyze query complexity independently.

    Args:
        query: Query string to analyze

    Returns:
        Complexity level: "simple", "medium", or "complex"
    """
    query_lower = query.lower().strip()
    word_count = len(query.split())

    # Complex indicators
    complex_patterns = [
        "compare",
        "contrast",
        "vs",
        "versus",
        "analyze",
        "breakdown",
        "explain how",
        "step by step",
        "relationship between",
        "impact of",
    ]

    if any(pattern in query_lower for pattern in complex_patterns) or word_count > 20:
        return "complex"
    elif word_count > 10:
        return "medium"
    else:
        return "simple"


def detect_query_intent(query: str) -> str:
    """Detect primary intent of the query.

    Args:
        query: Query string to analyze

    Returns:
        Intent category: "factual", "comparison", "analysis", "list", or "procedural"
    """
    query_lower = query.lower().strip()

    # Intent patterns
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        return "comparison"
    elif any(word in query_lower for word in ["analyze", "analysis", "breakdown"]):
        return "analysis"
    elif any(word in query_lower for word in ["list", "enumerate", "examples"]):
        return "list"
    elif any(word in query_lower for word in ["how", "step", "process", "procedure"]):
        return "procedural"
    else:
        return "factual"
