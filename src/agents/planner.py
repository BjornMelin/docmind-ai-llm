"""Query planning agent for multi-agent coordination system.

This module implements the PlannerAgent that decomposes complex queries into
manageable sub-tasks. The agent analyzes query structure, identifies key
components, and creates execution plans for systematic processing.

Features:
- Query decomposition and sub-task generation
- Execution order planning (parallel/sequential)
- Complexity estimation for sub-tasks
- Pattern-based decomposition strategies
- Context-aware planning with conversation history
- Performance monitoring under 100ms

Example:
    Using the planner agent::

        from agents.planner import PlannerAgent

        planner = PlannerAgent(llm)
        plan = planner.plan_query(
            "Compare AI vs ML performance metrics",
            complexity="complex"
        )
        print(f"Sub-tasks: {plan.sub_tasks}")
        print(f"Execution: {plan.execution_order}")
"""

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from llama_index.core.memory import ChatMemoryBuffer
from loguru import logger
from pydantic import BaseModel, Field

from src.agents.tools import plan_query


class QueryPlan(BaseModel):
    """Query execution plan with sub-tasks and metadata."""

    original_query: str = Field(description="Original user query")
    sub_tasks: list[str] = Field(description="List of sub-tasks to execute")
    execution_order: str = Field(description="Execution order (parallel/sequential)")
    estimated_complexity: str = Field(
        description="Estimated complexity (low/medium/high)"
    )
    processing_time_ms: float = Field(description="Time taken for planning")
    task_count: int = Field(description="Number of sub-tasks generated")
    decomposition_strategy: str = Field(
        default="", description="Strategy used for decomposition"
    )
    reasoning: str = Field(default="", description="Explanation of planning decisions")


class PlannerAgent:
    """Specialized agent for query planning and decomposition.

    Breaks down complex or multi-part queries into manageable sub-tasks that
    can be processed independently or sequentially. Uses pattern recognition
    to identify optimal decomposition strategies for different query types.

    Decomposition Strategies:
    - Comparison queries: Extract entities and create comparison framework
    - Analysis queries: Identify components and research phases
    - Process queries: Break into sequential steps
    - List queries: Structure enumeration and categorization
    - Multi-part queries: Split on connectors and synthesize
    """

    def __init__(self, llm: Any):
        """Initialize planner agent.

        Args:
            llm: Language model for planning decisions
        """
        self.llm = llm
        self.total_plans = 0
        self.planning_times = []

        # Create LangGraph agent
        self.agent = create_react_agent(
            llm=self.llm,
            tools=[plan_query],
        )

        logger.info("PlannerAgent initialized")

    def plan_query(
        self,
        query: str,
        complexity: str = "medium",
        context: ChatMemoryBuffer | None = None,
        **kwargs,
    ) -> QueryPlan:
        """Decompose query into structured sub-tasks.

        Analyzes query structure and creates an execution plan with sub-tasks
        that can be processed independently or sequentially based on the
        query type and complexity.

        Args:
            query: Original user query to decompose
            complexity: Query complexity level (simple/medium/complex)
            context: Optional conversation context for planning
            **kwargs: Additional parameters for planning logic

        Returns:
            QueryPlan with sub-tasks, execution order, and metadata

        Example:
            >>> plan = planner.plan_query("Compare AI vs ML", "complex")
            >>> print(plan.sub_tasks)
            >>> # ["Define AI", "Define ML", "Compare performance"]
        """
        start_time = time.perf_counter()
        self.total_plans += 1

        try:
            # Skip planning for simple queries
            if complexity == "simple":
                processing_time = time.perf_counter() - start_time
                return QueryPlan(
                    original_query=query,
                    sub_tasks=[query],
                    execution_order="sequential",
                    estimated_complexity="low",
                    processing_time_ms=round(processing_time * 1000, 2),
                    task_count=1,
                    decomposition_strategy="simple_passthrough",
                    reasoning="Simple query requires no decomposition",
                )

            # Prepare agent input
            messages = [HumanMessage(content=f"Plan this {complexity} query: {query}")]

            # Include context information if available
            # Note: state not used directly but kept for future context handling

            # Execute planning through agent
            result = self.agent.invoke(
                {"messages": messages},
                {"recursion_limit": 3},  # Limit iterations for performance
            )

            # Parse agent response
            plan_data = self._parse_agent_response(result, query, complexity)

            # Calculate processing time
            processing_time = time.perf_counter() - start_time
            self.planning_times.append(processing_time)

            # Add timing and reasoning
            plan_data["processing_time_ms"] = round(processing_time * 1000, 2)
            plan_data["reasoning"] = self._generate_reasoning(plan_data, query)

            plan = QueryPlan(**plan_data)

            logger.info(
                f"Query planned: {len(plan.sub_tasks)} tasks, {plan.execution_order} "
                f"({processing_time * 1000:.1f}ms)"
            )

            return plan

        except Exception as e:
            logger.error(f"Query planning failed: {e}")

            # Fallback planning
            processing_time = time.perf_counter() - start_time
            return self._fallback_planning(query, complexity, processing_time)

    def _parse_agent_response(
        self, result: dict, query: str, complexity: str
    ) -> dict[str, Any]:
        """Parse agent response to extract planning data."""
        try:
            # Get the final message from agent
            messages = result.get("messages", [])
            if not messages:
                raise ValueError("No messages in agent response")

            last_message = messages[-1]
            content = getattr(last_message, "content", str(last_message))

            # Try to parse JSON from agent response
            try:
                plan_data = json.loads(content)
                if isinstance(plan_data, dict) and "sub_tasks" in plan_data:
                    return plan_data
            except json.JSONDecodeError:
                pass

            # If JSON parsing failed, use fallback planning
            logger.warning("Could not parse JSON from agent, using fallback planning")
            return self._generate_fallback_plan(query, complexity)

        except Exception as e:
            logger.error(f"Failed to parse agent response: {e}")
            return self._generate_fallback_plan(query, complexity)

    def _generate_fallback_plan(self, query: str, complexity: str) -> dict[str, Any]:
        """Generate fallback plan when agent fails."""
        query_lower = query.lower()
        sub_tasks = []
        execution_order = "sequential"
        strategy = "default"

        # Comparison queries
        if any(
            word in query_lower for word in ["compare", "vs", "versus", "difference"]
        ):
            strategy = "comparison"
            # Extract entities to compare
            parts = query.replace(" vs ", " versus ").split(" versus ")
            if len(parts) == 1:
                parts = query.replace(" and ", " | ").split(" | ")

            if len(parts) >= 2:
                entity1, entity2 = parts[0].strip(), parts[1].strip()
                sub_tasks = [
                    f"Find information about {entity1}",
                    f"Find information about {entity2}",
                    f"Compare {entity1} and {entity2}",
                    f"Summarize key differences between {entity1} and {entity2}",
                ]
                execution_order = "parallel"
            else:
                # Fallback for unclear comparison
                sub_tasks = [
                    f"Identify key concepts in: {query}",
                    "Research each concept separately",
                    "Compare and contrast the concepts",
                ]

        # Analysis queries
        elif any(word in query_lower for word in ["analyze", "analysis", "breakdown"]):
            strategy = "analysis"
            sub_tasks = [
                f"Identify key components of: {query}",
                "Research background information",
                "Analyze relationships and patterns",
                "Synthesize findings and insights",
            ]

        # Process/explanation queries
        elif any(word in query_lower for word in ["how", "process", "step", "explain"]):
            strategy = "process"
            sub_tasks = [
                "Find definition and overview of the topic",
                "Identify key steps or components",
                "Research detailed explanations",
                "Organize information in logical sequence",
            ]

        # List/enumeration queries
        elif any(
            word in query_lower for word in ["list", "enumerate", "examples", "types"]
        ):
            strategy = "enumeration"
            sub_tasks = [
                f"Find comprehensive information about: {query}",
                "Extract and categorize relevant items",
                "Organize findings into structured list",
            ]

        # Default decomposition for complex queries
        else:
            strategy = "multi_part"
            # Split on common connectors
            connectors = [" and ", " or ", " also ", " additionally ", " furthermore "]
            parts = [query]

            for connector in connectors:
                new_parts = []
                for part in parts:
                    new_parts.extend(part.split(connector))
                parts = new_parts

            if len(parts) > 1:
                sub_tasks = [part.strip() for part in parts if part.strip()]
                sub_tasks.append("Synthesize information from all parts")
            else:
                # Single complex query - break into research phases
                sub_tasks = [
                    f"Research background information for: {query}",
                    "Find detailed analysis and examples",
                    "Synthesize comprehensive response",
                ]

        # Ensure we have at least one sub-task
        if not sub_tasks:
            sub_tasks = [query]
            strategy = "passthrough"

        return {
            "original_query": query,
            "sub_tasks": sub_tasks,
            "execution_order": execution_order,
            "estimated_complexity": "high" if len(sub_tasks) > 3 else "medium",
            "task_count": len(sub_tasks),
            "decomposition_strategy": strategy,
        }

    def _fallback_planning(
        self, query: str, complexity: str, processing_time: float
    ) -> QueryPlan:
        """Create fallback plan when planning fails."""
        return QueryPlan(
            original_query=query,
            sub_tasks=[query],  # Single task fallback
            execution_order="sequential",
            estimated_complexity="medium",
            processing_time_ms=round(processing_time * 1000, 2),
            task_count=1,
            decomposition_strategy="error_fallback",
            reasoning="Planning failed, using single-task fallback",
        )

    def _generate_reasoning(self, plan_data: dict, query: str) -> str:
        """Generate human-readable reasoning for planning decisions."""
        strategy = plan_data.get("decomposition_strategy", "default")
        task_count = plan_data.get("task_count", 1)
        execution_order = plan_data.get("execution_order", "sequential")

        reasons = []

        # Strategy reasoning
        strategy_explanations = {
            "comparison": (
                "Comparison query → Extract entities and create comparison framework"
            ),
            "analysis": "Analysis query → Break into research and synthesis phases",
            "process": "Process query → Identify steps and sequential organization",
            "enumeration": "List query → Structure enumeration and categorization",
            "multi_part": "Multi-part query → Split on connectors and synthesize",
            "simple_passthrough": "Simple query → No decomposition needed",
            "error_fallback": "Planning error → Single task fallback",
        }

        if strategy in strategy_explanations:
            reasons.append(strategy_explanations[strategy])

        # Task count reasoning
        if task_count > 1:
            reasons.append(
                f"Generated {task_count} sub-tasks for systematic processing"
            )

        # Execution order reasoning
        if execution_order == "parallel":
            reasons.append("Independent tasks can be executed in parallel")
        else:
            reasons.append("Sequential execution for dependent tasks")

        return "; ".join(reasons)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get planning performance statistics.

        Returns:
            Dictionary with planning performance metrics
        """
        if not self.planning_times:
            return {
                "total_plans": self.total_plans,
                "avg_planning_time_ms": 0.0,
                "max_planning_time_ms": 0.0,
                "min_planning_time_ms": 0.0,
            }

        avg_time = sum(self.planning_times) / len(self.planning_times)
        max_time = max(self.planning_times)
        min_time = min(self.planning_times)

        return {
            "total_plans": self.total_plans,
            "avg_planning_time_ms": round(avg_time * 1000, 2),
            "max_planning_time_ms": round(max_time * 1000, 2),
            "min_planning_time_ms": round(min_time * 1000, 2),
            "performance_target_met": avg_time < 0.1,  # 100ms target
        }

    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.total_plans = 0
        self.planning_times = []
        logger.info("Planner performance stats reset")


# Factory function for backward compatibility
def create_planner_agent(llm: Any) -> PlannerAgent:
    """Create planner agent instance.

    Args:
        llm: Language model for planning decisions

    Returns:
        Configured PlannerAgent instance
    """
    return PlannerAgent(llm)


# Query decomposition utilities
def decompose_comparison_query(query: str) -> list[str]:
    """Decompose comparison queries into entity extraction and analysis.

    Args:
        query: Comparison query to decompose

    Returns:
        List of sub-tasks for comparison analysis
    """
    # Extract entities to compare
    parts = query.replace(" vs ", " versus ").split(" versus ")
    if len(parts) == 1:
        parts = query.replace(" and ", " | ").split(" | ")

    if len(parts) >= 2:
        entity1, entity2 = parts[0].strip(), parts[1].strip()
        return [
            f"Research {entity1}",
            f"Research {entity2}",
            f"Compare {entity1} and {entity2}",
            "Summarize key differences",
        ]
    else:
        return [
            "Identify concepts to compare",
            "Research each concept",
            "Perform comparison analysis",
        ]


def detect_decomposition_strategy(query: str) -> str:
    """Detect optimal decomposition strategy for query.

    Args:
        query: Query to analyze

    Returns:
        Decomposition strategy: "comparison", "analysis", "process",
        "enumeration", or "multi_part"
    """
    query_lower = query.lower()

    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        return "comparison"
    elif any(word in query_lower for word in ["analyze", "analysis", "breakdown"]):
        return "analysis"
    elif any(word in query_lower for word in ["how", "process", "step", "explain"]):
        return "process"
    elif any(word in query_lower for word in ["list", "enumerate", "examples"]):
        return "enumeration"
    else:
        return "multi_part"
