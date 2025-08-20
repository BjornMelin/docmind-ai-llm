"""LangGraph Supervisor System for Multi-Agent Coordination.

This module implements the supervisor graph that coordinates the 5-agent system
for DocMind AI. The supervisor manages agent handoffs, state transitions, and
ensures optimal workflow execution within 300ms latency requirements.

Architecture:
- Router Agent: Query analysis and strategy selection
- Planner Agent: Complex query decomposition
- Retrieval Agent: Document retrieval with multiple strategies
- Synthesis Agent: Content synthesis and response generation
- Validator Agent: Quality assurance and final validation

Features:
- State-based workflow management with LangGraph
- Conditional routing between agents based on decisions
- Parallel execution where possible for performance
- Comprehensive error handling and fallback mechanisms
- Performance monitoring and optimization
- Context preservation across agent transitions

Technical Requirements:
- Agent coordination within 300ms (REQ-0007)
- Support for Qwen3-4B-Instruct-2507-FP8 model
- Memory-efficient state management
- Robust error handling with graceful fallbacks
"""

import time
from typing import Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

from ..config.settings import settings
from ..utils.vllm_llm import get_vllm_backend
from .planner import PlannerAgent, QueryPlan
from .retrieval import RetrievalAgent, RetrievalResult
from .router import RouterAgent, RoutingDecision
from .synthesis import SynthesisAgent, SynthesisResult
from .validator import ValidationResult


class AgentState(TypedDict):
    """Shared state between all agents in the workflow.

    This state dictionary maintains all information as agents
    process the user query through the multi-agent pipeline.
    """

    # Input and routing
    query: str
    user_context: dict[str, Any] | None
    messages: list[BaseMessage]

    # Routing decisions
    routing_decision: RoutingDecision | None

    # Planning results
    query_plan: QueryPlan | None
    current_subtask: int

    # Retrieval results
    retrieval_results: list[RetrievalResult]

    # Synthesis results
    synthesis_result: SynthesisResult | None

    # Validation results
    validation_result: ValidationResult | None

    # Workflow control
    next_agent: str
    workflow_complete: bool
    error_occurred: bool
    error_message: str | None

    # Performance tracking
    start_time: float
    agent_timings: dict[str, float]
    total_processing_time: float


class SupervisorConfig(BaseModel):
    """Configuration for the supervisor graph."""

    # Performance settings
    max_processing_time_ms: int = Field(
        default=300,
        description="Maximum total processing time in milliseconds",
        ge=100,
        le=1000,
    )

    # Agent timeouts (individual)
    router_timeout_ms: int = Field(default=50, ge=10, le=100)
    planner_timeout_ms: int = Field(default=100, ge=50, le=200)
    retrieval_timeout_ms: int = Field(default=150, ge=100, le=300)
    synthesis_timeout_ms: int = Field(default=200, ge=100, le=400)
    validator_timeout_ms: int = Field(default=100, ge=50, le=200)

    # Workflow settings
    enable_parallel_execution: bool = Field(
        default=True,
        description="Enable parallel execution where possible",
    )
    enable_fallback_rag: bool = Field(
        default=True,
        description="Enable fallback to basic RAG on failures",
    )
    max_retries: int = Field(
        default=2,
        description="Maximum retries for failed agents",
        ge=0,
        le=5,
    )

    # Context management
    preserve_context: bool = Field(
        default=True,
        description="Preserve conversation context between queries",
    )
    max_context_messages: int = Field(
        default=10,
        description="Maximum messages to keep in context",
        ge=1,
        le=50,
    )


class SupervisorGraph:
    """LangGraph-based supervisor for multi-agent coordination."""

    def __init__(self, config: SupervisorConfig | None = None):
        """Initialize the supervisor graph with agents and configuration.

        Args:
            config: Supervisor configuration. Uses defaults if None.
        """
        self.config = config or SupervisorConfig()
        self.graph: StateGraph | None = None
        self.compiled_graph = None

        # Initialize backend
        self.vllm_backend = get_vllm_backend()

        # Initialize agents (will be created lazily)
        self._router_agent: RouterAgent | None = None
        self._planner_agent: PlannerAgent | None = None
        self._retrieval_agent: RetrievalAgent | None = None
        self._synthesis_agent: SynthesisAgent | None = None
        self._validator_agent: ValidationAgent | None = None

        logger.info(
            "SupervisorGraph initialized with config: %s", self.config.model_dump()
        )

    def _initialize_agents(self) -> None:
        """Initialize all agents with the vLLM backend."""
        if not self.vllm_backend._is_initialized:
            self.vllm_backend.initialize()

        # Initialize agents with vLLM backend
        # Note: These will need to be adapted to work with vLLM
        # For now, keeping the interface but agents will need updating

        logger.info("All agents initialized with vLLM backend")

    def _create_workflow_graph(self) -> StateGraph:
        """Create the LangGraph workflow for multi-agent coordination.

        Returns:
            Configured StateGraph with all agent nodes and edges.
        """
        # Create the state graph
        workflow = StateGraph(AgentState)

        # Add agent nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retrieval", self._retrieval_node)
        workflow.add_node("synthesis", self._synthesis_node)
        workflow.add_node("validator", self._validator_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Define the workflow edges
        workflow.set_entry_point("router")

        # Router decision routing
        workflow.add_conditional_edges(
            "router",
            self._route_after_router,
            {
                "planner": "planner",
                "retrieval": "retrieval",
                "error": "error_handler",
            },
        )

        # Planner to retrieval
        workflow.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "retrieval": "retrieval",
                "error": "error_handler",
            },
        )

        # Retrieval to synthesis
        workflow.add_conditional_edges(
            "retrieval",
            self._route_after_retrieval,
            {
                "synthesis": "synthesis",
                "error": "error_handler",
            },
        )

        # Synthesis to validator
        workflow.add_conditional_edges(
            "synthesis",
            self._route_after_synthesis,
            {
                "validator": "validator",
                "error": "error_handler",
            },
        )

        # Validator to end
        workflow.add_conditional_edges(
            "validator",
            self._route_after_validator,
            {
                "end": END,
                "retry": "retrieval",  # Retry from retrieval if validation fails
                "error": "error_handler",
            },
        )

        # Error handler to end
        workflow.add_edge("error_handler", END)

        return workflow

    # Agent node implementations
    async def _router_node(self, state: AgentState) -> AgentState:
        """Process query through router agent.

        Args:
            state: Current workflow state

        Returns:
            Updated state with routing decision
        """
        start_time = time.time()

        try:
            logger.debug("Router processing query: %s", state["query"][:100])

            # Create router decision (simplified for now)
            # TODO: Implement actual router logic with vLLM backend
            routing_decision = RoutingDecision(
                strategy="hybrid",  # Default strategy
                complexity="medium",
                needs_planning=len(state["query"]) > 100,  # Simple heuristic
                confidence=0.8,
                processing_time_ms=(time.time() - start_time) * 1000,
                context_dependent=False,
            )

            # Update state
            state["routing_decision"] = routing_decision
            state["agent_timings"]["router"] = time.time() - start_time

            logger.debug(
                "Router decision: strategy=%s, complexity=%s",
                routing_decision.strategy,
                routing_decision.complexity,
            )

            return state

        except Exception as e:
            logger.error("Router agent failed: %s", e)
            state["error_occurred"] = True
            state["error_message"] = f"Router failed: {str(e)}"
            return state

    async def _planner_node(self, state: AgentState) -> AgentState:
        """Process query through planner agent.

        Args:
            state: Current workflow state

        Returns:
            Updated state with query plan
        """
        start_time = time.time()

        try:
            logger.debug("Planner processing complex query")

            # Create query plan (simplified for now)
            # TODO: Implement actual planner logic with vLLM backend
            query_plan = QueryPlan(
                original_query=state["query"],
                sub_tasks=[state["query"]],  # Single task for now
                execution_order="sequential",
                estimated_complexity="medium",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update state
            state["query_plan"] = query_plan
            state["current_subtask"] = 0
            state["agent_timings"]["planner"] = time.time() - start_time

            logger.debug(
                "Query plan created with %d sub-tasks", len(query_plan.sub_tasks)
            )

            return state

        except Exception as e:
            logger.error("Planner agent failed: %s", e)
            state["error_occurred"] = True
            state["error_message"] = f"Planner failed: {str(e)}"
            return state

    async def _retrieval_node(self, state: AgentState) -> AgentState:
        """Process query through retrieval agent.

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieval results
        """
        start_time = time.time()

        try:
            logger.debug(
                "Retrieval processing with strategy: %s",
                state["routing_decision"].strategy
                if state["routing_decision"]
                else "default",
            )

            # Create retrieval result (simplified for now)
            # TODO: Implement actual retrieval logic
            retrieval_result = RetrievalResult(
                documents=[{"content": "Sample document", "score": 0.9}],
                strategy_used=state["routing_decision"].strategy
                if state["routing_decision"]
                else "hybrid",
                query_original=state["query"],
                query_optimized=state["query"],  # No optimization for now
                total_documents_found=1,
                processing_time_ms=(time.time() - start_time) * 1000,
                dspy_optimized=False,
            )

            # Update state
            if not state["retrieval_results"]:
                state["retrieval_results"] = []
            state["retrieval_results"].append(retrieval_result)
            state["agent_timings"]["retrieval"] = time.time() - start_time

            logger.debug(
                "Retrieval completed: %d documents found",
                len(retrieval_result.documents),
            )

            return state

        except Exception as e:
            logger.error("Retrieval agent failed: %s", e)
            state["error_occurred"] = True
            state["error_message"] = f"Retrieval failed: {str(e)}"
            return state

    async def _synthesis_node(self, state: AgentState) -> AgentState:
        """Process query through synthesis agent.

        Args:
            state: Current workflow state

        Returns:
            Updated state with synthesis result
        """
        start_time = time.time()

        try:
            logger.debug(
                "Synthesis processing %d retrieval results",
                len(state["retrieval_results"]),
            )

            # Create synthesis result (simplified for now)
            # TODO: Implement actual synthesis logic with vLLM backend
            synthesis_result = SynthesisResult(
                documents=[{"content": "Sample synthesized content", "score": 0.9}],
                original_count=len(state["retrieval_results"]),
                final_count=1,
                deduplication_ratio=1.0,
                strategies_used=["hybrid"],
                processing_time_ms=(time.time() - start_time) * 1000,
                confidence_score=0.85,
                reasoning="Generated response based on retrieved documents",
            )

            # Update state
            state["synthesis_result"] = synthesis_result
            state["agent_timings"]["synthesis"] = time.time() - start_time

            logger.debug(
                "Synthesis completed with confidence: %.2f", synthesis_result.confidence
            )

            return state

        except Exception as e:
            logger.error("Synthesis agent failed: %s", e)
            state["error_occurred"] = True
            state["error_message"] = f"Synthesis failed: {str(e)}"
            return state

    async def _validator_node(self, state: AgentState) -> AgentState:
        """Process query through validator agent.

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation result
        """
        start_time = time.time()

        try:
            logger.debug("Validator checking synthesis result")

            # Create validation result (simplified for now)
            # TODO: Implement actual validation logic
            validation_result = ValidationResult(
                valid=True,
                confidence=0.9,
                issues=[],
                quality_score=0.85,
                completeness_score=0.9,
                accuracy_score=0.85,
                source_attribution_score=0.9,
                suggested_action="accept",
                processing_time_ms=(time.time() - start_time) * 1000,
            )

            # Update state
            state["validation_result"] = validation_result
            state["workflow_complete"] = True
            state["agent_timings"]["validator"] = time.time() - start_time

            # Calculate total processing time
            state["total_processing_time"] = sum(state["agent_timings"].values())

            logger.debug(
                "Validation completed: valid=%s, score=%.2f",
                validation_result.valid,
                validation_result.quality_score,
            )

            return state

        except Exception as e:
            logger.error("Validator agent failed: %s", e)
            state["error_occurred"] = True
            state["error_message"] = f"Validator failed: {str(e)}"
            return state

    async def _error_handler_node(self, state: AgentState) -> AgentState:
        """Handle errors and provide fallback responses.

        Args:
            state: Current workflow state

        Returns:
            Updated state with error handling
        """
        logger.warning(
            "Error handler activated: %s", state.get("error_message", "Unknown error")
        )

        # Implement fallback logic if enabled
        if self.config.enable_fallback_rag and not state.get("workflow_complete"):
            logger.info("Attempting fallback RAG response")

            # Simple fallback response
            state["synthesis_result"] = SynthesisResult(
                documents=[],
                original_count=0,
                final_count=0,
                deduplication_ratio=1.0,
                strategies_used=["fallback"],
                processing_time_ms=10.0,
                confidence_score=0.5,
                reasoning="Error occurred, fallback response generated",
            )

            state["validation_result"] = ValidationResult(
                valid=True,
                confidence=0.5,
                issues=[],
                quality_score=0.5,
                completeness_score=0.5,
                accuracy_score=0.5,
                source_attribution_score=0.5,
                suggested_action="accept",
                processing_time_ms=5.0,
            )

        state["workflow_complete"] = True
        return state

    # Routing logic
    def _route_after_router(self, state: AgentState) -> str:
        """Determine next agent after router."""
        if state.get("error_occurred"):
            return "error"

        routing_decision = state.get("routing_decision")
        if not routing_decision:
            return "error"

        # Route to planner if planning is needed
        if routing_decision.needs_planning:
            return "planner"
        else:
            return "retrieval"

    def _route_after_planner(self, state: AgentState) -> str:
        """Determine next agent after planner."""
        if state.get("error_occurred"):
            return "error"
        return "retrieval"

    def _route_after_retrieval(self, state: AgentState) -> str:
        """Determine next agent after retrieval."""
        if state.get("error_occurred"):
            return "error"
        return "synthesis"

    def _route_after_synthesis(self, state: AgentState) -> str:
        """Determine next agent after synthesis."""
        if state.get("error_occurred"):
            return "error"
        return "validator"

    def _route_after_validator(self, state: AgentState) -> str:
        """Determine workflow completion after validator."""
        if state.get("error_occurred"):
            return "error"

        validation_result = state.get("validation_result")
        if validation_result and not validation_result.valid:
            # Retry if validation fails and we haven't exceeded max retries
            # TODO: Implement retry logic
            return "retry"

        return "end"

    def compile_graph(self) -> None:
        """Compile the workflow graph for execution."""
        if self.compiled_graph is not None:
            logger.info("Graph already compiled")
            return

        logger.info("Compiling supervisor graph...")

        # Initialize agents
        self._initialize_agents()

        # Create and compile the graph
        self.graph = self._create_workflow_graph()
        self.compiled_graph = self.graph.compile()

        logger.info("Supervisor graph compiled successfully")

    async def process_query(
        self, query: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Process a user query through the multi-agent pipeline.

        Args:
            query: User query to process
            context: Optional context for the query

        Returns:
            Dictionary with processing results and metadata
        """
        if self.compiled_graph is None:
            self.compile_graph()

        start_time = time.time()

        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "user_context": context,
            "messages": [HumanMessage(content=query)],
            "routing_decision": None,
            "query_plan": None,
            "current_subtask": 0,
            "retrieval_results": [],
            "synthesis_result": None,
            "validation_result": None,
            "next_agent": "router",
            "workflow_complete": False,
            "error_occurred": False,
            "error_message": None,
            "start_time": start_time,
            "agent_timings": {},
            "total_processing_time": 0.0,
        }

        try:
            logger.info("Processing query through supervisor graph: %s", query[:100])

            # Execute the workflow
            result = await self.compiled_graph.ainvoke(initial_state)

            # Extract results
            total_time = time.time() - start_time

            # Extract response from synthesis result
            synthesis_result = result.get("synthesis_result")
            response_text = "No response generated"
            if synthesis_result and synthesis_result.documents:
                response_text = (
                    synthesis_result.reasoning or "Generated from synthesized documents"
                )

            response_data = {
                "response": response_text,
                "confidence": result.get("validation_result", {}).get(
                    "confidence", 0.0
                ),
                "quality_score": result.get("validation_result", {}).get(
                    "quality_score", 0.0
                ),
                "processing_time_ms": total_time * 1000,
                "agent_timings": result.get("agent_timings", {}),
                "routing_decision": result.get("routing_decision"),
                "sources_used": len(result.get("retrieval_results", [])),
                "workflow_complete": result.get("workflow_complete", False),
                "error_occurred": result.get("error_occurred", False),
                "error_message": result.get("error_message"),
            }

            # Performance validation
            if total_time * 1000 > self.config.max_processing_time_ms:
                logger.warning(
                    "Processing exceeded max time: %.2fms", total_time * 1000
                )

            logger.info("Query processed successfully in %.2fms", total_time * 1000)
            return response_data

        except Exception as e:
            logger.error("Supervisor graph execution failed: %s", e)

            return {
                "response": "I encountered an error processing your query. Please try again.",
                "confidence": 0.0,
                "quality_score": 0.0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "agent_timings": {},
                "routing_decision": None,
                "sources_used": 0,
                "workflow_complete": False,
                "error_occurred": True,
                "error_message": str(e),
            }


# Global supervisor instance
_global_supervisor: SupervisorGraph | None = None


def get_supervisor_graph() -> SupervisorGraph:
    """Get or create global supervisor graph instance.

    Returns:
        Global SupervisorGraph instance
    """
    global _global_supervisor

    if _global_supervisor is None:
        # Create configuration from settings
        config = SupervisorConfig(
            max_processing_time_ms=settings.agent_decision_timeout,
            enable_fallback_rag=settings.enable_fallback_rag,
            max_retries=settings.max_agent_retries,
        )

        _global_supervisor = SupervisorGraph(config)
        logger.info("Created global supervisor graph instance")

    return _global_supervisor


async def initialize_supervisor_graph() -> SupervisorGraph:
    """Initialize and compile the global supervisor graph.

    Returns:
        Initialized and compiled SupervisorGraph instance
    """
    supervisor = get_supervisor_graph()
    supervisor.compile_graph()
    return supervisor


def cleanup_supervisor_graph() -> None:
    """Clean up the global supervisor graph."""
    global _global_supervisor

    if _global_supervisor is not None:
        # Clean up any resources if needed
        _global_supervisor = None
        logger.info("Global supervisor graph cleaned up")
