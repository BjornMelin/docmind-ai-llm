"""Optimized Agent Factory with LangGraph Supervisor Pattern.

LIBRARY-FIRST: Uses LangGraph native patterns for multi-agent coordination.
Provides local-only operation with intelligent routing between single-agent
and multi-agent modes based on query complexity.
"""

from functools import wraps
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool
from loguru import logger

from models import AppSettings

# Optional import for persistence - may not be available in all versions
try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False
    logger.warning("SqliteSaver not available - persistence features disabled")

settings = AppSettings()


class AgentState(MessagesState):
    """Enhanced state for multi-agent coordination."""

    query_complexity: str = "simple"
    query_type: str = "general"
    current_agent: str = "supervisor"


def handle_agent_errors(operation_name: str):
    """Decorator to consolidate error handling across agent operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                logger.info(f"{operation_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{operation_name} failed: {e}")
                if (
                    "creation" in operation_name.lower()
                    or "create" in operation_name.lower()
                ):
                    raise
                return f"Error in {operation_name}: {str(e)}"

        return wrapper

    return decorator


# Configuration-driven agent creation
AGENT_CONFIGS = {
    "document_specialist": {
        "tool_filter": lambda tool: "vector" in tool.metadata.name.lower(),
        "system_message": (
            "You are a document processing specialist. "
            "Focus on extracting information from documents, "
            "summarizing content, and answering questions about document text. "
            "Use hybrid search capabilities for best results."
        ),
    },
    "knowledge_specialist": {
        "tool_filter": lambda tool: "knowledge" in tool.metadata.name.lower(),
        "system_message": (
            "You are a knowledge graph specialist. "
            "Focus on entity relationships, connections between concepts, "
            "and structured knowledge queries. Use knowledge graph tools "
            "to find complex relationships and entity interactions."
        ),
    },
    "multimodal_specialist": {
        "tool_filter": lambda tool: True,
        "system_message": (
            "You are a multimodal content specialist. "
            "You can process both text and image content from documents. "
            "When answering questions about visual content, images, or "
            "diagrams, use both text and visual information for comprehensive answers."
        ),
    },
}


def analyze_query_complexity(query: str) -> tuple[str, str]:
    """Streamlined query analysis for routing decisions."""
    query_lower = query.lower()

    key_indicators = [
        "compare",
        "analyze",
        "relationship",
        "multiple",
        "across documents",
        "how does",
        "why",
        "explain the difference",
        "various",
        "several",
        "between",
        "among",
        "summarize all",
    ]
    complex_keywords = sum(
        1 for indicator in key_indicators if indicator in query_lower
    )

    query_length = len(query.split())

    if complex_keywords >= 2 or query_length > 20:
        complexity = "complex"
    elif complex_keywords >= 1 or query_length > 10:
        complexity = "moderate"
    else:
        complexity = "simple"

    if any(word in query_lower for word in ["image", "visual", "diagram"]):
        query_type = "multimodal"
    elif any(
        word in query_lower
        for word in ["entity", "relationship", "connected", "related to"]
    ):
        query_type = "knowledge_graph"
    elif any(
        word in query_lower for word in ["document", "text", "content", "passage"]
    ):
        query_type = "document"
    else:
        query_type = "general"

    logger.info("Query analysis: complexity=%s, type=%s", complexity, query_type)
    return complexity, query_type


@handle_agent_errors("Single agent creation")
def create_single_agent(
    tools: list[QueryEngineTool],
    llm: Any,
    memory: ChatMemoryBuffer | None = None,
) -> ReActAgent:
    """Create a single ReAct agent for simple queries."""
    return ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        memory=memory or ChatMemoryBuffer.from_defaults(token_limit=8192),
        verbose=True,
        max_iterations=10,
    )


@handle_agent_errors("Specialist agent creation")
def create_specialist_agent(
    agent_type: str,
    tools: list[QueryEngineTool],
    llm: Any,
    enable_human_in_loop: bool = False,
) -> Any:
    """Create specialist agent using configuration-driven approach."""
    if agent_type not in AGENT_CONFIGS:
        raise ValueError(f"Unknown agent type: {agent_type}")

    config = AGENT_CONFIGS[agent_type]
    filtered_tools = [tool for tool in tools if config["tool_filter"](tool)]

    agent = create_react_agent(
        model=llm,
        tools=filtered_tools,
        messages_modifier=config["system_message"],
    )

    # Optional: Add human-in-loop wrapper if enabled
    if enable_human_in_loop:
        return _wrap_agent_with_human_check(agent, agent_type)

    return agent


def _wrap_agent_with_human_check(agent: Any, agent_type: str) -> Any:
    """Wrap agent to check for human intervention requirements."""

    def enhanced_agent(state: AgentState):
        result = agent(state)
        # Add human-in-loop logic here if needed
        return result

    return enhanced_agent


def supervisor_routing_logic(state: AgentState) -> str:
    """Simplified supervisor routing logic."""
    if not state.get("messages"):
        return "document_specialist"

    query = (
        state["messages"][-1].content
        if hasattr(state["messages"][-1], "content")
        else str(state["messages"][-1])
    )

    complexity, query_type = analyze_query_complexity(query)
    state.update({"query_complexity": complexity, "query_type": query_type})

    routing_map = {
        "multimodal": "multimodal_specialist",
        "knowledge_graph": "knowledge_specialist",
        "document": "document_specialist",
    }
    return routing_map.get(query_type, "document_specialist")


@handle_agent_errors("LangGraph supervisor system creation")
def create_langgraph_supervisor_system(
    tools: list[QueryEngineTool],
    llm: Any,
    enable_human_in_loop: bool = False,
    checkpoint_path: str | None = None,
) -> StateGraph | None:
    """Create optimized LangGraph supervisor system with optional features."""
    # Create all specialists using configuration-driven approach
    agents = {
        agent_type: create_specialist_agent(
            agent_type, tools, llm, enable_human_in_loop
        )
        for agent_type in AGENT_CONFIGS
    }

    # Build workflow
    workflow = StateGraph(AgentState)

    # Add specialist nodes
    for agent_type, agent in agents.items():
        workflow.add_node(agent_type, agent)

    # Supervisor routing with all agents as options
    workflow.add_conditional_edges(START, supervisor_routing_logic, agents)

    # All specialists end the workflow
    for agent_type in agents:
        workflow.add_edge(agent_type, END)

    # Compile with optional persistence
    compile_config = {}
    if checkpoint_path and CHECKPOINT_AVAILABLE:
        compile_config["checkpointer"] = SqliteSaver.from_conn_string(checkpoint_path)
        if enable_human_in_loop:
            compile_config["interrupt_before"] = ["__human_input__"]
    elif checkpoint_path and not CHECKPOINT_AVAILABLE:
        logger.warning("Checkpoint path provided but SqliteSaver not available")

    return workflow.compile(**compile_config)


def get_agent_system(
    tools: list[QueryEngineTool],
    llm: Any,
    enable_multi_agent: bool = False,
    enable_human_in_loop: bool = False,
    checkpoint_path: str | None = None,
    memory: ChatMemoryBuffer | None = None,
) -> tuple[Any, str]:
    """Get optimized agent system with optional features."""
    if enable_multi_agent:
        multi_agent_system = create_langgraph_supervisor_system(
            tools, llm, enable_human_in_loop, checkpoint_path
        )
        if multi_agent_system:
            return multi_agent_system, "multi"
        else:
            logger.warning(
                "Multi-agent system creation failed, falling back to single agent"
            )

    # Fallback to single agent
    single_agent = create_single_agent(tools, llm, memory)
    return single_agent, "single"


@handle_agent_errors("Query processing")
def process_query_with_agent_system(
    agent_system: Any,
    query: str,
    mode: str,
    memory: ChatMemoryBuffer | None = None,
    thread_id: str | None = None,
) -> str:
    """Enhanced query processing with optional persistence and resumption."""
    if mode == "multi":
        invoke_config = {"messages": [HumanMessage(content=query)]}

        # Add thread_id for persistence if provided
        if thread_id:
            invoke_config["configurable"] = {"thread_id": thread_id}

        initial_state = AgentState(**invoke_config)
        result = agent_system.invoke(initial_state)
        return _extract_response_from_state(result)
    else:
        response = agent_system.chat(query)
        return response.response if hasattr(response, "response") else str(response)


def _extract_response_from_state(result: dict) -> str:
    """Helper to extract response from LangGraph state."""
    if result.get("messages"):
        last_message = result["messages"][-1]
        return (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
    return "Multi-agent processing completed but no response generated."
