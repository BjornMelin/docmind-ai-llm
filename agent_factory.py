"""Agent Factory with LangGraph Supervisor Pattern.

LIBRARY-FIRST: Uses LangGraph native patterns for multi-agent coordination.
Provides local-only operation with intelligent routing between single-agent
and multi-agent modes based on query complexity.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool

from models import AppSettings

settings = AppSettings()
logger = logging.getLogger(__name__)


"""
Class documentation for AgentState.

TODO: Add detailed description.
"""


class AgentState(MessagesState):
    """Enhanced state for multi-agent coordination."""

    # Query analysis
    query_complexity: str = "simple"  # simple, complex, specialized
    query_type: str = "general"  # general, document, knowledge_graph, multimodal

    # Agent coordination
    current_agent: str = "supervisor"
    task_progress: dict[str, Any] = {}
    agent_outputs: dict[str, Any] = {}

    # Results
    final_answer: str = ""
    confidence_score: float = 0.0


def analyze_query_complexity(query: str) -> tuple[str, str]:
    """Analyze query complexity and type for intelligent routing.

    Args:
        query: User query string.

    Returns:
        Tuple of (complexity, query_type) for routing decisions.
    """
    query_lower = query.lower()

    # Determine complexity based on query characteristics
    complexity_indicators = [
        "compare",
        "analyze",
        "relationship",
        "how does",
        "why",
        "explain the difference",
        "multiple",
        "various",
        "several",
        "across documents",
        "between",
        "among",
        "summarize all",
    ]

    complex_keywords = sum(
        1 for indicator in complexity_indicators if indicator in query_lower
    )
    query_length = len(query.split())

    if complex_keywords >= 2 or query_length > 20:
        complexity = "complex"
    elif complex_keywords >= 1 or query_length > 10:
        complexity = "moderate"
    else:
        complexity = "simple"

    # Determine query type
    if any(
        word in query_lower
        for word in ["image", "picture", "visual", "diagram", "chart"]
    ):
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


def create_single_agent(
    tools: list[QueryEngineTool],
    llm: Any,
    memory: ChatMemoryBuffer | None = None,
) -> ReActAgent:
    """Create a single ReAct agent for simple queries.

    Args:
        tools: List of query engine tools.
        llm: Language model instance.
        memory: Optional chat memory buffer.

    Returns:
        Configured ReActAgent.
    """
    try:
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            memory=memory or ChatMemoryBuffer.from_defaults(token_limit=8192),
            verbose=True,
            max_iterations=10,
        )
        logger.info("Single ReActAgent created successfully")
        return agent
    except Exception as e:
        logger.error("Single agent creation failed: %s", e)
        raise


def create_document_specialist_agent(tools: list[QueryEngineTool], llm: Any) -> Any:
    """Create document processing specialist agent.

    Args:
        tools: Available query engine tools.
        llm: Language model instance.

    Returns:
        Document specialist agent.
    """
    # Filter tools for document processing
    doc_tools = [tool for tool in tools if "vector" in tool.metadata.name.lower()]

    return create_react_agent(
        model=llm,
        tools=doc_tools,
        messages_modifier=(
            "You are a document processing specialist. "
            "Focus on extracting information from documents, "
            "summarizing content, and answering questions about document text. "
            "Use hybrid search capabilities for best results."
        ),
    )


def create_knowledge_specialist_agent(tools: list[QueryEngineTool], llm: Any) -> Any:
    """Create knowledge graph specialist agent.

    Args:
        tools: Available query engine tools.
        llm: Language model instance.

    Returns:
        Knowledge graph specialist agent.
    """
    # Filter tools for knowledge graph queries
    kg_tools = [tool for tool in tools if "knowledge" in tool.metadata.name.lower()]

    return create_react_agent(
        model=llm,
        tools=kg_tools,
        messages_modifier=(
            "You are a knowledge graph specialist. "
            "Focus on entity relationships, connections between concepts, "
            "and structured knowledge queries. Use knowledge graph tools "
            "to find complex relationships and entity interactions."
        ),
    )


def create_multimodal_specialist_agent(tools: list[QueryEngineTool], llm: Any) -> Any:
    """Create multimodal content specialist agent.

    Args:
        tools: Available query engine tools.
        llm: Language model instance.

    Returns:
        Multimodal specialist agent.
    """
    return create_react_agent(
        model=llm,
        tools=tools,  # All tools for multimodal processing
        messages_modifier=(
            "You are a multimodal content specialist. "
            "You can process both text and image content from documents. "
            "When answering questions about visual content, images, or "
            "diagrams, use both text and visual information for comprehensive answers."
        ),
    )


def supervisor_routing_logic(state: AgentState) -> str:
    """LangGraph supervisor routing logic.

    Args:
        state: Current agent state.

    Returns:
        Next agent to route to.
    """
    # Get the latest message
    if not state.get("messages"):
        return "document_specialist"

    last_message = state["messages"][-1]
    query = (
        last_message.content if hasattr(last_message, "content") else str(last_message)
    )

    # Analyze query for routing
    complexity, query_type = analyze_query_complexity(query)

    # Update state
    state["query_complexity"] = complexity
    state["query_type"] = query_type

    # Route based on query type and complexity
    if query_type == "multimodal":
        return "multimodal_specialist"
    elif query_type == "knowledge_graph":
        return "knowledge_specialist"
    elif complexity in ["complex", "moderate"]:
        # For complex queries, start with document specialist
        return "document_specialist"
    else:
        # Simple queries go to document specialist
        return "document_specialist"


def create_langgraph_supervisor_system(
    tools: list[QueryEngineTool], llm: Any
) -> StateGraph | None:
    """Create LangGraph supervisor multi-agent system.

    Args:
        tools: Available query engine tools.
        llm: Language model instance.

    Returns:
        Compiled LangGraph StateGraph or None if not available.
    """
    try:
        # Create specialist agents
        doc_agent = create_document_specialist_agent(tools, llm)
        kg_agent = create_knowledge_specialist_agent(tools, llm)
        multimodal_agent = create_multimodal_specialist_agent(tools, llm)

        # Create supervisor graph
        workflow = StateGraph(AgentState)

        # Add specialist nodes
        workflow.add_node("document_specialist", doc_agent)
        workflow.add_node("knowledge_specialist", kg_agent)
        workflow.add_node("multimodal_specialist", multimodal_agent)

        # Supervisor routing
        workflow.add_conditional_edges(
            START,
            supervisor_routing_logic,
            {
                "document_specialist": "document_specialist",
                "knowledge_specialist": "knowledge_specialist",
                "multimodal_specialist": "multimodal_specialist",
            },
        )

        # All specialists end the workflow
        workflow.add_edge("document_specialist", END)
        workflow.add_edge("knowledge_specialist", END)
        workflow.add_edge("multimodal_specialist", END)

        # Compile the graph
        multi_agent_system = workflow.compile()

        logger.info("LangGraph supervisor multi-agent system created successfully")
        return multi_agent_system

    except Exception as e:
        logger.error("LangGraph supervisor system creation failed: %s", e)
        return None


def get_agent_system(
    tools: list[QueryEngineTool],
    llm: Any,
    enable_multi_agent: bool = False,
    memory: ChatMemoryBuffer | None = None,
) -> tuple[Any, str]:
    """Get appropriate agent system based on configuration.

    Args:
        tools: Available query engine tools.
        llm: Language model instance.
        enable_multi_agent: Whether to use multi-agent mode.
        memory: Optional chat memory buffer.

    Returns:
        Tuple of (agent_system, mode) where mode is "single" or "multi".
    """
    if enable_multi_agent:
        multi_agent_system = create_langgraph_supervisor_system(tools, llm)
        if multi_agent_system:
            return multi_agent_system, "multi"
        else:
            logger.warning(
                "Multi-agent system creation failed, falling back to single agent"
            )

    # Fallback to single agent
    single_agent = create_single_agent(tools, llm, memory)
    return single_agent, "single"


def process_query_with_agent_system(
    agent_system: Any,
    query: str,
    mode: str,
    memory: ChatMemoryBuffer | None = None,
) -> str:
    """Process query with the appropriate agent system.

    Args:
        agent_system: Single agent or multi-agent system.
        query: User query string.
        mode: "single" or "multi" agent mode.
        memory: Optional chat memory for single agent mode.

    Returns:
        Agent response string.
    """
    try:
        if mode == "multi":
            # LangGraph multi-agent processing
            initial_state = AgentState(messages=[HumanMessage(content=query)])
            result = agent_system.invoke(initial_state)

            # Extract response from final state
            if result.get("messages"):
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                else:
                    return str(last_message)
            else:
                return "Multi-agent processing completed but no response generated."

        else:
            # Single agent processing
            if hasattr(agent_system, "chat"):
                response = agent_system.chat(query)
                return (
                    response.response
                    if hasattr(response, "response")
                    else str(response)
                )
            else:
                return "Single agent processing error."

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return f"Error processing query: {str(e)}"
