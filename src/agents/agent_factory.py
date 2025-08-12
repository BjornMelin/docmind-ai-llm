"""Minimal LlamaIndex ReActAgent Factory - KISS Implementation.

Simple, library-first implementation using pure LlamaIndex ReActAgent.
Replaces complex multi-agent orchestration with single intelligent agent.
Target: <80 lines, <2s response time, 82.5% accuracy
"""


from llama_index.core.agent import ReActAgent
from llama_index.core.llms.base import BaseLLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool
from loguru import logger


def create_agentic_rag_system(
    tools: list[QueryEngineTool], llm: BaseLLM, memory: ChatMemoryBuffer | None = None
) -> ReActAgent:
    """Create single ReActAgent with full agentic capabilities."""
    if not tools:
        logger.warning("No tools provided for ReActAgent creation")
        return ReActAgent.from_tools([], llm)

    system_prompt = """You are an intelligent document analysis agent.
Think step-by-step and use the most appropriate tools for each query:
- Use multiple tools for comprehensive analysis
- Cross-reference results when needed
- Provide detailed, well-structured responses
- Always explain your reasoning"""

    return ReActAgent.from_tools(
        tools=tools,
        llm=llm,
        memory=memory or ChatMemoryBuffer.from_defaults(token_limit=8192),
        system_prompt=system_prompt,
        verbose=True,
        max_iterations=3,
    )


# Backward compatibility functions
def create_single_agent(
    tools: list[QueryEngineTool], llm: BaseLLM, memory: ChatMemoryBuffer | None = None
) -> ReActAgent:
    """Legacy compatibility."""
    return create_agentic_rag_system(tools, llm, memory)


def get_agent_system(
    tools: list[QueryEngineTool],
    llm: BaseLLM,
    enable_multi_agent: bool = False,
    enable_human_in_loop: bool = False,
    checkpoint_path: str | None = None,
    memory: ChatMemoryBuffer | None = None,
) -> tuple[ReActAgent, str]:
    """Get ReActAgent system - always returns single optimized agent."""
    if enable_multi_agent:
        logger.info("Multi-agent requested but using single optimized agent")
    return create_agentic_rag_system(tools, llm, memory), "single"


def process_query_with_agent_system(
    agent_system: ReActAgent,
    query: str,
    mode: str,
    memory: ChatMemoryBuffer | None = None,
    thread_id: str | None = None,
) -> str:
    """Process query with ReActAgent."""
    try:
        response = agent_system.chat(query)
        return response.response if hasattr(response, "response") else str(response)
    except (ValueError, TypeError, RuntimeError, AttributeError) as e:
        logger.error(f"Query processing failed: {e}")
        return f"Error processing query: {str(e)}"
