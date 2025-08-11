"""Agent utilities for DocMind AI with hybrid search integration.

This module provides comprehensive agent functionality including:
- ReActAgent creation with tool integration
- Hybrid search tools with RRF fusion and ColBERT reranking
- Query engine configuration with vector and knowledge graph indexes
- Asynchronous streaming chat capabilities
- Multi-step reasoning with agent workflows
- Memory management and context preservation

Agent capabilities:
- Hybrid vector search with BGE-Large + SPLADE++ embeddings
- Knowledge graph queries for entity relationships
- ColBERT reranking for improved relevance
- RRF fusion with research-backed weights
- GPU acceleration when available
- Streaming responses with async generators

Example:
    Creating and using agents::

        from agents.agent_utils import create_agent_with_tools, create_tools_from_index

        # Create tools from indexes
        tools = create_tools_from_index(index_data)

        # Create ReActAgent
        agent = create_agent_with_tools(index_data, llm)

        # Use agent for queries
        response = agent.chat("Analyze the document content")

        # Async streaming chat
        async for chunk in chat_with_agent(agent, query, memory):
            print(chunk, end="")

Attributes:
    settings (AppSettings): Global application settings for agent configuration.
"""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.ollama import Ollama
from loguru import logger

from models import AppSettings
from utils.exceptions import (
    ConfigurationError,
    handle_agent_error,
)
from utils.retry_utils import async_with_timeout, with_fallback


def log_error_with_context(
    error: Exception, operation: str, context: dict | None = None, **kwargs
) -> None:
    """Log errors with comprehensive context information."""
    error_context = {
        "operation": operation,
        "error_type": type(error).__name__,
        "error_message": str(error),
        **(context or {}),
        **kwargs,
    }
    logger.error(
        f"Operation failed: {operation}",
        extra={"error_context": error_context},
        exception=error,
    )


def log_performance(operation: str, duration: float, **kwargs) -> None:
    """Log performance metrics with structured data."""
    logger.info(
        f"Performance: {operation} completed",
        extra={
            "performance": {
                "operation": operation,
                "duration_seconds": round(duration, 3),
                "duration_human": f"{duration:.2f}s",
                **kwargs,
            }
        },
    )


settings = AppSettings()


@with_fallback(lambda index_data: [])
def create_tools_from_index(index_data: dict[str, Any]) -> list[QueryEngineTool]:
    """Create enhanced query tools with structured error handling.

    Uses the ToolFactory to create QueryEngineTool instances from provided indexes,
    ensuring consistent configuration and eliminating code duplication.
    Provides the same search capabilities with centralized tool creation
    and comprehensive error handling.

    Args:
        index_data: Dictionary containing indexed components:
            - 'vector' (VectorStoreIndex): Hybrid vector index
            - 'kg' (KnowledgeGraphIndex | None): Knowledge graph index
            - 'retriever' (QueryFusionRetriever | None): Hybrid fusion retriever

    Returns:
        List of QueryEngineTool instances with consistent configuration.
        Returns empty list on critical failures.

    Raises:
        AgentError: If tool creation fails critically.

    Example:
        >>> index_data = {'vector': vector_idx, 'kg': kg_idx, 'retriever': retriever}
        >>> tools = create_tools_from_index(index_data)
        >>> print(f"Created {len(tools)} query tools")
    """
    start_time = time.perf_counter()

    logger.info(
        "Creating query tools from index data",
        extra={
            "has_vector": "vector" in index_data and index_data["vector"] is not None,
            "has_kg": "kg" in index_data and index_data["kg"] is not None,
            "has_retriever": "retriever" in index_data
            and index_data["retriever"] is not None,
        },
    )

    try:
        from agents.tool_factory import ToolFactory

        # Validate index_data contains required components
        if not index_data:
            raise ConfigurationError(
                "Empty index_data provided for tool creation",
                context={
                    "index_data_keys": list(index_data.keys()) if index_data else []
                },
                operation="tool_creation_validation",
            )

        tools = ToolFactory.create_basic_tools(index_data)

        duration = time.perf_counter() - start_time
        log_performance(
            "tool_creation_from_index",
            duration,
            tool_count=len(tools),
            has_vector=index_data.get("vector") is not None,
            has_kg=index_data.get("kg") is not None,
        )

        logger.success(
            f"Created {len(tools)} query tools from index data",
            extra={
                "tool_count": len(tools),
                "duration_ms": round(duration * 1000, 2),
            },
        )

        return tools

    except Exception as e:
        log_error_with_context(
            e,
            "tool_creation_from_index",
            context={
                "index_data_keys": list(index_data.keys()) if index_data else [],
                "has_vector": index_data.get("vector") is not None
                if index_data
                else False,
                "has_kg": index_data.get("kg") is not None if index_data else False,
            },
        )

        # Re-raise to trigger fallback decorator
        raise handle_agent_error(
            e,
            operation="tool_creation_from_index",
            index_data_keys=list(index_data.keys()) if index_data else [],
        ) from e


@with_fallback(
    lambda index_data, llm: ReActAgent.from_tools(
        tools=[],
        llm=llm,
        verbose=True,
        memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
    )
)
def create_agent_with_tools(index_data: dict[str, Any], llm: Any) -> ReActAgent:
    """Create ReActAgent with structured error handling.

    Constructs a fully-featured ReActAgent equipped with hybrid search tools,
    knowledge graph queries, and memory management. Integrates all available
    features including RRF fusion, ColBERT reranking, and GPU
    acceleration for optimal performance with comprehensive error handling.

    Agent capabilities:
    - Multi-step reasoning with ReAct pattern
    - Hybrid vector search with dense/sparse embeddings
    - Knowledge graph entity relationship queries
    - Memory-aware conversations with token management
    - Error handling and graceful degradation
    - Verbose logging for debugging and monitoring
    - Structured error recovery with fallbacks

    Args:
        index_data: Dictionary containing indexed components from index creation:
            - 'vector' (VectorStoreIndex): Hybrid vector index with embeddings
            - 'kg' (KnowledgeGraphIndex | None): Knowledge graph index
            - 'retriever' (QueryFusionRetriever | None): Fusion retriever
        llm: Language model instance (Ollama, LMStudio, etc.) for agent reasoning.
            Must support the ReAct prompting pattern.

    Returns:
        ReActAgent configured with:
        - Enhanced query tools from create_tools_from_index
        - Chat memory buffer with 8192 token limit
        - Maximum 10 reasoning iterations
        - Verbose output for debugging
        - Robust error handling with fallbacks

    Raises:
        AgentError: If agent creation fails critically after fallbacks.

    Note:
        The agent uses ReAct (Reasoning + Acting) pattern for systematic
        problem-solving. Memory buffer prevents context overflow in long
        conversations. Tool selection is automatic based on query analysis.
        Falls back to basic configuration on errors.

    Example:
        >>> from llama_index.llms.ollama import Ollama
        >>> llm = Ollama(model="llama2")
        >>> agent = create_agent_with_tools(index_data, llm)
        >>> response = agent.chat("What are the main themes in the documents?")
        >>> print(response.response)
    """
    start_time = time.perf_counter()

    logger.info(
        "Creating ReActAgent with tools",
        extra={
            "llm_type": type(llm).__name__,
            "has_index_data": bool(index_data),
            "index_components": list(index_data.keys()) if index_data else [],
        },
    )

    try:
        # Get tools from index data with error handling
        tools = create_tools_from_index(index_data)

        if not tools:
            logger.warning(
                "No tools created from index data, proceeding with empty tool list",
                extra={
                    "index_data_keys": list(index_data.keys()) if index_data else []
                },
            )

        # Create ReActAgent with enhanced memory and error handling
        try:
            agent = ReActAgent.from_tools(
                tools=tools,
                llm=llm,
                verbose=True,
                max_iterations=10,
                memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
            )

            tool_names = [tool.metadata.name for tool in tools]

            duration = time.perf_counter() - start_time
            log_performance(
                "react_agent_creation",
                duration,
                tool_count=len(tools),
                max_iterations=10,
                memory_token_limit=8192,
            )

            logger.success(
                f"ReActAgent created with {len(tools)} enhanced tools",
                extra={
                    "tool_count": len(tools),
                    "tool_names": tool_names,
                    "llm_type": type(llm).__name__,
                    "duration_ms": round(duration * 1000, 2),
                },
            )
            return agent

        except Exception as e:
            log_error_with_context(
                e,
                "react_agent_creation_enhanced",
                context={
                    "tool_count": len(tools),
                    "llm_type": type(llm).__name__,
                    "memory_token_limit": 8192,
                },
            )

            logger.warning(
                "Enhanced ReActAgent creation failed, trying basic configuration",
                extra={"original_error": str(e)},
            )

            # Fallback with basic configuration
            agent = ReActAgent.from_tools(
                tools,
                llm=llm,
                verbose=True,
            )

            tool_names = [tool.metadata.name for tool in tools] if tools else []

            logger.warning(
                f"Using fallback ReActAgent configuration with {len(tools)} tools",
                extra={
                    "tool_count": len(tools),
                    "tool_names": tool_names,
                    "fallback_mode": True,
                },
            )
            return agent

    except Exception as e:
        log_error_with_context(
            e,
            "react_agent_creation",
            context={
                "llm_type": type(llm).__name__,
                "index_data_keys": list(index_data.keys()) if index_data else [],
            },
        )

        # Re-raise to trigger fallback decorator
        raise handle_agent_error(
            e,
            operation="react_agent_creation",
            llm_type=type(llm).__name__,
            index_data_keys=list(index_data.keys()) if index_data else [],
        ) from e


@with_fallback(
    lambda agent,
    index_data,
    prompt_type: f"Analysis failed for prompt type: {prompt_type}"
)
def analyze_documents_agentic(
    agent: ReActAgent, index_data: dict[str, Any], prompt_type: str
) -> str:
    """Perform agentic document analysis with structured error handling.

    Executes document analysis using the ReActAgent's multi-step reasoning
    capabilities. The agent systematically queries different index types
    and synthesizes information to provide comprehensive analysis with
    comprehensive error handling and performance monitoring.

    Analysis process:
    1. Query vector index for content similarity
    2. Query knowledge graph for entity relationships
    3. Synthesize findings across multiple reasoning steps
    4. Generate structured analysis output

    Args:
        agent: ReActAgent instance configured with query tools. If None,
            creates a fallback agent with basic configuration.
        index_data: Dictionary containing indexed components:
            - 'vector' (VectorStoreIndex): For content similarity queries
            - 'kg' (KnowledgeGraphIndex): For entity relationship queries
        prompt_type: Type of analysis prompt to execute. Used to guide
            the agent's reasoning and focus areas.

    Returns:
        Analysis response string containing the agent's multi-step reasoning
        and final conclusions.

    Raises:
        AgentError: If analysis fails critically after fallbacks.

    Note:
        This function provides a compatibility layer and fallback mechanism.
        In normal operation, agents should be properly initialized via
        create_agent_with_tools in the main application.
        Falls back to basic analysis on errors.

    Example:
        >>> agent = create_agent_with_tools(index_data, llm)
        >>> analysis = analyze_documents_agentic(agent, index_data, "summary")
        >>> print(analysis)
    """
    start_time = time.perf_counter()

    logger.info(
        "Starting agentic document analysis",
        extra={
            "prompt_type": prompt_type,
            "has_agent": agent is not None,
            "has_vector": "vector" in index_data and index_data["vector"] is not None,
            "has_kg": "kg" in index_data and index_data["kg"] is not None,
        },
    )

    try:
        # Validate inputs
        if not index_data:
            raise ConfigurationError(
                "Empty index_data provided for analysis",
                context={"prompt_type": prompt_type},
                operation="analysis_input_validation",
            )

        # Safe query engine creation with error handling
        vector_query_engine = None
        kg_query_engine = None

        if index_data.get("vector"):
            try:
                vector_query_engine = index_data["vector"].as_query_engine()
            except Exception as e:
                logger.warning(f"Failed to create vector query engine: {e}")

        if index_data.get("kg"):
            try:
                kg_query_engine = index_data["kg"].as_query_engine()
            except Exception as e:
                logger.warning(f"Failed to create KG query engine: {e}")

        # Use ToolFactory for consistent tool creation
        from agents.tool_factory import ToolFactory

        tools = []
        if vector_query_engine:
            try:
                tools.append(
                    ToolFactory.create_query_tool(
                        vector_query_engine,
                        "vector_query",
                        "Query documents using vector similarity search",
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to create vector query tool: {e}")

        if kg_query_engine:
            try:
                tools.append(
                    ToolFactory.create_query_tool(
                        kg_query_engine,
                        "knowledge_graph_query",
                        "Query knowledge graph for entity relationships",
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to create KG query tool: {e}")

        if not agent:
            logger.warning(
                "No agent provided, creating fallback agent",
                extra={"tool_count": len(tools), "prompt_type": prompt_type},
            )

            # Agent will be properly initialized in app.py with user-selected LLM
            # This is a fallback that should not be reached in normal operation
            agent = ReActAgent.from_tools(
                tools,
                llm=Ollama(model=settings.default_model),
                verbose=True,
            )

        # Perform analysis with error handling
        try:
            response = agent.chat(f"Analyze with prompt: {prompt_type}")
            analysis_result = response.response

        except Exception as e:
            log_error_with_context(
                e,
                "agent_chat_analysis",
                context={
                    "prompt_type": prompt_type,
                    "tool_count": len(tools) if tools else 0,
                },
            )

            # Fallback to basic analysis
            logger.warning(
                f"Agent chat failed, providing basic analysis: {e}",
                extra={"prompt_type": prompt_type},
            )
            analysis_result = (
                f"Analysis partially completed for prompt type: {prompt_type}. "
                "Agent encountered processing issues."
            )

        # Log performance and results
        duration = time.perf_counter() - start_time
        log_performance(
            "agentic_document_analysis",
            duration,
            prompt_type=prompt_type,
            tool_count=len(tools) if tools else 0,
            has_vector=vector_query_engine is not None,
            has_kg=kg_query_engine is not None,
        )

        logger.success(
            f"Agentic analysis completed for prompt type: {prompt_type}",
            extra={
                "prompt_type": prompt_type,
                "analysis_length": len(analysis_result),
                "duration_seconds": round(duration, 2),
            },
        )

        return analysis_result

    except Exception as e:
        log_error_with_context(
            e,
            "agentic_document_analysis",
            context={
                "prompt_type": prompt_type,
                "has_agent": agent is not None,
                "index_data_keys": list(index_data.keys()) if index_data else [],
            },
        )

        # Re-raise to trigger fallback decorator
        raise handle_agent_error(
            e,
            operation="agentic_document_analysis",
            prompt_type=prompt_type,
            has_agent=agent is not None,
        ) from e


@async_with_timeout(timeout_seconds=120)  # 2 minute timeout for chat
async def chat_with_agent(
    agent: ReActAgent, user_input: str, memory: ChatMemoryBuffer
) -> AsyncGenerator[str, None]:
    """Stream chat responses from ReActAgent with structured error handling.

    Provides asynchronous streaming chat interface with the ReActAgent,
    supporting knowledge graph queries, multimodal processing, and memory
    management. Yields response chunks as they become available for
    real-time user interaction with comprehensive error handling.

    Features:
    - Asynchronous streaming response generation
    - Memory-aware conversation context
    - Multi-step reasoning with intermediate results
    - Support for multimodal content processing
    - Error handling with meaningful error messages
    - Compatible with various LLM backends
    - Timeout protection and performance monitoring

    Args:
        agent: ReActAgent instance configured with appropriate tools and LLM.
            Must support async streaming chat functionality.
        user_input: User query string. Can include questions about document
            content, entity relationships, or general analysis requests.
        memory: ChatMemoryBuffer instance for maintaining conversation context
            across multiple interactions.

    Yields:
        str: Chunks of response text as they become available from the agent.
        Each chunk represents a partial response that can be displayed
        immediately for improved user experience.

    Raises:
        AgentError: If chat generation fails critically.
        asyncio.TimeoutError: If chat exceeds timeout limit.

    Note:
        The function uses asyncio.to_thread for compatibility with synchronous
        agent implementations. Multimodal handling depends on the underlying
        LLM capabilities (e.g., Gemma for native multimodal, Nemotron for text).
        Includes timeout protection and error recovery.

    Example:
        >>> import asyncio
        >>> from llama_index.core.memory import ChatMemoryBuffer
        >>>
        >>> async def chat_example():
        ...     memory = ChatMemoryBuffer.from_defaults()
        ...     async for chunk in chat_with_agent(agent, "Summarize the docs", memory):
        ...         print(chunk, end="", flush=True)
        >>>
        >>> asyncio.run(chat_example())
    """
    start_time = time.perf_counter()
    chunk_count = 0

    logger.info(
        "Starting agent chat",
        extra={
            "user_input_length": len(user_input),
            "agent_type": type(agent).__name__,
            "memory_token_limit": getattr(memory, "token_limit", None),
        },
    )

    try:
        # Validate inputs
        if not user_input or not user_input.strip():
            raise ConfigurationError(
                "Empty user input provided for chat",
                context={"user_input_length": len(user_input)},
                operation="chat_input_validation",
            )

        if not agent:
            raise ConfigurationError(
                "No agent provided for chat",
                context={"user_input_length": len(user_input)},
                operation="chat_agent_validation",
            )

        # Perform async streaming chat with error handling
        try:
            response = await asyncio.to_thread(
                agent.async_stream_chat, user_input, memory=memory
            )

            async for chunk in response.async_response_gen():  # Using async_gen
                chunk_count += 1
                yield chunk

        except Exception as stream_error:
            log_error_with_context(
                stream_error,
                "agent_stream_chat",
                context={
                    "user_input_length": len(user_input),
                    "chunk_count": chunk_count,
                    "agent_type": type(agent).__name__,
                },
            )

            logger.error(
                f"Agent streaming chat failed: {stream_error}",
                extra={
                    "user_input_length": len(user_input),
                    "chunk_count": chunk_count,
                },
            )

            # Yield error message as fallback
            error_message = (
                f"Chat processing encountered an error: {str(stream_error)[:100]}..."
            )
            yield error_message

        # Note: Multimodal handling depends on LLM backend:
        # - Gemma: Native multimodal support
        # - Nemotron: Text-only processing with feature extraction

    except Exception as e:
        log_error_with_context(
            e,
            "agent_chat",
            context={
                "user_input_length": len(user_input),
                "agent_type": type(agent).__name__ if agent else "None",
                "memory_token_limit": getattr(memory, "token_limit", None),
            },
        )

        # Re-raise structured error
        raise handle_agent_error(
            e,
            operation="agent_chat",
            user_input_length=len(user_input),
            agent_type=type(agent).__name__ if agent else "None",
        ) from e

    finally:
        # Log performance metrics
        duration = time.perf_counter() - start_time
        log_performance(
            "agent_chat",
            duration,
            user_input_length=len(user_input),
            chunk_count=chunk_count,
            agent_type=type(agent).__name__ if agent else "None",
        )

        logger.success(
            "Agent chat completed",
            extra={
                "user_input_length": len(user_input),
                "chunk_count": chunk_count,
                "duration_seconds": round(duration, 2),
            },
        )
