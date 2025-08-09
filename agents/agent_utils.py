"""Advanced agent utilities for DocMind AI with hybrid search integration.

This module provides comprehensive agent functionality including:
- ReActAgent creation with advanced tool integration
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
import logging
from collections.abc import AsyncGenerator
from typing import Any

from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from models import AppSettings

settings = AppSettings()


def create_tools_from_index(index_data: dict[str, Any]) -> list[QueryEngineTool]:
    """Create enhanced query tools with hybrid search and reranking capabilities.

    Constructs QueryEngineTool instances from provided indexes, integrating
    advanced search features including RRF fusion, ColBERT reranking, and
    multi-modal query capabilities. Provides intelligent tool selection
    based on available index components.

    Features:
    - Hybrid fusion retriever with RRF (when available)
    - Fallback to hybrid vector search
    - Knowledge graph queries for entity relationships
    - ColBERT reranking for improved relevance
    - GPU acceleration integration
    - Comprehensive error handling and fallbacks

    Args:
        index_data: Dictionary containing indexed components:
            - 'vector' (VectorStoreIndex): Hybrid vector index
            - 'kg' (KnowledgeGraphIndex | None): Knowledge graph index
            - 'retriever' (QueryFusionRetriever | None): Hybrid fusion retriever

    Returns:
        List of QueryEngineTool instances configured with:
        - Hybrid fusion search tool (if retriever available)
        - Hybrid vector search tool (fallback)
        - Knowledge graph query tool (if KG index available)
        - ColBERT reranking postprocessors

    Note:
        Tools are created in order of preference: hybrid fusion > hybrid vector > KG.
        ColBERT reranker is configured based on settings.reranker_model.
        Each tool includes detailed metadata for agent decision-making.

    Example:
        >>> index_data = {'vector': vector_idx, 'kg': kg_idx, 'retriever': retriever}
        >>> tools = create_tools_from_index(index_data)
        >>> print(f"Created {len(tools)} query tools")
        >>> for tool in tools:
        ...     print(f"Tool: {tool.metadata.name}")
    """
    tools = []

    # Setup ColBERT reranker if configured
    postprocessors = []
    if settings.reranker_model:
        reranker = ColbertRerank(
            model=settings.reranker_model,
            top_n=settings.reranking_top_k,
            keep_retrieval_score=True,
        )
        postprocessors.append(reranker)

    # Check if hybrid retriever is available
    if "retriever" in index_data and index_data["retriever"] is not None:
        # Create query engine with hybrid fusion retriever and reranking
        hybrid_query_engine = RetrieverQueryEngine(
            retriever=index_data["retriever"],
            node_postprocessors=postprocessors,
        )

        tools.append(
            QueryEngineTool(
                query_engine=hybrid_query_engine,
                metadata=ToolMetadata(
                    name="hybrid_fusion_search",
                    description=(
                        "Advanced hybrid search with QueryFusionRetriever using RRF "
                        "(Reciprocal Rank Fusion) to combine dense (BGE-Large) and "
                        "sparse (SPLADE++) embeddings with ColBERT reranking. "
                        "Best for: comprehensive document retrieval, finding relevant "
                        "content through semantic similarity and keyword matching, "
                        "complex queries requiring both context understanding and "
                        "precise term matching. Provides superior relevance through "
                        "RRF score fusion and ColBERT late-interaction reranking."
                    ),
                ),
            ),
        )
        logging.info("Hybrid fusion search tool added with ColBERT reranking")
    else:
        # Fallback to standard vector query engine
        vector_query_engine = index_data["vector"].as_query_engine(
            similarity_top_k=settings.reranking_top_k,
            hybrid_alpha=settings.rrf_fusion_alpha,  # RRF fusion parameter
            node_postprocessors=postprocessors,  # Native ColBERT reranking
        )

        # Add fallback vector search tool if hybrid retriever not available
        tools.append(
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="hybrid_vector_search",
                    description=(
                        "Advanced hybrid search combining dense (BGE-Large) and sparse "
                        "(SPLADE++) embeddings with RRF fusion and ColBERT reranking. "
                        "Best for: semantic search, document retrieval, finding "
                        "similar content, answering questions about document content, "
                        "summarization, and general information extraction. Uses GPU "
                        "acceleration when available for 100x performance improvement."
                    ),
                ),
            ),
        )
        logging.info("Fallback hybrid vector search tool added")

    # Add Knowledge Graph tool if available
    if index_data.get("kg") is not None:
        kg_query_engine = index_data["kg"].as_query_engine(
            similarity_top_k=10,  # KG queries may need more results
            include_text=True,  # Include source text with entities
            node_postprocessors=postprocessors if len(postprocessors) > 0 else None,
        )

        tools.append(
            QueryEngineTool(
                query_engine=kg_query_engine,
                metadata=ToolMetadata(
                    name="knowledge_graph_query",
                    description=(
                        "Knowledge graph search for entity and relationship-based "
                        "queries. "
                        "Best for: finding connections between concepts, identifying "
                        "entities and their relationships, exploring document "
                        "structure, "
                        "understanding document hierarchies, and answering questions "
                        "about how different concepts relate to each other. "
                        "Complements vector search by providing structured knowledge "
                        "representation."
                    ),
                ),
            ),
        )
        logging.info("Knowledge Graph query tool added successfully")
    else:
        logging.warning(
            "Knowledge Graph index not available - only vector search will be used"
        )

    return tools


def create_agent_with_tools(index_data: dict[str, Any], llm: Any) -> ReActAgent:
    """Create ReActAgent with advanced hybrid search capabilities.

    Constructs a fully-featured ReActAgent equipped with hybrid search tools,
    knowledge graph queries, and memory management. Integrates all available
    advanced features including RRF fusion, ColBERT reranking, and GPU
    acceleration for optimal performance.

    Agent capabilities:
    - Multi-step reasoning with ReAct pattern
    - Hybrid vector search with dense/sparse embeddings
    - Knowledge graph entity relationship queries
    - Memory-aware conversations with token management
    - Error handling and graceful degradation
    - Verbose logging for debugging and monitoring

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
        Exception: If agent creation fails, falls back to basic configuration.

    Note:
        The agent uses ReAct (Reasoning + Acting) pattern for systematic
        problem-solving. Memory buffer prevents context overflow in long
        conversations. Tool selection is automatic based on query analysis.

    Example:
        >>> from llama_index.llms.ollama import Ollama
        >>> llm = Ollama(model="llama2")
        >>> agent = create_agent_with_tools(index_data, llm)
        >>> response = agent.chat("What are the main themes in the documents?")
        >>> print(response.response)
    """
    # Get tools from index data
    tools = create_tools_from_index(index_data)

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
        logging.info("ReActAgent created with %s enhanced tools", len(tools))
        logging.info("Tools available: %s", ", ".join(tool_names))
        return agent

    except Exception as e:
        logging.error("ReActAgent creation failed: %s", e)
        # Fallback with basic configuration
        agent = ReActAgent.from_tools(
            tools,
            llm=llm,
            verbose=True,
        )
        tool_names = [tool.metadata.name for tool in tools] if tools else []
        logging.warning(
            "Using fallback ReActAgent configuration with tools: %s",
            ", ".join(tool_names),
        )
        return agent


def analyze_documents_agentic(
    agent: ReActAgent, index_data: dict[str, Any], prompt_type: str
) -> str:
    """Perform agentic document analysis with multi-step reasoning.

    Executes document analysis using the ReActAgent's multi-step reasoning
    capabilities. The agent systematically queries different index types
    and synthesizes information to provide comprehensive analysis.

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

    Note:
        This function provides a compatibility layer and fallback mechanism.
        In normal operation, agents should be properly initialized via
        create_agent_with_tools in the main application.

    Example:
        >>> agent = create_agent_with_tools(index_data, llm)
        >>> analysis = analyze_documents_agentic(agent, index_data, "summary")
        >>> print(analysis)
    """
    vector_query_engine = index_data["vector"].as_query_engine()
    kg_query_engine = index_data["kg"].as_query_engine()

    tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_query",
                description="Query documents using vector similarity search "
                "for general content retrieval",
            ),
        ),
        QueryEngineTool(
            query_engine=kg_query_engine,
            metadata=ToolMetadata(
                name="knowledge_graph_query",
                description="Query documents using knowledge graph "
                "for entity and relationship-based queries",
            ),
        ),
    ]
    if not agent:
        # Agent will be properly initialized in app.py with user-selected LLM
        # This is a fallback that should not be reached in normal operation
        agent = ReActAgent.from_tools(
            tools,
            llm=Ollama(model=settings.default_model),
            verbose=True,
        )
    response = agent.chat(f"Analyze with prompt: {prompt_type}")
    return response.response  # Parse to AnalysisOutput


async def chat_with_agent(
    agent: ReActAgent, user_input: str, memory: ChatMemoryBuffer
) -> AsyncGenerator[str, None]:
    """Stream chat responses from ReActAgent asynchronously.

    Provides asynchronous streaming chat interface with the ReActAgent,
    supporting knowledge graph queries, multimodal processing, and memory
    management. Yields response chunks as they become available for
    real-time user interaction.

    Features:
    - Asynchronous streaming response generation
    - Memory-aware conversation context
    - Multi-step reasoning with intermediate results
    - Support for multimodal content processing
    - Error handling with meaningful error messages
    - Compatible with various LLM backends

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
        Exception: If chat generation fails, logs error and re-raises.

    Note:
        The function uses asyncio.to_thread for compatibility with synchronous
        agent implementations. Multimodal handling depends on the underlying
        LLM capabilities (e.g., Gemma for native multimodal, Nemotron for text).

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
    try:
        response = await asyncio.to_thread(
            agent.async_stream_chat, user_input, memory=memory
        )
        async for chunk in response.async_response_gen():  # Using async_gen
            yield chunk
        # Note: Multimodal handling depends on LLM backend:
        # - Gemma: Native multimodal support
        # - Nemotron: Text-only processing with feature extraction
    except Exception as e:
        logging.error("Chat generation error: %s", str(e))
        raise
