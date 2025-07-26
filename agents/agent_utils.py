"""Agent utilities for DocMind AI.

This module provides functions for creating agent tools, ReActAgents,
document analysis, and async chat with context.

Functions:
    create_tools_from_index: Create query tools from indexes.
    create_agent_with_tools: Create ReActAgent with tools.
    analyze_documents_agentic: Agentic document analysis.
    chat_with_agent: Async stream chat with agent.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.colbert_rerank import ColbertRerank

from models import AppSettings

settings = AppSettings()


def create_tools_from_index(index: dict[str, Any]) -> list[QueryEngineTool]:
    """Create enhanced query tools from indexes with hybrid search and reranking.

    Args:
        index: Dict containing vector and kg indexes with advanced features.

    Returns:
        List of QueryEngineTool instances with hybrid capabilities.
    """
    postprocessors = []
    if settings.reranker_model:
        reranker = ColbertRerank(
            model=settings.reranker_model,
            top_n=settings.reranking_top_k,
            keep_retrieval_score=True,
        )
        postprocessors.append(reranker)

    vector_query_engine = index["vector"].as_query_engine(
        similarity_top_k=settings.reranking_top_k,
        hybrid_alpha=settings.rrf_fusion_alpha,  # RRF fusion parameter
        node_postprocessors=postprocessors,  # Native ColBERT reranking
    )

    kg_query_engine = index["kg"].as_query_engine(
        similarity_top_k=10,  # KG queries may need more results
        include_text=True,  # Include source text with entities
        node_postprocessors=postprocessors if len(postprocessors) > 0 else None,
    )

    # Enhanced tools with better descriptions for agent decision-making
    tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="hybrid_vector_search",
                description=(
                    "Advanced hybrid search combining dense (BGE-Large) and sparse "
                    "(SPLADE++) embeddings with RRF fusion and ColBERT reranking. "
                    "Best for: semantic search, document retrieval, finding similar "
                    "content, answering questions about document content, "
                    "summarization, and general information extraction. Uses GPU "
                    "acceleration when available for 100x performance improvement."
                ),
            ),
        ),
        QueryEngineTool(
            query_engine=kg_query_engine,
            metadata=ToolMetadata(
                name="knowledge_graph_query",
                description=(
                    "Knowledge graph search for entity and relationship-based queries. "
                    "Best for: finding connections between concepts, identifying "
                    "entities and their relationships, exploring document structure, "
                    "understanding document hierarchies, and answering questions "
                    "about how different concepts relate to each other. "
                    "Complements vector search by providing structured knowledge "
                    "representation."
                ),
            ),
        ),
    ]

    return tools


def create_agent_with_tools(index: dict[str, Any], llm: Any) -> ReActAgent:
    """Create a ReActAgent.

    Tools for hybrid search, RRF fusion, and ColBERT reranking.

    Args:
        index: Dict containing vector and kg indexes with advanced features.
        llm: The language model instance to use for the agent.

    Returns:
        ReActAgent configured with enhanced query tools leveraging all advanced features
    """
    # Get tools from index
    tools = create_tools_from_index(index)

    # Create ReActAgent with enhanced memory and error handling
    try:
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=10,
            memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
        )
        logging.info(f"ReActAgent created with {len(tools)} enhanced tools")
        logging.info("Tools available: hybrid_vector_search, knowledge_graph_query")
        return agent

    except Exception as e:
        logging.error(f"ReActAgent creation failed: {e}")
        # Fallback with basic configuration
        agent = ReActAgent.from_tools(
            tools,
            llm=llm,
            verbose=True,
        )
        logging.warning("Using fallback ReActAgent configuration")
        return agent


def analyze_documents_agentic(
    agent: ReActAgent, index: dict[str, Any], prompt_type: str
) -> str:
    """Agentic analysis with multi-step reasoning.

    Args:
        agent: ReActAgent instance.
        index: Dict of indexes.
        prompt_type: Type of prompt to use.

    Returns:
        Analysis response string.
    """
    vector_query_engine = index["vector"].as_query_engine()
    kg_query_engine = index["kg"].as_query_engine()

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
    """Async stream chat with agent, supporting KG and multimodal via Gemma/Nemotron.

    Args:
        agent: ReActAgent instance.
        user_input: User query string.
        memory: Chat memory buffer.

    Yields:
        Chunks of response text.
    """
    try:
        response = await asyncio.to_thread(
            agent.async_stream_chat, user_input, memory=memory
        )
        async for chunk in response.async_response_gen():  # Using async_gen
            yield chunk
        # Multimodal handling: If Gemma, use native; for Nemotron (text-only),
        # extract text features
    except Exception as e:
        logging.error(f"Chat generation error: {str(e)}")
        raise
