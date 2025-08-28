"""Tool factory for DocMind AI agents with consistent configuration.

This module provides a centralized factory for creating agent tools with
optimal configuration, reranking, and hybrid search capabilities. Eliminates
code duplication across different tool creation patterns.

Features:
- Consistent tool metadata and configuration
- ColBERT reranking integration
- Hybrid search tool creation
- Vector and knowledge graph query tools
- Comprehensive error handling and fallbacks
- Detailed tool descriptions for agent decision-making

Example:
    Using the tool factory::

        from agents.tool_factory import ToolFactory

        # Create tools from index data
        tools = ToolFactory.create_tools_from_indexes(
            vector_index=index_data['vector'],
            kg_index=index_data['kg'],
            retriever=index_data['retriever']
        )

        # Create single tool
        search_tool = ToolFactory.create_vector_search_tool(vector_index)

Attributes:
    settings (DocMindSettings): Global application settings for tool configuration.
"""

from typing import Any

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from loguru import logger

from src.config import settings

# Constants

KG_SIMILARITY_TOP_K = 10

# Tool configuration constants
DEFAULT_RERANKING_TOP_K = settings.retrieval.reranking_top_k
DEFAULT_VECTOR_SIMILARITY_TOP_K = settings.retrieval.top_k


class ToolFactory:
    """Factory for creating agent tools with consistent configuration.

    Provides centralized tool creation with optimized settings, reranking
    integration, and consistent metadata across different tool types.
    Eliminates code duplication and ensures all tools use best practices.
    """

    @staticmethod
    def create_query_tool(
        query_engine: Any, name: str, description: str
    ) -> QueryEngineTool:
        """Create a query engine tool with standard configuration.

        Creates a QueryEngineTool with consistent metadata and configuration
        for use in agent workflows. Provides standardized tool interface.

        Args:
            query_engine: Query engine instance to wrap as a tool.
            name: Tool name for agent identification and selection.
            description: Detailed description for agent decision-making.

        Returns:
            QueryEngineTool: Configured tool ready for agent use.

        Example:
            >>> engine = index.as_query_engine()
            >>> tool = ToolFactory.create_query_tool(
            ...     engine, "search", "Search documents"
            ... )
        """
        return QueryEngineTool(
            query_engine=query_engine,
            metadata=ToolMetadata(
                name=name,
                description=description,
                return_direct=False,  # Allow agent reasoning over results
            ),
        )

    @classmethod
    def _create_reranker(cls) -> ColbertRerank | None:
        """Create ColBERT reranker if configured.

        Creates a ColbertRerank postprocessor based on settings configuration.
        Returns None if no reranker model is specified.

        Returns:
            ColbertRerank or None: Configured reranker or None if disabled.
        """
        if not settings.retrieval.reranker_model:
            return None

        try:
            reranker = ColbertRerank(
                top_n=settings.retrieval.reranking_top_k,
                model=settings.retrieval.reranker_model,
                keep_retrieval_score=True,
            )
            logger.info(
                "ColBERT reranker created: %s", settings.retrieval.reranker_model
            )
            return reranker
        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            logger.warning("Failed to create ColBERT reranker: %s", e)
            return None

    @classmethod
    def create_vector_search_tool(cls, index: Any) -> QueryEngineTool:
        """Create vector search tool with reranking.

        Creates a vector search tool with optimal configuration including
        similarity search, hybrid search when available, and ColBERT reranking.

        Args:
            index: VectorStoreIndex for semantic similarity search.

        Returns:
            QueryEngineTool: Configured vector search tool.

        Example:
            >>> tool = ToolFactory.create_vector_search_tool(vector_index)
            >>> response = tool.call("Find similar documents")
        """
        # Create reranker if configured
        reranker = cls._create_reranker()
        postprocessors = [reranker] if reranker else []

        # Configure query engine with optimal settings
        query_engine = index.as_query_engine(
            similarity_top_k=settings.retrieval.top_k,
            node_postprocessors=postprocessors,
            verbose=False,
        )

        return cls.create_query_tool(
            query_engine,
            "vector_search",
            (
                "Semantic similarity search using dense embeddings (BGE-Large). "
                "Best for: finding conceptually similar content, answering "
                "questions about document meaning, summarization, and general "
                "information retrieval. Uses GPU acceleration when available "
                "for improved performance."
            ),
        )

    @classmethod
    def create_kg_search_tool(cls, kg_index: Any) -> QueryEngineTool | None:
        """Create knowledge graph search tool.

        Creates a knowledge graph query tool for entity and relationship
        searches. Returns None if no knowledge graph index is provided.

        Args:
            kg_index: KnowledgeGraphIndex for entity relationship queries.

        Returns:
            QueryEngineTool or None: Configured KG tool or None if no index.

        Example:
            >>> tool = ToolFactory.create_kg_search_tool(kg_index)
            >>> if tool:
            ...     response = tool.call("Find entities related to AI")
        """
        if not kg_index:
            return None

        # Create reranker if configured
        reranker = cls._create_reranker()
        postprocessors = [reranker] if reranker else []

        query_engine = kg_index.as_query_engine(
            similarity_top_k=KG_SIMILARITY_TOP_K,  # KG queries may need more results
            include_text=True,  # Include source text with entities
            node_postprocessors=postprocessors,
            verbose=False,
        )

        return cls.create_query_tool(
            query_engine,
            "knowledge_graph",
            (
                "Knowledge graph search for entity and relationship-based queries. "
                "Best for: finding connections between concepts, identifying "
                "entities and their relationships, exploring document structure, "
                "understanding hierarchies, and answering questions about how "
                "different concepts relate to each other. Complements vector "
                "search with structured knowledge representation."
            ),
        )

    @classmethod
    def create_hybrid_search_tool(cls, retriever: Any) -> QueryEngineTool:
        """Create hybrid search tool with RRF fusion and reranking.

        Creates a hybrid search tool using QueryFusionRetriever
        with RRF (Reciprocal Rank Fusion) and ColBERT reranking for optimal
        search performance.

        Args:
            retriever: QueryFusionRetriever with RRF fusion capabilities.

        Returns:
            QueryEngineTool: Configured hybrid fusion search tool.

        Example:
            >>> tool = ToolFactory.create_hybrid_search_tool(fusion_retriever)
            >>> response = tool.call("Complex query requiring hybrid search")
        """
        # Create reranker if configured
        reranker = cls._create_reranker()
        postprocessors = [reranker] if reranker else []

        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=postprocessors,
        )

        return cls.create_query_tool(
            query_engine,
            "hybrid_fusion_search",
            (
                "Advanced hybrid search with QueryFusionRetriever using RRF "
                "(Reciprocal Rank Fusion) to combine dense (BGE-Large) and "
                "sparse (SPLADE++) embeddings with ColBERT reranking. "
                "Best for: comprehensive document retrieval, finding relevant "
                "content through semantic similarity and keyword matching, "
                "complex queries requiring both context understanding and "
                "precise term matching. Provides superior relevance through "
                "RRF score fusion and ColBERT late-interaction reranking."
            ),
        )

    @classmethod
    def create_hybrid_vector_tool(cls, index: Any) -> QueryEngineTool:
        """Create hybrid vector search tool as fallback.

        Creates a hybrid vector search tool using the index's built-in
        hybrid search capabilities when fusion retriever is not available.

        Args:
            index: VectorStoreIndex with hybrid search capabilities.

        Returns:
            QueryEngineTool: Configured hybrid vector search tool.

        Example:
            >>> tool = ToolFactory.create_hybrid_vector_tool(vector_index)
            >>> response = tool.call("Search using hybrid embeddings")
        """
        # Create reranker if configured
        reranker = cls._create_reranker()
        postprocessors = [reranker] if reranker else []

        query_engine = index.as_query_engine(
            similarity_top_k=settings.retrieval.top_k,
            node_postprocessors=postprocessors,
            verbose=False,
        )

        return cls.create_query_tool(
            query_engine,
            "hybrid_vector_search",
            (
                "Hybrid search combining dense (BGE-Large) and sparse "
                "(SPLADE++) embeddings with ColBERT reranking. "
                "Best for: semantic search, document retrieval, finding "
                "similar content, answering questions about document content, "
                "summarization, and general information extraction. Uses "
                "hybrid embeddings for both semantic and keyword matching."
            ),
        )

    @classmethod
    def create_tools_from_indexes(
        cls,
        vector_index: Any,
        kg_index: Any | None = None,
        retriever: Any | None = None,
    ) -> list[QueryEngineTool]:
        """Create all available tools from index components.

        Creates a comprehensive set of query tools based on available index
        components. Prioritizes features when available and provides
        fallbacks for missing components.

        Args:
            vector_index: VectorStoreIndex for vector search (required).
            kg_index: KnowledgeGraphIndex for entity queries (optional).
            retriever: QueryFusionRetriever for hybrid fusion (optional).

        Returns:
            List[QueryEngineTool]: All available tools in priority order.

        Note:
            Tools are created in order of sophistication:
            1. Hybrid fusion search (if retriever available)
            2. Hybrid vector search (fallback)
            3. Knowledge graph search (if KG index available)
            4. Basic vector search (always available)

        Example:
            >>> tools = ToolFactory.create_tools_from_indexes(
            ...     vector_index=vector_idx,
            ...     kg_index=kg_idx,
            ...     retriever=fusion_retriever
            ... )
            >>> print(f"Created {len(tools)} tools")
        """
        tools = []

        if not vector_index:
            logger.error("Vector index is required for tool creation")
            return tools

        # Add hybrid fusion search if retriever is available (highest priority)
        if retriever:
            tools.append(cls.create_hybrid_search_tool(retriever))
            logger.info("Added hybrid fusion search tool")
        else:
            # Fallback to hybrid vector search
            tools.append(cls.create_hybrid_vector_tool(vector_index))
            logger.info("Added hybrid vector search tool (fallback)")

        # Add knowledge graph search if available
        if kg_index:
            kg_tool = cls.create_kg_search_tool(kg_index)
            if kg_tool:
                tools.append(kg_tool)
                logger.info("Added knowledge graph search tool")
        else:
            logger.info("Knowledge graph index not available")

        # Add basic vector search as additional option
        tools.append(cls.create_vector_search_tool(vector_index))
        logger.info("Added vector search tool")

        logger.info("Created %d tools for agent", len(tools))
        return tools

    @classmethod
    def create_basic_tools(cls, index_data: dict[str, Any]) -> list[QueryEngineTool]:
        """Create basic tools from index data dictionary.

        Provides a convenient dictionary-based interface for tool creation.
        Extracts components and creates tools from the provided dictionary.

        Args:
            index_data: Dictionary containing indexed components:
                - 'vector' (VectorStoreIndex): Vector index
                - 'kg' (KnowledgeGraphIndex): Knowledge graph index
                - 'retriever' (QueryFusionRetriever): Fusion retriever

        Returns:
            List[QueryEngineTool]: Tools created from available components.

        Example:
            >>> tools = ToolFactory.create_basic_tools(index_data)
            >>> agent = ReActAgent.from_tools(tools, llm)
        """
        return cls.create_tools_from_indexes(
            vector_index=index_data.get("vector"),
            kg_index=index_data.get("kg"),
            retriever=index_data.get("retriever"),
        )
