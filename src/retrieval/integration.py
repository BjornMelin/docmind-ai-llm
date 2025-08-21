"""Integration layer for FEAT-002 retrieval system with app.py compatibility.

This module provides compatibility functions to replace the deprecated
embedding utilities while using the new BGE-M3 unified architecture.
"""

import asyncio
from typing import Any

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from loguru import logger

from .embeddings.bge_m3_manager import create_bgem3_embedding

# ADR-018 and ADR-019 compliance imports
from .postprocessor.cross_encoder_rerank import create_bge_cross_encoder_reranker
from .query_engine.router_engine import create_adaptive_router_engine
from .vector_store.qdrant_unified import create_unified_qdrant_store


async def create_index_async(
    documents: list[Any],
    use_gpu: bool = True,
    collection_name: str = "docmind_feat002_unified",
    qdrant_url: str = "http://localhost:6333",
) -> VectorStoreIndex:
    """Create unified index using BGE-M3 embeddings (FEAT-002 replacement).

    Compatibility function that replaces the deprecated create_index_async
    with the new BGE-M3 unified architecture.

    Args:
        documents: List of documents to index
        use_gpu: Enable GPU acceleration (RTX 4090 optimization)
        collection_name: Qdrant collection name
        qdrant_url: Qdrant server URL

    Returns:
        VectorStoreIndex using BGE-M3 unified embeddings
    """
    try:
        logger.info("Creating unified index with BGE-M3 embeddings (FEAT-002)")

        # Configure device based on GPU availability
        device = "cuda" if use_gpu else "cpu"

        # Initialize BGE-M3 embedding model
        embedding_model = create_bgem3_embedding(
            use_fp16=use_gpu,
            device=device,
            max_length=8192,
        )

        # Configure global settings
        Settings.embed_model = embedding_model
        Settings.chunk_size = 512
        Settings.chunk_overlap = 50

        # Initialize unified vector store
        vector_store = create_unified_qdrant_store(
            url=qdrant_url,
            collection_name=collection_name,
            embedding_dim=1024,  # BGE-M3 dimension
            rrf_alpha=0.7,  # 70% dense, 30% sparse
        )

        # Create ingestion pipeline with BGE-M3
        pipeline = IngestionPipeline(
            transformations=[
                SentenceSplitter(
                    chunk_size=512,
                    chunk_overlap=50,
                    separator=" ",
                ),
                embedding_model,
            ]
        )

        # Process documents to nodes
        logger.info(f"Processing {len(documents)} documents with BGE-M3 pipeline")
        nodes = await asyncio.to_thread(pipeline.run, documents=documents)

        # Generate unified embeddings for nodes
        texts = [node.get_content() for node in nodes]
        embeddings_result = await asyncio.to_thread(
            embedding_model.get_unified_embeddings,
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert=True,
        )

        # Add nodes to unified vector store
        node_ids = await asyncio.to_thread(
            vector_store.add,
            nodes=nodes,
            dense_embeddings=embeddings_result.get("dense", []),
            sparse_embeddings=embeddings_result.get("sparse", []),
            colbert_embeddings=embeddings_result.get("colbert", []),
        )

        # Create vector index from unified store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embedding_model,
        )

        logger.info(f"Created unified index with {len(node_ids)} nodes using BGE-M3")
        return index

    except Exception as e:
        logger.error(f"Failed to create unified index: {e}")
        raise


def create_hybrid_retriever_compat(
    index: VectorStoreIndex,
    use_reranking: bool = True,
    similarity_top_k: int = 10,
) -> Any:
    """Create hybrid retriever using FEAT-002 RouterQueryEngine.

    Compatibility function that replaces the deprecated create_hybrid_retriever
    with the new AdaptiveRouterQueryEngine.

    Args:
        index: VectorStoreIndex to create retriever from
        use_reranking: Enable BGE-reranker-v2-m3 reranking
        similarity_top_k: Number of top results to retrieve

    Returns:
        AdaptiveRouterQueryEngine configured for hybrid retrieval
    """
    try:
        logger.info("Creating adaptive router engine (FEAT-002 replacement)")

        # Create reranker if requested
        reranker = None
        if use_reranking:
            reranker = create_bge_cross_encoder_reranker(
                top_n=similarity_top_k,
                use_fp16=True,
                device="cuda",
            )

        # Create adaptive router engine
        router_engine = create_adaptive_router_engine(
            vector_index=index,
            kg_index=None,  # Optional knowledge graph
            hybrid_retriever=None,  # Router provides strategy selection
            reranker=reranker,
        )

        logger.info("Adaptive router engine created successfully")
        return router_engine

    except Exception as e:
        logger.error(f"Failed to create adaptive router engine: {e}")
        raise


# Backward compatibility aliases
create_hybrid_retriever = create_hybrid_retriever_compat


# ADR-018 and ADR-019 experimental features
def create_experimental_components(
    nodes: list[Any] | None = None,
    enable_graphrag: bool | None = None,
    enable_dspy: bool | None = None,
) -> dict[str, Any]:
    """Create experimental components for ADR compliance.

    Creates PropertyGraphIndex (ADR-019) and DSPy optimizer (ADR-018)
    components with proper feature flag handling.

    Args:
        nodes: Document nodes for graph construction
        enable_graphrag: Override for GraphRAG feature flag
        enable_dspy: Override for DSPy feature flag

    Returns:
        Dictionary containing experimental components
    """
    components = {}

    try:
        # Import here to avoid circular imports and linter reordering
        from .graph.property_graph import create_property_graph_index
        from .optimization.dspy_optimizer import create_dspy_optimizer

        # PropertyGraphIndex (ADR-019)
        property_graph = create_property_graph_index(
            nodes=nodes,
            enable_experimental=enable_graphrag,
        )
        components["property_graph"] = property_graph

        # DSPy Query Optimizer (ADR-018)
        dspy_optimizer = create_dspy_optimizer(
            enable_experimental=enable_dspy,
        )
        components["dspy_optimizer"] = dspy_optimizer

        logger.info("Experimental components created for ADR compliance")

    except Exception as e:
        logger.warning(f"Failed to create experimental components: {e}")
        # Return empty components if creation fails
        components = {
            "property_graph": None,
            "dspy_optimizer": None,
        }

    return components


def is_experimental_features_enabled() -> dict[str, bool]:
    """Check which experimental features are enabled.

    Returns:
        Dictionary of feature flags for experimental components
    """
    try:
        from .graph.property_graph import is_property_graph_enabled
        from .optimization.dspy_optimizer import is_dspy_optimization_enabled

        return {
            "graphrag": is_property_graph_enabled(),
            "dspy": is_dspy_optimization_enabled(),
        }
    except ImportError:
        return {
            "graphrag": False,
            "dspy": False,
        }
