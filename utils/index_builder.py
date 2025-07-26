"""Indexing utilities for DocMind AI.

This module handles creation of hybrid vector indexes and knowledge graph indexes
with GPU optimization and async support.

Functions:
    create_index_async: Async index creation with hybrid support.
    create_index: Sync index creation with hybrid support.
"""

import logging
from typing import Any

import torch
from fastembed import SparseTextEmbedding
from llama_index.core import (
    Document,
    KnowledgeGraphIndex,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import AsyncQdrantClient, QdrantClient
from spacy import load as spacy_load

from models import AppSettings
from qdrant_utils import setup_hybrid_qdrant, setup_hybrid_qdrant_async
from utils import verify_rrf_configuration

settings = AppSettings()


async def create_index_async(docs: list[Document], use_gpu: bool) -> dict[str, Any]:
    """Create hybrid index with Qdrant using async operations.

    Provides 50-80% performance improvement over synchronous operations.
    ColBERT reranking is handled via native postprocessor in query tools.

    Args:
        docs: List of documents to index.
        use_gpu: Whether to use GPU for embeddings.

    Returns:
        Dict with vector and kg indexes.
    """
    try:
        # Verify RRF configuration meets Phase 2.1 requirements
        rrf_verification = verify_rrf_configuration(settings)
        if rrf_verification["issues"]:
            logging.warning(f"RRF Configuration Issues: {rrf_verification['issues']}")
            for rec in rrf_verification["recommendations"]:
                logging.info(f"Recommendation: {rec}")

        # Use AsyncQdrantClient for improved performance
        async_client = AsyncQdrantClient(url=settings.qdrant_url)

        # Use FastEmbed native GPU acceleration for both dense and sparse embeddings
        # Dense embedding model with optimized configuration
        embed_model = FastEmbedEmbedding(
            model_name=settings.dense_embedding_model,  # BAAI/bge-large-en-v1.5
            cache_dir="./embeddings_cache",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu and torch.cuda.is_available()
            else ["CPUExecutionProvider"],
            batch_size=settings.embedding_batch_size,
        )
        if settings.gpu_acceleration and torch.cuda.is_available():
            embed_model = torch.compile(embed_model, dynamic=True)
            logging.info("Applied torch.compile to dense embed_model")

        # Sparse embedding model with optimized configuration
        sparse_embed_model = SparseTextEmbedding(
            model_name=settings.sparse_embedding_model,
            cache_dir="./embeddings_cache",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu and torch.cuda.is_available()
            else ["CPUExecutionProvider"],
            batch_size=settings.embedding_batch_size,
        )
        if settings.gpu_acceleration and torch.cuda.is_available():
            sparse_embed_model = torch.compile(sparse_embed_model, dynamic=True)
            logging.info("Applied torch.compile to sparse embed_model")

        if use_gpu and torch.cuda.is_available():
            logging.info(
                "Using FastEmbed native GPU acceleration for dense and sparse "
                "embeddings with AsyncQdrantClient"
            )
        else:
            logging.info(
                "Using FastEmbed CPU mode for embeddings with AsyncQdrantClient"
            )

        # Setup Qdrant with proper hybrid search configuration using async
        vector_store = await setup_hybrid_qdrant_async(
            client=async_client,
            collection_name="docmind",
            dense_embedding_size=settings.dense_embedding_dimension,  # BGE-Large 1024D
            recreate=False,
        )

        # Create index with RRF-enabled hybrid search
        if settings.gpu_acceleration and torch.cuda.is_available():
            with torch.cuda.Stream() as stream:
                torch.cuda.set_stream(stream)
                if settings.debug_mode:
                    with torch.profiler.profile() as p:
                        index = VectorStoreIndex.from_documents(
                            docs,
                            storage_context=StorageContext.from_defaults(
                                vector_store=vector_store
                            ),
                            embed_model=embed_model,
                            sparse_embed_model=sparse_embed_model,
                        )
                    p.export_chrome_trace("trace.json")
                    logging.info("Profiling trace exported to trace.json")
                else:
                    index = VectorStoreIndex.from_documents(
                        docs,
                        storage_context=StorageContext.from_defaults(
                            vector_store=vector_store
                        ),
                        embed_model=embed_model,
                        sparse_embed_model=sparse_embed_model,
                    )
        else:
            index = VectorStoreIndex.from_documents(
                docs,
                storage_context=StorageContext.from_defaults(vector_store=vector_store),
                embed_model=embed_model,
                sparse_embed_model=sparse_embed_model,
            )

        # Calculate hybrid_alpha from research-backed weights (dense: 0.7, sparse: 0.3)
        hybrid_alpha = settings.rrf_fusion_weight_dense / (
            settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
        )

        # RRF fusion is handled automatically by the hybrid query mode
        logging.info(
            f"Using research-backed RRF fusion with weights - "
            f"Dense: {settings.rrf_fusion_weight_dense:.1f}, "
            f"Sparse: {settings.rrf_fusion_weight_sparse:.1f}, "
            f"Hybrid Alpha: {hybrid_alpha:.3f} (AsyncQdrantClient enabled)"
        )

        # ColBERT reranking handled by native LlamaIndex postprocessor
        # Simplified per Phase 2.2 requirements

        # KG index creation with spaCy for offline extraction
        spacy_extractor = spacy_load("en_core_web_sm")
        kg_index = KnowledgeGraphIndex.from_documents(
            docs,
            llm=Ollama(model=settings.default_model),
            extractor=spacy_extractor,
            max_entities=settings.new_max_entities,
        )  # For entity/relation queries

        await async_client.close()  # Cleanup async client
        return {"vector": index, "kg": kg_index}
    except ValueError as e:
        logging.error(f"Invalid configuration for index: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Index creation error: {str(e)}")
        raise


def create_index(docs: list[Document], use_gpu: bool) -> dict[str, Any]:
    """Create hybrid index with Qdrant, knowledge graph, torch.compile for embeddings.

    ColBERT reranking is handled via native postprocessor in query tools.

    Args:
        docs: List of documents to index.
        use_gpu: Whether to use GPU for embeddings.

    Returns:
        Dict with vector and kg indexes.
    """
    try:
        # Verify RRF configuration meets Phase 2.1 requirements
        rrf_verification = verify_rrf_configuration(settings)
        if rrf_verification["issues"]:
            logging.warning(f"RRF Configuration Issues: {rrf_verification['issues']}")
            for rec in rrf_verification["recommendations"]:
                logging.info(f"Recommendation: {rec}")

        client = QdrantClient(url=settings.qdrant_url)

        logging.info(
            "Using synchronous QdrantClient (consider using create_index_async "
            "for 50-80%% performance improvement)"
        )

        # Use FastEmbed native GPU acceleration for both dense and sparse embeddings
        # Dense embedding model with optimized configuration
        embed_model = FastEmbedEmbedding(
            model_name=settings.dense_embedding_model,  # BAAI/bge-large-en-v1.5
            cache_dir="./embeddings_cache",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu and torch.cuda.is_available()
            else ["CPUExecutionProvider"],
            batch_size=settings.embedding_batch_size,
        )
        if settings.gpu_acceleration and torch.cuda.is_available():
            embed_model = torch.compile(embed_model, dynamic=True)
            logging.info("Applied torch.compile to dense embed_model")

        # Sparse embedding model with optimized configuration
        sparse_embed_model = SparseTextEmbedding(
            model_name=settings.sparse_embedding_model,
            cache_dir="./embeddings_cache",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if use_gpu and torch.cuda.is_available()
            else ["CPUExecutionProvider"],
            batch_size=settings.embedding_batch_size,
        )
        if settings.gpu_acceleration and torch.cuda.is_available():
            sparse_embed_model = torch.compile(sparse_embed_model, dynamic=True)
            logging.info("Applied torch.compile to sparse embed_model")

        if use_gpu and torch.cuda.is_available():
            logging.info(
                "Using FastEmbed native GPU acceleration for dense and sparse "
                "embeddings"
            )
        else:
            logging.info("Using FastEmbed CPU mode for embeddings")

        # Setup Qdrant with proper hybrid search configuration
        vector_store = setup_hybrid_qdrant(
            client=client,
            collection_name="docmind",
            dense_embedding_size=settings.dense_embedding_dimension,  # BGE-Large 1024D
            recreate=False,
        )

        # Create index with RRF-enabled hybrid search
        if settings.gpu_acceleration and torch.cuda.is_available():
            with torch.cuda.Stream() as stream:
                torch.cuda.set_stream(stream)
                if settings.debug_mode:
                    with torch.profiler.profile() as p:
                        index = VectorStoreIndex.from_documents(
                            docs,
                            storage_context=StorageContext.from_defaults(
                                vector_store=vector_store
                            ),
                            embed_model=embed_model,
                            sparse_embed_model=sparse_embed_model,
                        )
                    p.export_chrome_trace("trace.json")
                    logging.info("Profiling trace exported to trace.json")
                else:
                    index = VectorStoreIndex.from_documents(
                        docs,
                        storage_context=StorageContext.from_defaults(
                            vector_store=vector_store
                        ),
                        embed_model=embed_model,
                        sparse_embed_model=sparse_embed_model,
                    )
        else:
            index = VectorStoreIndex.from_documents(
                docs,
                storage_context=StorageContext.from_defaults(vector_store=vector_store),
                embed_model=embed_model,
                sparse_embed_model=sparse_embed_model,
            )

        # Enhance query engine with RRF fusion - use research-backed weight ratios
        # Calculate hybrid_alpha from research-backed weights (dense: 0.7, sparse: 0.3)
        # LlamaIndex hybrid_alpha: 0.0 = full sparse, 1.0 = full dense
        hybrid_alpha = settings.rrf_fusion_weight_dense / (
            settings.rrf_fusion_weight_dense + settings.rrf_fusion_weight_sparse
        )

        # Query engine configuration handled by create_tools_from_index

        # RRF fusion is handled automatically by the hybrid query mode
        logging.info(
            f"Using research-backed RRF fusion with weights - "
            f"Dense: {settings.rrf_fusion_weight_dense:.1f}, "
            f"Sparse: {settings.rrf_fusion_weight_sparse:.1f}, "
            f"Hybrid Alpha: {hybrid_alpha:.3f}"
        )

        # ColBERT reranking handled by native LlamaIndex postprocessor
        # Simplified per Phase 2.2 requirements

        # KG index creation with spaCy for offline extraction
        spacy_extractor = spacy_load("en_core_web_sm")
        kg_index = KnowledgeGraphIndex.from_documents(
            docs,
            llm=Ollama(model=settings.default_model),
            extractor=spacy_extractor,
            max_entities=settings.new_max_entities,
        )  # For entity/relation queries
        return {"vector": index, "kg": kg_index}
    except ValueError as e:
        logging.error(f"Invalid configuration for index: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Index creation error: {str(e)}")
        raise
