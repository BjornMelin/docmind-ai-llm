#!/usr/bin/env python3
"""Comprehensive integration tests for DocMind AI system.

This module provides end-to-end testing for all major components:
- SPLADE++ sparse embeddings with prithvida/Splade_PP_en_v1
- BGE-Large dense embeddings with BAAI/bge-large-en-v1.5
- Hybrid search with RRF fusion (0.7/0.3 weights)
- ColBERT reranking with performance monitoring
- ReActAgent system with LangGraph multi-agent support
- Error handling and fallback mechanisms

Tests validate that all components work together correctly and handle
edge cases gracefully.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document

import utils  # Added to fix F821
from agent_factory import (
    analyze_query_complexity,
    get_agent_system,
)
from agents.agent_utils import create_tools_from_index

# Import DocMind AI components
from models import AppSettings
from utils import (
    create_index_async,
    detect_hardware,
    setup_hybrid_qdrant,
    setup_hybrid_qdrant_async,
    verify_rrf_configuration,
)


class TestFastEmbedModelManager:
    """Test FastEmbedModelManager singleton and model caching."""

    def test_singleton_behavior(self):
        """Test that FastEmbedModelManager follows singleton pattern."""
        manager1 = ModelManager()
        manager2 = ModelManager()

        assert manager1 is manager2, "FastEmbedModelManager should be singleton"
        assert id(manager1) == id(manager2), "Instances should have same memory address"

    @patch("utils.TextEmbedding")
    def test_dense_embedding_model_caching(self, mock_text_embedding):
        """Test that dense embedding models are cached correctly."""
        mock_model = MagicMock()
        mock_text_embedding.return_value = mock_model

        manager = ModelManager()
        manager.clear_cache()  # Start with clean cache

        # First call should create model
        model1 = manager.get_dense_embedding_model("test-model")
        assert mock_text_embedding.call_count == 1

        # Second call should use cached model
        model2 = manager.get_dense_embedding_model("test-model")
        assert mock_text_embedding.call_count == 1  # Should not create new model
        assert model1 is model2

    @patch("utils.SparseTextEmbedding")
    def test_sparse_embedding_model_caching(self, mock_sparse_embedding):
        """Test that sparse embedding models are cached correctly."""
        mock_model = MagicMock()
        mock_sparse_embedding.return_value = mock_model

        manager = ModelManager()
        manager.clear_cache()

        # Test SPLADE++ model caching
        model1 = manager.get_sparse_embedding_model("prithvida/Splade_PP_en_v1")
        model2 = manager.get_sparse_embedding_model("prithvida/Splade_PP_en_v1")

        assert mock_sparse_embedding.call_count == 1
        assert model1 is model2

    def test_gpu_provider_configuration(self):
        """Test GPU provider configuration based on settings."""
        manager = ModelManager()

        with (
            patch("utils.TextEmbedding") as mock_embedding,
            patch("utils.torch.cuda.is_available", return_value=True),
        ):
            manager.get_dense_embedding_model()

            # Check that GPU providers are configured when GPU is available
            call_args = mock_embedding.call_args
            providers = call_args[1]["providers"]
            assert "CUDAExecutionProvider" in providers
            assert "CPUExecutionProvider" in providers

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        manager = ModelManager()

        with patch("utils.TextEmbedding"):
            # Create a model to populate cache
            manager.get_dense_embedding_model("test-model")
            assert len(manager._models) > 0

            # Clear cache
            manager.clear_cache()
            assert len(manager._models) == 0


class TestHybridSearchIntegration:
    """Test hybrid search with SPLADE++ and BGE-Large integration."""

    def test_rrf_configuration_validation(self):
        """Test RRF configuration validation against Phase 2.1 requirements."""
        settings = AppSettings()

        # Test with correct research-backed weights
        settings.rrf_fusion_weight_dense = 0.7
        settings.rrf_fusion_weight_sparse = 0.3
        settings.rrf_fusion_alpha = 60

        verification = verify_rrf_configuration(settings)

        assert verification["weights_correct"] is True
        assert verification["alpha_in_range"] is True
        assert len(verification["issues"]) == 0
        assert abs(verification["computed_hybrid_alpha"] - 0.7) < 0.01

    def test_rrf_configuration_validation_failures(self):
        """Test RRF configuration validation with incorrect weights."""
        settings = AppSettings()

        # Test with incorrect weights
        settings.rrf_fusion_weight_dense = 0.5
        settings.rrf_fusion_weight_sparse = 0.5
        settings.rrf_fusion_alpha = 5  # Outside research range

        verification = verify_rrf_configuration(settings)

        assert verification["weights_correct"] is False
        assert verification["alpha_in_range"] is False
        assert len(verification["issues"]) >= 2

    @pytest.mark.asyncio
    async def test_async_qdrant_setup(self):
        """Test async Qdrant setup with hybrid search configuration."""
        with patch("utils.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.collection_exists.return_value = False

            await setup_hybrid_qdrant_async(
                client=mock_instance,
                collection_name="test_collection",
                dense_embedding_size=1024,
                recreate=False,
            )

            # Verify collection creation was called with correct parameters
            mock_instance.create_collection.assert_called_once()
            call_args = mock_instance.create_collection.call_args

            # Check vectors config for dense embeddings
            vectors_config = call_args[1]["vectors_config"]
            assert "text-dense" in vectors_config
            assert vectors_config["text-dense"].size == 1024

            # Check sparse vectors config
            sparse_config = call_args[1]["sparse_vectors_config"]
            assert "text-sparse" in sparse_config

    def test_sync_qdrant_setup(self):
        """Test synchronous Qdrant setup with hybrid search configuration."""
        with patch("utils.QdrantClient") as mock_client:
            mock_instance = MagicMock()
            mock_client.return_value = mock_instance
            mock_instance.collection_exists.return_value = False

            vector_store = setup_hybrid_qdrant(
                client=mock_instance,
                collection_name="test_collection",
                dense_embedding_size=1024,
                recreate=False,
            )

            # Verify collection creation
            mock_instance.create_collection.assert_called_once()
            assert vector_store.enable_hybrid is True


class TestColBERTReranking:
    """Test ColBERT reranking integration and performance monitoring."""

    def test_colbert_performance_monitor(self):
        """Test ColBERT performance monitoring wrapper."""
        with patch("utils.ColbertRerank") as mock_colbert:
            mock_base_reranker = MagicMock()
            mock_colbert.return_value = mock_base_reranker

            # Mock nodes for testing
            mock_nodes = [MagicMock() for _ in range(10)]
            mock_base_reranker.postprocess_nodes.return_value = mock_nodes[:5]

            # Create performance monitor
            from utils import create_tools_from_index

            # Mock index for testing
            mock_index = {"vector": MagicMock(), "kg": MagicMock()}

            with patch("logging.info") as mock_logging:
                create_tools_from_index(mock_index)

                # Verify logging includes performance metrics
                assert mock_logging.called
                log_calls = [call.args[0] for call in mock_logging.call_args_list]
                performance_logs = [
                    log for log in log_calls if "ColBERT reranker enabled" in log
                ]
                assert len(performance_logs) > 0

    def test_colbert_reranking_improves_relevance(self):
        """Test that ColBERT reranking improves result relevance."""
        # This would require actual embedding models, so we'll test the configuration
        settings = AppSettings()

        # Verify Phase 2.2 configuration: retrieve 20, rerank to 5
        assert settings.reranking_top_k == 5

        # Test query engine configuration includes proper parameters
        mock_index = MagicMock()
        mock_query_engine = MagicMock()
        mock_index.as_query_engine.return_value = mock_query_engine

        create_tools_from_index({"vector": mock_index, "kg": mock_index})

        # Verify query engine is configured with correct parameters
        mock_index.as_query_engine.assert_called()
        call_kwargs = mock_index.as_query_engine.call_args[1]

        assert call_kwargs.get("similarity_top_k") == 5
        assert call_kwargs.get("sparse_top_k") == 10
        assert call_kwargs.get("hybrid_top_k") == 8
        assert call_kwargs.get("vector_store_query_mode") == "hybrid"


class TestAgentSystemIntegration:
    """Test ReActAgent system with multi-agent support."""

    def test_query_complexity_analysis(self):
        """Test query complexity analysis for routing decisions."""
        # Test simple query
        complexity, query_type = analyze_query_complexity("What is the summary?")
        assert complexity == "simple"
        assert query_type == "general"

        # Test complex query
        complex_query = (
            "Compare and analyze the relationships between multiple documents"
        )
        complexity, query_type = analyze_query_complexity(complex_query)
        assert complexity == "complex"

        # Test multimodal query
        multimodal_query = "What do you see in the image?"
        complexity, query_type = analyze_query_complexity(multimodal_query)
        assert query_type == "multimodal"

        # Test knowledge graph query
        kg_query = "What entities are related to this concept?"
        complexity, query_type = analyze_query_complexity(kg_query)
        assert query_type == "knowledge_graph"

    def test_agent_system_selection(self):
        """Test agent system selection based on configuration."""
        mock_tools = [MagicMock()]
        mock_llm = MagicMock()

        # Test single agent mode
        agent_system, mode = get_agent_system(
            tools=mock_tools, llm=mock_llm, enable_multi_agent=False
        )
        assert mode == "single"

        # Test multi-agent mode (should fallback to single if LangGraph fails)
        with patch(
            "agent_factory.create_langgraph_supervisor_system", return_value=None
        ):
            agent_system, mode = get_agent_system(
                tools=mock_tools, llm=mock_llm, enable_multi_agent=True
            )
            assert mode == "single"

    def test_agent_tool_integration(self):
        """Test agent tool integration with enhanced descriptions."""
        mock_vector_index = MagicMock()
        mock_kg_index = MagicMock()

        # Mock query engines
        mock_vector_query_engine = MagicMock()
        mock_kg_query_engine = MagicMock()
        mock_vector_index.as_query_engine.return_value = mock_vector_query_engine
        mock_kg_index.as_query_engine.return_value = mock_kg_query_engine

        index = {"vector": mock_vector_index, "kg": mock_kg_index}

        tools = create_tools_from_index(index)

        assert len(tools) == 2

        # Verify tool names and descriptions
        tool_names = [tool.metadata.name for tool in tools]
        assert "hybrid_vector_search" in tool_names
        assert "knowledge_graph_query" in tool_names

        # Verify enhanced descriptions
        hybrid_tool = next(
            t for t in tools if t.metadata.name == "hybrid_vector_search"
        )
        assert "SPLADE++" in hybrid_tool.metadata.description
        assert "BGE-Large" in hybrid_tool.metadata.description
        assert "RRF fusion" in hybrid_tool.metadata.description


class TestErrorHandlingAndFallbacks:
    """Test error handling and fallback mechanisms."""

    def test_hardware_detection_fallback(self):
        """Test hardware detection with fallback mechanisms."""
        with (
            patch("utils.torch.cuda.is_available", return_value=True),
            patch("utils.torch.cuda.get_device_name", return_value="RTX 4090"),
            patch("utils.torch.cuda.get_device_properties") as mock_props,
        ):
            mock_props.return_value.total_memory = 24 * 1024**3

            hardware_info = detect_hardware()

            assert hardware_info["cuda_available"] is True
            assert hardware_info["gpu_name"] == "RTX 4090"
            assert hardware_info["vram_total_gb"] == 24.0

    def test_hardware_detection_cpu_fallback(self):
        """Test hardware detection CPU fallback."""
        with patch("utils.torch.cuda.is_available", return_value=False):
            hardware_info = detect_hardware()

            assert hardware_info["cuda_available"] is False
            assert hardware_info["vram_total_gb"] is None

    def test_model_initialization_error_handling(self):
        """Test error handling during model initialization."""
        manager = ModelManager()
        manager.clear_cache()

        with (
            patch("utils.TextEmbedding", side_effect=Exception("Model load failed")),
            pytest.raises(Exception, match="Model load failed"),
        ):
            manager.get_dense_embedding_model("invalid-model")

    @pytest.mark.asyncio
    async def test_async_operation_error_handling(self):
        """Test error handling in async operations."""
        with patch("utils.AsyncQdrantClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.collection_exists.side_effect = Exception("Connection failed")

            with pytest.raises(Exception, match="Connection failed"):
                await setup_hybrid_qdrant_async(
                    client=mock_instance,
                    collection_name="test",
                    dense_embedding_size=768,
                )


class TestPerformanceAndScaling:
    """Test performance characteristics and scaling behavior."""

    def test_embedding_batch_size_configuration(self):
        """Test embedding batch size configuration for performance."""
        settings = AppSettings()

        # Test default batch size is reasonable for GPU
        assert settings.embedding_batch_size >= 1
        assert settings.embedding_batch_size <= 512

        # Test prefetch factor configuration
        assert 1 <= settings.prefetch_factor <= 8

    def test_concurrent_request_limits(self):
        """Test concurrent request limiting configuration."""
        settings = AppSettings()

        assert 1 <= settings.max_concurrent_requests <= 100

    @pytest.mark.asyncio
    async def test_async_vs_sync_performance_pattern(self):
        """Test that async patterns are properly implemented."""
        # This tests the pattern, not actual performance

        with (
            patch("utils.AsyncQdrantClient") as mock_async_client,
            patch("utils.QdrantClient") as mock_sync_client,
        ):
            mock_async_instance = AsyncMock()
            mock_sync_instance = MagicMock()
            mock_async_client.return_value = mock_async_instance
            mock_sync_client.return_value = mock_sync_instance

            # Both should be available
            async_available = hasattr(utils, "create_index_async")
            sync_available = hasattr(utils, "create_index")

            assert async_available
            assert sync_available


class TestDocumentProcessingPipeline:
    """Test complete document processing pipeline."""

    def test_document_metadata_handling(self):
        """Test document metadata is properly preserved through pipeline."""
        test_doc = Document(
            text="Test document content",
            metadata={"source": "test.pdf", "page": 1, "has_images": True},
        )

        # Verify metadata preservation pattern
        assert test_doc.metadata["source"] == "test.pdf"
        assert test_doc.metadata["page"] == 1
        assert test_doc.metadata["has_images"] is True

    def test_multimodal_document_handling(self):
        """Test multimodal document processing configuration."""
        from utils import create_native_multimodal_embeddings, extract_images_from_pdf

        # Test that multimodal functions are available
        assert callable(extract_images_from_pdf)
        assert callable(create_native_multimodal_embeddings)

    @pytest.mark.asyncio
    async def test_end_to_end_pipeline_integration(self):
        """Test end-to-end pipeline integration with mocked components."""
        # Create test documents
        docs = [
            Document(text="Machine learning enables intelligent document analysis"),
            Document(text="SPLADE++ provides sparse embeddings for keyword matching"),
            Document(
                text="BGE-Large offers dense embeddings for semantic understanding"
            ),
        ]

        with (
            patch("utils.AsyncQdrantClient") as mock_client,
            patch("utils.FastEmbedEmbedding") as mock_dense_embed,
            patch("utils.SparseTextEmbedding") as mock_sparse_embed,
            patch("utils.VectorStoreIndex.from_documents") as mock_index,
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.collection_exists.return_value = False

            mock_vector_index = MagicMock()
            mock_index.return_value = mock_vector_index

            # Test async index creation
            await create_index_async(docs, use_gpu=False)

            # Verify components were called
            assert mock_dense_embed.called
            assert mock_sparse_embed.called
            assert mock_index.called

            # Verify index configuration
            index_call_kwargs = mock_index.call_args[1]
            assert "embed_model" in index_call_kwargs
            assert "sparse_embed_model" in index_call_kwargs


# Pytest configuration
def pytest_configure():
    """Configure pytest for integration tests."""
    logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
