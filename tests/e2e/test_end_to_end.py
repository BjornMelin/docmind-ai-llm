#!/usr/bin/env python3
"""End-to-end integration tests for DocMind AI system.

This module tests the complete workflow from document loading through
agent processing, validating that all components work together correctly
in realistic scenarios.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document

from agent_factory import (
    analyze_query_complexity,
    get_agent_system,
    process_query_with_agent_system,
)
from agents.agent_utils import create_tools_from_index
from models import AppSettings
from utils import create_index_async


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow scenarios."""

    @pytest.mark.asyncio
    async def test_complete_document_processing_pipeline(self):
        """Test complete document processing from upload to query response."""
        # Step 1: Create test documents
        test_documents = [
            Document(
                text=(
                    "DocMind AI is an advanced document analysis system that uses "
                    "SPLADE++ sparse embeddings."
                ),
                metadata={"source": "doc1.pdf", "page": 1},
            ),
            Document(
                text=(
                    "BGE-Large dense embeddings provide rich semantic understanding "
                    "for document retrieval."
                ),
                metadata={"source": "doc2.pdf", "page": 1},
            ),
            Document(
                text=(
                    "ColBERT reranking improves search result relevance through "
                    "late interaction mechanisms."
                ),
                metadata={"source": "doc3.pdf", "page": 1},
            ),
            Document(
                text=(
                    "Hybrid search combines both dense and sparse retrieval methods "
                    "for optimal performance."
                ),
                metadata={"source": "doc4.pdf", "page": 1},
            ),
        ]

        # Step 2: Mock the index creation process
        with (
            patch("utils.AsyncQdrantClient") as mock_async_client,
            patch("utils.FastEmbedEmbedding") as mock_dense_embed,
            patch("utils.SparseTextEmbedding") as mock_sparse_embed,
            patch("utils.VectorStoreIndex.from_documents") as mock_index,
        ):
            # Setup comprehensive mocks
            mock_client_instance = AsyncMock()
            mock_async_client.return_value = mock_client_instance
            mock_client_instance.collection_exists.return_value = False
            mock_client_instance.close = AsyncMock()

            # Mock vector index
            mock_index_instance = MagicMock()
            mock_query_engine = MagicMock()
            mock_index_instance.as_query_engine.return_value = mock_query_engine
            mock_index.return_value = mock_index_instance

            # Step 3: Create index
            index = await create_index_async(test_documents, use_gpu=False)

            # Verify index creation
            assert index is not None
            assert mock_dense_embed.called
            assert mock_sparse_embed.called

        # Step 4: Test tool creation
        with patch("utils.ColbertRerank") as mock_colbert:
            mock_colbert_instance = MagicMock()
            mock_colbert.return_value = mock_colbert_instance

            # Create mock index dict for tools
            mock_index_dict = {"vector": mock_index, "kg": mock_index}

            tools = create_tools_from_index(mock_index_dict)

            # Verify tools are created
            assert len(tools) == 2
            assert any("hybrid_vector_search" in tool.metadata.name for tool in tools)
            assert any("knowledge_graph_query" in tool.metadata.name for tool in tools)

        # Step 5: Test agent creation and query processing
        with patch("llama_index.llms.ollama.Ollama") as mock_llm:
            mock_llm_instance = MagicMock()
            mock_llm.return_value = mock_llm_instance

            # Test agent system creation
            agent_system, mode = get_agent_system(
                tools=tools, llm=mock_llm_instance, enable_multi_agent=False
            )

            assert mode == "single"
            assert agent_system is not None

        # Step 6: Test query processing
        test_query = "What are the key features of DocMind AI?"

        with patch.object(agent_system, "chat") as mock_chat:
            mock_response = MagicMock()
            mock_response.response = (
                "DocMind AI features SPLADE++ sparse embeddings, BGE-Large dense "
                "embeddings, and ColBERT reranking for advanced document analysis."
            )
            mock_chat.return_value = mock_response

            response = process_query_with_agent_system(
                agent_system=agent_system, query=test_query, mode=mode
            )

            # Verify response
            assert response is not None
            assert "SPLADE++" in response
            assert "BGE-Large" in response
            assert "ColBERT" in response

    def test_workflow_with_different_query_types(self):
        """Test workflow with different types of queries."""
        query_test_cases = [
            ("What is this document about?", "simple", "document"),
            ("Compare the different embedding approaches", "complex", "general"),
            ("What entities are mentioned in the text?", "simple", "knowledge_graph"),
            ("Analyze the visual elements in this document", "moderate", "multimodal"),
        ]

        for query, expected_complexity, expected_type in query_test_cases:
            # Test query analysis
            complexity, query_type = analyze_query_complexity(query)

            assert complexity == expected_complexity
            assert query_type == expected_type

            # Mock agent system for query processing
            mock_agent = MagicMock()
            mock_response = MagicMock()
            mock_response.response = f"Response for {query_type} query: {query}"
            mock_agent.chat.return_value = mock_response

            response = process_query_with_agent_system(
                agent_system=mock_agent, query=query, mode="single"
            )

            assert query_type in response or "Response" in response

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery in the complete workflow."""
        # Test with invalid documents
        invalid_documents = [
            Document(text="", metadata={}),  # Empty document
            Document(text=None, metadata={"source": "invalid.pdf"}),  # None text
        ]

        # Should handle invalid documents gracefully
        with (
            patch("utils.AsyncQdrantClient") as mock_client,
            patch("utils.FastEmbedEmbedding"),
            patch("utils.SparseTextEmbedding"),
            patch("utils.VectorStoreIndex.from_documents"),
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.collection_exists.side_effect = ValueError("Connection issue")

            # Test error handling in index creation
            with pytest.raises(ValueError, match="Connection issue"):
                await create_index_async(invalid_documents, use_gpu=False)

    def test_hardware_dependent_workflow(self):
        """Test workflow behavior with different hardware configurations."""
        # Test GPU configuration workflow
        with (
            patch("utils.torch.cuda.is_available", return_value=True),
            patch("utils.torch.cuda.get_device_name", return_value="RTX 4090"),
        ):
            settings = AppSettings()

            # Verify GPU settings affect workflow
            assert settings.gpu_acceleration is True

            # Test model manager configuration
            manager = ModelManager()

            with patch("utils.TextEmbedding") as mock_embedding:
                manager.get_dense_embedding_model()

                # Verify GPU providers are used
                call_args = mock_embedding.call_args
                providers = call_args[1]["providers"]
                assert "CUDAExecutionProvider" in providers

        # Test CPU fallback workflow
        with patch("utils.torch.cuda.is_available", return_value=False):
            settings = AppSettings()

            # CPU mode should still work
            manager = ModelManager()

            with patch("utils.TextEmbedding") as mock_embedding:
                manager.get_dense_embedding_model()

                # Should use CPU providers
                call_args = mock_embedding.call_args
                providers = call_args[1]["providers"]
                assert "CPUExecutionProvider" in providers


class TestWorkflowPerformance:
    """Test workflow performance characteristics."""

    @pytest.mark.asyncio
    async def test_async_workflow_performance(self):
        """Test async workflow performance benefits."""
        import time

        # Create test documents
        docs = [Document(text=f"Test document {i}") for i in range(5)]

        # Mock async components for timing test
        with (
            patch("utils.AsyncQdrantClient") as mock_client,
            patch("utils.FastEmbedEmbedding"),
            patch("utils.SparseTextEmbedding"),
            patch("utils.VectorStoreIndex.from_documents") as mock_index,
        ):
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            mock_instance.collection_exists.return_value = False
            mock_instance.close = AsyncMock()

            mock_vector_index = MagicMock()
            mock_index.return_value = mock_vector_index

            start_time = time.perf_counter()

            # Test async workflow
            result = await create_index_async(docs, use_gpu=False)

            async_time = time.perf_counter() - start_time

            # Should complete in reasonable time
            assert async_time < 2.0
            assert result is not None

    def test_concurrent_workflow_handling(self):
        """Test handling of concurrent workflow operations."""
        from concurrent.futures import ThreadPoolExecutor

        def create_agent_workflow():
            # Mock components for concurrent test
            mock_tools = [MagicMock(), MagicMock()]
            mock_llm = MagicMock()

            agent_system, mode = get_agent_system(
                tools=mock_tools, llm=mock_llm, enable_multi_agent=False
            )

            return (agent_system, mode)

        # Test concurrent agent creation
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(create_agent_workflow) for _ in range(5)]
            results = [future.result() for future in futures]

        # All should succeed
        assert len(results) == 5
        for agent_system, mode in results:
            assert agent_system is not None
            assert mode == "single"


class TestWorkflowIntegration:
    """Test integration between workflow components."""

    def test_settings_workflow_integration(self):
        """Test that settings properly integrate throughout workflow."""
        settings = AppSettings()

        # Verify key settings for workflow
        assert settings.dense_embedding_model is not None
        assert settings.sparse_embedding_model is not None
        assert settings.reranking_top_k == 5  # Phase 2.2 requirement
        assert settings.enable_colbert_reranking is True

        # Verify RRF configuration
        assert abs(settings.rrf_fusion_weight_dense - 0.7) < 0.05
        assert abs(settings.rrf_fusion_weight_sparse - 0.3) < 0.05

    def test_model_manager_workflow_integration(self):
        """Test FastEmbedModelManager integration throughout workflow."""
        manager = ModelManager()

        # Test singleton behavior in workflow context
        manager1 = ModelManager()
        manager2 = ModelManager()

        assert manager1 is manager2
        assert manager1 is manager

        # Test cache management
        manager.clear_cache()
        assert len(manager._models) == 0

    def test_agent_tool_workflow_integration(self):
        """Test agent and tool integration in workflow."""
        # Mock index for tool creation
        mock_vector_index = MagicMock()
        mock_kg_index = MagicMock()

        mock_vector_query_engine = MagicMock()
        mock_kg_query_engine = MagicMock()
        mock_vector_index.as_query_engine.return_value = mock_vector_query_engine
        mock_kg_index.as_query_engine.return_value = mock_kg_query_engine

        index_dict = {"vector": mock_vector_index, "kg": mock_kg_index}

        # Create tools
        tools = create_tools_from_index(index_dict)

        # Verify tool integration
        assert len(tools) == 2

        tool_names = [tool.metadata.name for tool in tools]
        assert "hybrid_vector_search" in tool_names
        assert "knowledge_graph_query" in tool_names

        # Verify enhanced descriptions
        for tool in tools:
            assert (
                len(tool.metadata.description) > 50
            )  # Should have detailed descriptions


class TestWorkflowValidation:
    """Test workflow validation and correctness."""

    def test_workflow_component_availability(self):
        """Test that all workflow components are available."""
        # Test function imports
        from agent_factory import (
            analyze_query_complexity,
        )
        from utils import (
            create_index,
            create_index_async,
            create_tools_from_index,
        )

        # All imports should succeed
        assert callable(create_index_async)
        assert callable(create_index)
        assert callable(create_tools_from_index)
        assert callable(analyze_query_complexity)

    def test_workflow_configuration_validation(self):
        """Test workflow configuration validation."""
        settings = AppSettings()

        # Validate critical configurations
        assert settings.dense_embedding_model.startswith("BAAI/bge-large")
        assert "Splade_PP_en_v1" in settings.sparse_embedding_model
        assert settings.dense_embedding_dimension == 1024
        assert settings.rrf_fusion_alpha == 60

    def test_workflow_error_handling(self):
        """Test error handling throughout workflow."""
        # Test invalid settings handling
        settings = AppSettings()

        # Should handle validation gracefully
        from utils import verify_rrf_configuration

        verification = verify_rrf_configuration(settings)

        assert isinstance(verification, dict)
        assert "weights_correct" in verification
        assert "alpha_in_range" in verification


# Test configuration
def pytest_configure():
    """Configure pytest for end-to-end tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
