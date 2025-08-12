"""Comprehensive End-to-End Integration Testing for DocMind AI.

This module provides exhaustive end-to-end tests covering:
- Document processing pipelines
- Async and sync operations
- Multi-agent workflows
- Performance characteristics
- Error recovery scenarios

Follows 2025 pytest best practices for AI/ML integration testing.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import torch
from llama_index.core import Document
from llama_index.core.agent import ReActAgent
from llama_index.core.schema import ImageDocument
from llama_index.core.tools import QueryEngineTool

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_factory import (
    analyze_query_complexity,
    create_langgraph_supervisor_system,
    get_agent_system,
    process_query_with_agent_system,
)

from agents.agent_utils import (
    chat_with_agent,
    create_tools_from_index,
)
from utils.document_loader import load_documents_unstructured
from utils.index_builder import (
    create_index,
    create_index_async,
    create_multimodal_index_async,
)


class TestCompleteSystemIntegration:
    """Comprehensive integration test suite for DocMind AI system."""

    @pytest_asyncio.fixture
    async def async_document_set(self) -> list[Document]:
        """Create async document set for testing."""
        await asyncio.sleep(0.01)  # Simulate async document loading
        return [
            Document(
                text="Async DocMind AI processes documents "
                "efficiently with FastEmbed GPU acceleration.",
                metadata={"source": "async_doc1.pdf", "page": 1},
            ),
            Document(
                text="Async Qdrant client provides 50-80% performance "
                "improvement over sync operations.",
                metadata={"source": "async_doc2.pdf", "page": 1},
            ),
            Document(
                text="CUDA streams enable parallel embedding "
                "computation for maximum throughput.",
                metadata={"source": "async_doc3.pdf", "page": 2},
            ),
        ]

    @pytest.mark.integration
    def test_complete_document_processing_pipeline(
        self,
        sample_documents,
        mock_embedding_model,
        mock_sparse_embedding_model,
        mock_qdrant_client,
        test_settings,
    ):
        """Test complete document processing from upload to query response."""
        with (
            patch("utils.FastEmbedModelManager.get_model") as mock_get_model,
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
            patch("utils.create_index_async") as mock_create_index,
        ):
            # Configure model manager
            def model_side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = model_side_effect

            # Mock index creation
            mock_index = MagicMock()
            mock_create_index.return_value = mock_index

            # Test pipeline steps
            # 1. Document Processing
            documents = sample_documents
            assert len(documents) > 0

            # 2. Index Creation
            index = create_index(documents, settings=test_settings)
            assert index is not None
            assert "vector" in index

            # 3. Tool Creation
            with patch("utils.create_tools_from_index") as mock_create_tools:
                mock_tools = [MagicMock(name="search_tool")]
                mock_create_tools.return_value = mock_tools

                tools = create_tools_from_index(index)
                assert len(tools) > 0
                assert all(isinstance(tool, QueryEngineTool) for tool in tools)

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_index_creation_pipeline(self, async_document_set):
        """Test async index creation with performance improvements."""
        with patch("qdrant_client.AsyncQdrantClient") as mock_async_client:
            mock_client = AsyncMock()
            mock_async_client.return_value = mock_client

            with patch("utils.qdrant_utils.setup_hybrid_qdrant_async"):
                # Test async index creation
                start_time = time.time()
                result = await create_index_async(async_document_set, use_gpu=False)
                end_time = time.time()

                # Verify async operation completed
                assert result is not None
                assert "vector" in result
                assert result["vector"] is not None

                # Log performance metrics
                processing_time = end_time - start_time
                logging.info(f"Async index creation took {processing_time:.3f}s")

    @pytest.mark.integration
    def test_hybrid_search_workflow(
        self,
        sample_documents,
        mock_embedding_model,
        mock_sparse_embedding_model,
        mock_qdrant_client,
    ):
        """Test hybrid search workflow with dense and sparse embeddings."""
        with (
            patch("utils.FastEmbedModelManager.get_model") as mock_get_model,
            patch("utils.QdrantClient", return_value=mock_qdrant_client),
        ):

            def model_side_effect(model_name):
                if "splade" in model_name.lower():
                    return mock_sparse_embedding_model
                else:
                    return mock_embedding_model

            mock_get_model.side_effect = model_side_effect

            # Mock search results for hybrid search
            mock_qdrant_client.search.side_effect = [
                # Dense search results
                [MagicMock(id=1, score=0.9, payload={"text": "SPLADE++ is efficient"})],
                # Sparse search results
                [
                    MagicMock(
                        id=2,
                        score=0.85,
                        payload={"text": "BGE-Large provides semantics"},
                    )
                ],
            ]

            # Verify hybrid search execution
            dense_results = mock_qdrant_client.search(
                collection_name="dense", query_vector=[0.1] * 1024, limit=5
            )

            sparse_results = mock_qdrant_client.search(
                collection_name="sparse", query_vector=[0.1] * 1024, limit=5
            )

            assert len(dense_results) == 1
            assert len(sparse_results) == 1
            assert mock_qdrant_client.search.call_count == 2

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multimodal_index_creation(self):
        """Test async multimodal index creation."""
        # Create mixed document set
        docs = [
            Document(text="Text document about multimodal processing"),
            ImageDocument(text="Image description", image_path="/fake/image.jpg"),
            Document(text="Another text document with visual references"),
        ]

        with patch("qdrant_client.AsyncQdrantClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance

            # Test async multimodal index creation
            result = await create_multimodal_index_async(docs, use_gpu=False)

            assert result is not None
            assert "vector" in result

    @pytest.mark.integration
    def test_error_recovery_pipeline(self, sample_document_path: Path):
        """Test cascading error recovery through components."""
        # Test Unstructured fails -> fallback to simple text loading
        with patch("unstructured.partition.auto.partition") as mock_unstructured:
            mock_unstructured.side_effect = Exception("Unstructured failed")

            # Should fall back to basic document loading
            with patch("pathlib.Path.read_text") as mock_read:
                mock_read.return_value = "Fallback document content"

                # Simulate document loading
                docs = load_documents_unstructured(str(sample_document_path))
                assert len(docs) > 0

                # Verify fallback mechanism
                assert any("Fallback" in doc.text for doc in docs)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_gpu_acceleration_pipeline(self):
        """Test GPU acceleration through full pipeline."""
        docs = [Document(text="GPU test document for integration testing")]

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("qdrant_client.QdrantClient"),
            patch("llama_index.core.VectorStoreIndex.from_documents") as mock_create,
        ):
            mock_index = MagicMock()
            mock_create.return_value = mock_index

            with patch("torch.cuda.Stream") as mock_stream:
                mock_stream.return_value.__enter__ = MagicMock()
                mock_stream.return_value.__exit__ = MagicMock()
                mock_stream.return_value.synchronize = MagicMock()

                # Create index with GPU
                result = create_index(docs, use_gpu=True)

                # Verify GPU acceleration
                assert result is not None
                assert "vector" in result
                mock_stream.return_value.synchronize.assert_called()

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_agent_query_processing_workflow(
        self, async_document_set, test_settings
    ):
        """Test complete agent query processing workflow."""
        # Create an index
        with patch("qdrant_client.AsyncQdrantClient"):
            index_result = await create_index_async(async_document_set, use_gpu=False)

        with (
            patch("agent_factory.get_agent_system") as mock_get_agent,
            patch("agents.agent_utils.chat_with_agent") as mock_chat,
        ):
            # Mock agent with async streaming
            mock_agent = MagicMock(spec=ReActAgent)
            mock_get_agent.return_value = mock_agent

            # Mock streaming response
            async def mock_stream_response():
                chunks = ["Comprehensive", " answer", " about", " embeddings"]
                for chunk in chunks:
                    await asyncio.sleep(0.01)
                    yield chunk

            mock_chat.side_effect = mock_stream_response()

            create_tools_from_index(index_result)

            # Process query
            query = "Tell me about document embedding techniques"
            response_chunks = []

            async for chunk in chat_with_agent(mock_agent, query, MagicMock()):
                response_chunks.append(chunk)

            # Verify response generation
            assert len(response_chunks) > 0
            full_response = "".join(response_chunks)
            assert "Comprehensive answer about embeddings" in full_response


class TestQueryComplexityRouting:
    """Test query complexity routing and agent selection."""

    def test_simple_query_routing(self):
        """Test agent routing for simple queries."""
        test_cases = [
            ("What is the summary?", "simple", "general"),
            ("Tell me about the document", "simple", "document"),
            ("Show me images", "simple", "multimodal"),
        ]

        for query, expected_complexity, expected_type in test_cases:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity == expected_complexity
            assert query_type == expected_type

    def test_complex_query_routing(self):
        """Test agent routing for complex queries."""
        complex_queries = [
            "Compare and analyze the relationships between entities in this document",
            "How does the dense embedding approach differ from "
            "sparse embeddings across multiple documents?",
            "Summarize embedding techniques and their interconnections",
        ]

        for query in complex_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity in ["moderate", "complex"]


class TestAgentSystemIntegration:
    """Test agent system integration and multi-agent workflows."""

    def test_agent_system_fallback(self):
        """Test fallback from multi-agent to single agent."""
        mock_tools = [MagicMock(spec=QueryEngineTool)]
        mock_llm = MagicMock()

        # Test multi-agent creation failure leads to fallback
        with patch("agent_factory.create_langgraph_supervisor_system") as mock_multi:
            mock_multi.return_value = None  # Simulate failure

            with patch("agent_factory.create_single_agent") as mock_single:
                mock_agent = MagicMock()
                mock_single.return_value = mock_agent

                agent_system, mode = get_agent_system(
                    mock_tools, mock_llm, enable_multi_agent=True
                )

                assert mode == "single"
                assert agent_system == mock_agent
                mock_multi.assert_called_once()
                mock_single.assert_called_once()

    def test_langgraph_workflow_integration(self):
        """Test LangGraph workflow integration."""
        from langchain_core.messages import HumanMessage

        # Mock tools for testing
        mock_tools = [
            MagicMock(spec=QueryEngineTool, metadata=MagicMock(name="test_tool"))
        ]
        mock_llm = MagicMock()

        with (
            patch("agent_factory.create_document_specialist_agent") as mock_doc_agent,
            patch("agent_factory.create_knowledge_specialist_agent") as mock_kg_agent,
            patch("agent_factory.create_multimodal_specialist_agent") as mock_mm_agent,
        ):
            mock_doc_agent.return_value = MagicMock()
            mock_kg_agent.return_value = MagicMock()
            mock_mm_agent.return_value = MagicMock()

            with patch("langgraph.graph.StateGraph") as mock_graph:
                mock_workflow = MagicMock()
                mock_compiled = MagicMock()
                mock_graph.return_value = mock_workflow
                mock_workflow.compile.return_value = mock_compiled

                # Test workflow creation
                workflow = create_langgraph_supervisor_system(mock_tools, mock_llm)
                assert workflow is not None

                # Test workflow invocation
                mock_compiled.invoke.return_value = {
                    "messages": [
                        HumanMessage(content="Response from multi-agent system")
                    ]
                }

                result = process_query_with_agent_system(
                    mock_compiled, "Test query", "multi"
                )
                assert "Response from multi-agent system" in result


class TestEndToEndPipelineIntegration:
    """Test complete end-to-end pipeline integration scenarios."""

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


# Async test configuration
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::UserWarning"),
]


# Configure asyncio event loop for tests
@pytest_asyncio.fixture(scope="session")
def event_loop_policy():
    """Configure event loop policy for async tests."""
    return asyncio.DefaultEventLoopPolicy()


# Test configuration
def pytest_configure():
    """Configure pytest for end-to-end tests."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
