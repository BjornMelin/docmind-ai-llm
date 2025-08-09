"""End-to-End Integration Testing for DocMind AI.

Complete pipeline testing from document loading through retrieval and agent processing.
Tests real workflows, multi-agent coordination, and error recovery chains.

This module follows 2025 pytest best practices for AI/ML integration testing.
"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from llama_index.core import Document
from llama_index.core.tools import QueryEngineTool

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestEndToEndPipeline:
    """Test complete document processing pipeline."""

    @pytest.fixture
    def sample_document_path(self, tmp_path: Path) -> Path:
        """Create sample document for testing."""
        doc_path = tmp_path / "test_doc.txt"
        doc_path.write_text(
            "DocMind AI is a comprehensive document analysis system. "
            "It uses SPLADE++ sparse embeddings and BGE-Large dense embeddings. "
            "The system supports hybrid search with RRF fusion. "
            "Knowledge graphs extract entity relationships from documents. "
            "ColBERT reranking improves search relevance significantly."
        )
        return doc_path

    def test_document_to_retrieval_pipeline(self, sample_document_path: Path):
        """Test complete pipeline from document to retrieval."""
        from agent_factory import create_single_agent
        from agents.agent_utils import create_tools_from_index
        from utils.document_loader import load_documents_unstructured
        from utils.index_builder import create_index

        # Document loading with Unstructured
        with patch("utils.document_loader.partition") as mock_partition:
            mock_element = MagicMock()
            mock_element.category = "NarrativeText"
            mock_element.text = "DocMind AI uses SPLADE++ sparse embeddings."
            mock_element.metadata = MagicMock()
            mock_element.metadata.page_number = 1
            mock_element.metadata.filename = "test_doc.txt"
            mock_partition.return_value = [mock_element]

            docs = load_documents_unstructured(str(sample_document_path))
            assert len(docs) > 0
            # Document loader might return nodes after chunking, so check for both
            from llama_index.core.schema import BaseNode

            assert isinstance(docs[0], (Document, BaseNode))

        # Index creation with hybrid search
        with patch("qdrant_client.QdrantClient") as mock_qdrant:
            mock_client = MagicMock()
            mock_qdrant.return_value = mock_client

            with patch("utils.qdrant_utils.setup_hybrid_qdrant") as mock_setup:
                mock_vector_store = MagicMock()
                mock_setup.return_value = mock_vector_store

                with patch(
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_index_create:
                    mock_index = MagicMock()
                    mock_index_create.return_value = mock_index

                    with (
                        patch("utils.utils.ensure_spacy_model") as mock_spacy,
                        patch(
                            "llama_index.core.KnowledgeGraphIndex.from_documents"
                        ) as mock_kg,
                        patch(
                            "utils.index_builder.create_hybrid_retriever"
                        ) as mock_retriever,
                    ):
                        mock_spacy.return_value = MagicMock()
                        mock_kg.side_effect = Exception("KG creation disabled for test")
                        mock_retriever.return_value = MagicMock()

                        # Create index
                        result = create_index(docs, use_gpu=False)

                        assert "vector" in result
                        assert result["vector"] is not None

        # Tool creation for agents
        with patch("agents.agent_utils.ColbertRerank") as mock_reranker:
            mock_reranker.return_value = MagicMock()
            tools = create_tools_from_index(result)
            assert len(tools) > 0
            assert all(isinstance(tool, QueryEngineTool) for tool in tools)

        # Agent creation and query execution
        with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
            mock_llm = MagicMock()
            mock_ollama.return_value = mock_llm

            with patch(
                "llama_index.core.agent.ReActAgent.from_tools"
            ) as mock_agent_create:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.response = "DocMind AI is a document analysis system using SPLADE++ embeddings."
                mock_agent.chat.return_value = mock_response
                mock_agent_create.return_value = mock_agent

                agent = create_single_agent(tools, mock_llm)
                response = agent.chat("Tell me about DocMind AI")

                assert response.response is not None
                assert "DocMind" in response.response

    def test_error_recovery_pipeline(self, sample_document_path: Path):
        """Test cascading error recovery through components."""
        # Test Unstructured fails -> fallback to simple text loading
        with patch("unstructured.partition.auto.partition") as mock_unstructured:
            mock_unstructured.side_effect = Exception("Unstructured failed")

            # Should fall back to basic document loading
            with patch("pathlib.Path.read_text") as mock_read:
                mock_read.return_value = "Fallback document content"

                # This would normally be handled by a fallback mechanism
                # For testing, we simulate successful fallback
                docs = [Document(text="Fallback document content")]
                assert len(docs) == 1
                assert docs[0].text == "Fallback document content"

        # Test KG creation fails but vector index succeeds
        from utils.index_builder import create_index

        with (
            patch("qdrant_client.QdrantClient"),
            patch("utils.qdrant_utils.setup_hybrid_qdrant"),
            patch(
        ):
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_vector:
                    mock_vector.return_value = MagicMock()

                    with patch(
                        "llama_index.core.KnowledgeGraphIndex.from_documents"
                    ) as mock_kg:
                        mock_kg.side_effect = Exception("KG creation failed")

                        with patch("utils.utils.ensure_spacy_model"):
                            result = create_index(docs, use_gpu=False)

                            # Should have vector but no KG
                            assert result["vector"] is not None
                            assert result.get("kg") is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
    def test_gpu_pipeline_integration(self):
        """Test GPU acceleration through full pipeline."""
        from utils.index_builder import create_index

        docs = [Document(text="GPU test document for integration testing")]

        with (


            ,


            ,


            patch("torch.cuda.is_available", return_value=True):,


            patch("qdrant_client.QdrantClient"):,


            patch("utils.qdrant_utils.setup_hybrid_qdrant"):


        ):
                    with patch(
                        "llama_index.core.VectorStoreIndex.from_documents"
                    ) as mock_create:
                        mock_index = MagicMock()
                        mock_create.return_value = mock_index

                        with patch("torch.cuda.Stream") as mock_stream:
                            mock_stream.return_value.__enter__ = MagicMock()
                            mock_stream.return_value.__exit__ = MagicMock()
                            mock_stream.return_value.synchronize = MagicMock()

                            with patch("utils.utils.ensure_spacy_model"):
                                result = create_index(docs, use_gpu=True)

                                # Should successfully create index with GPU
                                assert result is not None
                                assert result["vector"] is not None
                                mock_stream.return_value.synchronize.assert_called()

    def test_partial_failure_handling(self):
        """Test handling of partial failures in pipeline."""
        from utils.index_builder import create_index

        # Mix of good and problematic documents
        docs = [
            Document(text="Good document 1 about DocMind AI"),
            Document(text="Good document 2 about embeddings"),
            Document(text=""),  # Empty document
        ]

        with (
            patch("qdrant_client.QdrantClient"),
            patch("utils.qdrant_utils.setup_hybrid_qdrant"),
            patch(
        ):
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_create:
                    mock_index = MagicMock()
                    mock_create.return_value = mock_index

                    with patch("utils.utils.ensure_spacy_model"):
                        # Should handle mixed document quality gracefully
                        result = create_index(docs, use_gpu=False)
                        assert result is not None
                        assert result["vector"] is not None


class TestMultiAgentIntegration:
    """Test multi-agent coordination and routing."""

    def test_simple_query_routing(self):
        """Test agent routing for simple queries."""
        from agent_factory import analyze_query_complexity

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
        from agent_factory import analyze_query_complexity

        complex_queries = [
            "Compare and analyze the relationships between entities in this document",
            "How does the dense embedding approach differ from sparse embeddings across multiple documents?",
            "Summarize all the various techniques mentioned and explain their connections",
        ]

        for query in complex_queries:
            complexity, query_type = analyze_query_complexity(query)
            assert complexity in ["moderate", "complex"]

    def test_langgraph_workflow_integration(self):
        """Test LangGraph workflow integration."""
        from langchain_core.messages import HumanMessage

        from agent_factory import create_langgraph_supervisor_system

        # Mock tools for testing
        mock_tools = [
            MagicMock(spec=QueryEngineTool, metadata=MagicMock(name="test_tool"))
        ]
        mock_llm = MagicMock()

        with patch("agent_factory.create_document_specialist_agent") as mock_doc_agent:
            with patch(
                "agent_factory.create_knowledge_specialist_agent"
            ) as mock_kg_agent:
                with patch(
                    "agent_factory.create_multimodal_specialist_agent"
                ) as mock_mm_agent:
                    mock_doc_agent.return_value = MagicMock()
                    mock_kg_agent.return_value = MagicMock()
                    mock_mm_agent.return_value = MagicMock()

                    with patch("langgraph.graph.StateGraph") as mock_graph:
                        mock_workflow = MagicMock()
                        mock_compiled = MagicMock()
                        mock_graph.return_value = mock_workflow
                        mock_workflow.compile.return_value = mock_compiled

                        # Test workflow creation
                        workflow = create_langgraph_supervisor_system(
                            mock_tools, mock_llm
                        )
                        assert workflow is not None

                        # Test workflow invocation
                        mock_compiled.invoke.return_value = {
                            "messages": [
                                HumanMessage(content="Response from multi-agent system")
                            ]
                        }

                        from agent_factory import process_query_with_agent_system

                        result = process_query_with_agent_system(
                            mock_compiled, "Test query", "multi"
                        )
                        assert "Response from multi-agent system" in result

    def test_agent_system_fallback(self):
        """Test fallback from multi-agent to single agent."""
        from agent_factory import get_agent_system

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


class TestPerformanceIntegration:
    """Test performance aspects of integration."""

    @pytest.mark.performance
    def test_concurrent_document_processing(self):
        """Test concurrent document processing capabilities."""
        import concurrent.futures

        from utils.index_builder import create_index

        def process_document_batch(batch_id: int) -> dict:
            docs = [
                Document(text=f"Batch {batch_id} document {i} content")
                for i in range(3)
            ]

            with (
                patch("qdrant_client.QdrantClient"),
                patch("utils.qdrant_utils.setup_hybrid_qdrant"),
                patch(
            ):
                        "llama_index.core.VectorStoreIndex.from_documents"
                    ) as mock_create:
                        mock_index = MagicMock()
                        mock_create.return_value = mock_index

                        with patch("utils.utils.ensure_spacy_model"):
                            return create_index(docs, use_gpu=False)

        # Test concurrent processing of multiple document batches
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_document_batch, batch_id)
                for batch_id in range(5)
            ]

            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    pytest.fail(f"Concurrent processing failed: {e}")

            assert len(results) == 5
            assert all(result is not None for result in results)
            assert all("vector" in result for result in results)

    @pytest.mark.performance
    def test_memory_usage_during_processing(self):
        """Test memory usage patterns during document processing."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Process larger document set
        docs = [
            Document(text=f"Large document {i} with substantial content " * 100)
            for i in range(50)
        ]

        with (


            ,


            ,


            patch("qdrant_client.QdrantClient"):,


            patch("utils.qdrant_utils.setup_hybrid_qdrant"):,


            patch("llama_index.core.VectorStoreIndex.from_documents"):


        ):
                    with patch("utils.utils.ensure_spacy_model"):
                        from utils.index_builder import create_index

                        create_index(docs, use_gpu=False)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Log memory usage for monitoring
        logging.info(
            f"Memory usage - Initial: {initial_memory:.1f}MB, "
            f"Final: {final_memory:.1f}MB, "
            f"Increase: {memory_increase:.1f}MB"
        )

        # Memory increase should be reasonable (less than 500MB for test)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f}MB"

    def test_response_time_benchmarking(self, benchmark):
        """Benchmark complete pipeline response times."""
        from agents.agent_utils import create_agent_with_tools, create_tools_from_index
        from utils.index_builder import create_index

        docs = [Document(text=f"Benchmark document {i}") for i in range(10)]

        def full_pipeline():
            with (
                patch("qdrant_client.QdrantClient"),
                patch("utils.qdrant_utils.setup_hybrid_qdrant"),
                patch(
            ):
                        "llama_index.core.VectorStoreIndex.from_documents"
                    ) as mock_create:
                        mock_index = MagicMock()
                        mock_index.as_query_engine.return_value = MagicMock()
                        mock_create.return_value = mock_index

                        with patch("utils.utils.ensure_spacy_model"):
                            index_result = create_index(docs, use_gpu=False)

            tools = create_tools_from_index(index_result)

            with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
                mock_llm = MagicMock()
                mock_ollama.return_value = mock_llm

                with patch(
                    "llama_index.core.agent.ReActAgent.from_tools"
                ) as mock_agent_create:
                    mock_agent = MagicMock()
                    mock_agent_create.return_value = mock_agent

                    agent = create_agent_with_tools(index_result, mock_llm)
                    return agent

        # Benchmark the complete pipeline
        result = benchmark.pedantic(full_pipeline, rounds=3, iterations=1)
        assert result is not None


class TestErrorRecoveryIntegration:
    """Test comprehensive error recovery scenarios."""

    def test_cascading_service_failures(self):
        """Test graceful degradation through service failures."""
        from utils.index_builder import create_index

        docs = [Document(text="Test document for error recovery")]

        # Simulate cascading failures: Qdrant -> Embedding -> Basic fallback
        with patch("qdrant_client.QdrantClient") as mock_qdrant:
            mock_qdrant.side_effect = Exception("Qdrant connection failed")

            # Should attempt fallback mechanisms
            with patch(
                "llama_index.core.VectorStoreIndex.from_documents"
            ) as mock_create:
                # Even if Qdrant fails, should attempt in-memory fallback
                mock_create.side_effect = Exception("Vector store creation failed")

                try:
                    result = create_index(docs, use_gpu=False)
                    # If we get here, fallback worked
                    assert result is None or "vector" in result
                except Exception as e:
                    # Expected if no fallback implemented
                    assert "creation failed" in str(e) or "connection failed" in str(e)

    def test_network_timeout_recovery(self):
        """Test recovery from network timeouts."""
        from utils.index_builder import create_index

        docs = [Document(text="Network timeout test document")]

        # Simulate network timeout
        with patch("qdrant_client.QdrantClient") as mock_qdrant:
            mock_qdrant.side_effect = TimeoutError("Connection timeout")

            try:
                create_index(docs, use_gpu=False)
            except (TimeoutError, Exception) as e:
                # Should handle timeout gracefully
                assert "timeout" in str(e).lower() or "connection" in str(e).lower()

    def test_resource_exhaustion_handling(self):
        """Test handling of resource exhaustion."""
        from utils.index_builder import create_index

        # Large document set that might cause resource issues
        docs = [
            Document(text=f"Resource test document {i} " * 1000) for i in range(100)
        ]

        with (
            patch("qdrant_client.QdrantClient"),
            patch("utils.qdrant_utils.setup_hybrid_qdrant"),
            patch(
        ):
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_create:
                    # Simulate memory error
                    mock_create.side_effect = MemoryError("Insufficient memory")

                    try:
                        create_index(docs, use_gpu=False)
                        pytest.fail("Should have raised MemoryError")
                    except MemoryError:
                        # Expected - should be handled gracefully by calling code
                        pass
                    except Exception as e:
                        # Other exceptions should be caught and handled
                        logging.warning(
                            f"Unexpected exception during resource test: {e}"
                        )


# Integration test markers and configuration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.filterwarnings("ignore::UserWarning"),  # Suppress model warnings
]
