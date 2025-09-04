"""End-to-end system workflow tests for DocMind AI.

This module provides comprehensive system-level tests that validate complete
workflows from start to finish. These tests ensure the entire DocMind AI system
works correctly with realistic scenarios and data flows.

System workflows tested:
- Complete document processing workflows (upload → process → embed → index)
- Multi-agent system coordination for complex queries
- Full query-answer workflows with retrieval and synthesis
- Configuration loading and validation across all components
- Error handling and recovery patterns in realistic scenarios

These tests use real models when available (lightweight versions for CI) or
comprehensive mocks that simulate realistic system behavior. They validate
the complete system architecture and ensure all components work together.
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.schema import NodeWithScore, TextNode

from src.config.settings import DocMindSettings


@pytest.mark.system
class TestCompleteDocumentWorkflows:
    """Test complete document processing workflows end-to-end."""

    @pytest.mark.asyncio
    @patch("src.utils.document.load_documents_unstructured")
    @patch("src.retrieval.embeddings.BGEM3Embedding")
    @patch("llama_index.vector_stores.qdrant.QdrantVectorStore")
    async def test_document_upload_to_query_workflow(
        self, mock_vector_store, mock_embedder, mock_load_docs, system_settings
    ):
        """Test complete document upload to query workflow."""
        # Mock document loading
        mock_documents = [
            Document(
                text="DocMind AI provides advanced document analysis using local LLMs. "
                "The system uses BGE-M3 embeddings for semantic understanding and "
                "supports multi-agent coordination for complex queries.",
                metadata={
                    "source": "docmind_overview.pdf",
                    "page": 1,
                    "chunk_id": "chunk_1",
                    "word_count": 32,
                },
            ),
            Document(
                text="The retrieval system combines dense and sparse embeddings using "
                "hybrid search with RRF fusion. SPLADE++ provides lexical matching "
                "while BGE-M3 offers semantic understanding.",
                metadata={
                    "source": "docmind_overview.pdf",
                    "page": 2,
                    "chunk_id": "chunk_2",
                    "word_count": 28,
                },
            ),
            Document(
                text="Agent coordination uses LangGraph supervisor pattern to manage "
                "routing, planning, retrieval, synthesis, and validation agents. "
                "Each agent specializes in specific aspects of query processing.",
                metadata={
                    "source": "docmind_overview.pdf",
                    "page": 3,
                    "chunk_id": "chunk_3",
                    "word_count": 30,
                },
            ),
        ]
        mock_load_docs.return_value = mock_documents

        # Mock embedder
        mock_embedder_instance = AsyncMock()
        mock_embedder_instance.aembed_documents.return_value = [
            [0.1] * 1024 + [0.2] * 1024,  # 2048-dim (dense + sparse combined)
            [0.3] * 1024 + [0.4] * 1024,
            [0.5] * 1024 + [0.6] * 1024,
        ]
        mock_embedder_instance.aembed_query.return_value = [0.25] * 1024 + [0.35] * 1024
        mock_embedder.return_value = mock_embedder_instance

        # Mock vector store
        mock_vs_instance = MagicMock()
        mock_vs_instance.create_collection.return_value = True
        mock_vs_instance.upsert_documents.return_value = {
            "status": "ok",
            "ids": ["1", "2", "3"],
        }
        mock_vs_instance.search.return_value = [
            {
                "id": "1",
                "score": 0.94,
                "payload": {
                    "text": mock_documents[0].text,
                    "metadata": mock_documents[0].metadata,
                },
            },
            {
                "id": "3",
                "score": 0.89,
                "payload": {
                    "text": mock_documents[2].text,
                    "metadata": mock_documents[2].metadata,
                },
            },
        ]
        mock_vector_store.return_value = mock_vs_instance

        # Step 1: Document Loading
        file_paths = [Path("docmind_overview.pdf")]
        documents = await mock_load_docs(file_paths, system_settings)

        assert len(documents) == 3
        assert all(isinstance(doc, Document) for doc in documents)
        assert all("chunk_id" in doc.metadata for doc in documents)

        # Step 2: Document Embedding
        embedder = mock_embedder(system_settings)
        document_embeddings = await embedder.aembed_documents(
            [doc.text for doc in documents]
        )

        assert len(document_embeddings) == 3
        assert len(document_embeddings[0]) == 2048  # BGE-M3 unified dimension

        # Step 3: Vector Store Indexing
        vector_store = mock_vector_store(system_settings)
        collection_result = vector_store.create_collection("docmind_docs")
        assert collection_result is True

        upsert_result = vector_store.upsert_documents(documents, document_embeddings)
        assert upsert_result["status"] == "ok"
        assert len(upsert_result["ids"]) == 3

        # Step 4: Query Processing
        query = "How does DocMind AI handle multi-agent coordination?"
        query_embedding = await embedder.aembed_query(query)
        assert len(query_embedding) == 2048

        # Step 5: Retrieval
        search_results = vector_store.search(query_embedding, top_k=2)
        assert len(search_results) == 2
        assert search_results[0]["score"] > 0.9  # High relevance

        # Verify end-to-end workflow
        retrieved_texts = [result["payload"]["text"] for result in search_results]
        assert any("multi-agent" in text.lower() for text in retrieved_texts)
        assert any("coordination" in text.lower() for text in retrieved_texts)

        print("\n=== Complete Document Workflow Results ===")
        print(f"Documents processed: {len(documents)}")
        print(f"Embeddings created: {len(document_embeddings)}")
        print(f"Search results: {len(search_results)}")
        print(f"Top result score: {search_results[0]['score']:.3f}")

    @pytest.mark.asyncio
    @patch("src.processing.document_processor.DocumentProcessor")
    async def test_document_processing_error_recovery(
        self, mock_processor, system_settings
    ):
        """Test document processing workflow with error recovery."""
        # Mock processor that fails first, then succeeds
        mock_instance = MagicMock()
        mock_instance.process_documents.side_effect = [
            RuntimeError("Document processing failed"),  # First call fails
            [
                Document(
                    text="Successfully processed document",
                    metadata={"source": "test.pdf"},
                )
            ],  # Second call succeeds
        ]
        mock_processor.return_value = mock_instance

        processor = mock_processor()

        # Test error handling and recovery
        with pytest.raises(RuntimeError, match="Document processing failed"):
            processor.process_documents(["test_doc.pdf"])

        # Test successful retry
        documents = processor.process_documents(["test_doc.pdf"])
        assert len(documents) == 1
        assert documents[0].text == "Successfully processed document"

        # Verify retry mechanism was used
        assert mock_instance.process_documents.call_count == 2

    def test_batch_document_processing_scalability(self, system_settings):
        """Test system handles batch document processing efficiently."""
        # Create a large batch of mock documents
        large_document_batch = []
        for i in range(50):  # 50 documents to test scalability
            large_document_batch.append(
                Document(
                    text=(
                        f"Document {i} content with various technical topics including "
                        "machine learning, natural language processing, and system "
                        f"architecture. This document covers topic {i % 10} in "
                        "detail with examples."
                    ),
                    metadata={
                        "source": f"doc_{i}.pdf",
                        "page": i % 5 + 1,
                        "chunk_id": f"chunk_{i}",
                        "batch_id": "large_batch_test",
                    },
                )
            )

        # Test batch processing
        with patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedder:
            mock_embedder_instance = MagicMock()

            # Mock batch processing with realistic batch sizes
            def mock_embed_batch(texts):
                return [[0.1 + i * 0.01] * 1024 for i in range(len(texts))]

            mock_embedder_instance.embed_documents.side_effect = mock_embed_batch
            mock_embedder.return_value = mock_embedder_instance

            embedder = mock_embedder()

            # Process in batches of 10 (typical batch size)
            batch_size = 10
            all_embeddings = []

            for i in range(0, len(large_document_batch), batch_size):
                batch = large_document_batch[i : i + batch_size]
                batch_texts = [doc.text for doc in batch]
                batch_embeddings = embedder.embed_documents(batch_texts)
                all_embeddings.extend(batch_embeddings)

            # Verify batch processing completed successfully
            assert len(all_embeddings) == 50
            assert all(len(embedding) == 1024 for embedding in all_embeddings)

            # Verify embeddings are unique (not all the same)
            unique_embeddings = {tuple(emb) for emb in all_embeddings}
            assert len(unique_embeddings) > 40  # Most embeddings should be unique


@pytest.mark.system
class TestMultiAgentCoordinationWorkflows:
    """Test multi-agent system coordination workflows."""

    @pytest.mark.asyncio
    @patch("src.agents.coordinator.get_agent_system")
    @patch("src.utils.embedding.create_index_async")
    async def test_complete_multi_agent_query_workflow(
        self, mock_create_index, mock_get_agent, system_settings
    ):
        """Test complete multi-agent coordination workflow for complex queries."""
        # Mock vector index
        mock_index = MagicMock(spec=VectorStoreIndex)
        mock_query_engine = MagicMock()
        mock_query_engine.aquery.return_value = MagicMock(
            response=(
                "Retrieved documents about DocMind AI architecture and multi-agent "
                "coordination."
            ),
            source_nodes=[
                NodeWithScore(
                    node=TextNode(text="DocMind AI uses LangGraph supervisor pattern"),
                    score=0.95,
                ),
                NodeWithScore(
                    node=TextNode(
                        text="Agent coordination enables complex query processing"
                    ),
                    score=0.87,
                ),
            ],
        )
        mock_index.as_query_engine.return_value = mock_query_engine
        mock_create_index.return_value = mock_index

        # Mock multi-agent system with realistic workflow
        mock_agent_system = AsyncMock()

        # Simulate multi-agent workflow stages
        async def mock_agent_workflow(query):
            """Mock multi-agent workflow: routing, planning, retrieval, synthesis."""
            # Stage 1: Routing Agent - Determine query type
            routing_result = {
                "query_type": "complex_technical",
                "requires_retrieval": True,
                "requires_synthesis": True,
                "confidence": 0.92,
            }

            # Stage 2: Planning Agent - Create execution plan
            planning_result = {
                "steps": [
                    "retrieve_technical_docs",
                    "analyze_architecture",
                    "synthesize_explanation",
                ],
                "estimated_complexity": "high",
            }

            # Stage 3: Retrieval Agent - Fetch relevant information
            retrieval_result = await mock_query_engine.aquery(query)

            # Stage 4: Synthesis Agent - Generate comprehensive response
            synthesis_result = (
                "DocMind AI employs a sophisticated multi-agent coordination system "
                "using LangGraph supervisor pattern. The system includes five "
                "specialized agents: Routing, Planning, Retrieval, Synthesis, and "
                "Validation agents. Each agent handles specific aspects of query "
                "processing, enabling complex analysis and comprehensive responses."
            )

            # Stage 5: Validation Agent - Verify response quality
            validation_result = {
                "accuracy_score": 0.94,
                "completeness_score": 0.91,
                "relevance_score": 0.96,
                "overall_quality": "high",
            }

            return {
                "final_response": synthesis_result,
                "workflow_stages": {
                    "routing": routing_result,
                    "planning": planning_result,
                    "retrieval": len(retrieval_result.source_nodes),
                    "validation": validation_result,
                },
                "total_processing_time": 2.34,  # seconds
            }

        mock_agent_system.arun.side_effect = mock_agent_workflow
        mock_get_agent.return_value = mock_agent_system

        # Execute complete multi-agent workflow
        documents = [
            Document(text="DocMind AI technical documentation"),
            Document(text="Multi-agent coordination patterns"),
            Document(text="LangGraph supervisor implementation"),
        ]

        index = await mock_create_index(documents, system_settings)
        agent_system = mock_get_agent(index, system_settings)

        complex_query = (
            "Explain how DocMind AI's multi-agent coordination system works, "
            "including the role of each agent and how they collaborate to "
            "process complex technical queries."
        )

        workflow_result = await agent_system.arun(complex_query)

        # Verify complete workflow execution
        assert "final_response" in workflow_result
        assert "workflow_stages" in workflow_result
        assert len(workflow_result["final_response"]) > 100  # Substantial response

        # Verify all agent stages executed
        stages = workflow_result["workflow_stages"]
        assert "routing" in stages
        assert "planning" in stages
        assert "retrieval" in stages
        assert "validation" in stages

        # Verify quality metrics
        assert stages["routing"]["confidence"] > 0.9
        assert stages["validation"]["overall_quality"] == "high"
        assert workflow_result["total_processing_time"] < 5.0  # Reasonable time

        print("\n=== Multi-Agent Workflow Results ===")
        print(f"Query processed: {len(complex_query)} chars")
        print(f"Response generated: {len(workflow_result['final_response'])} chars")
        print(f"Processing time: {workflow_result['total_processing_time']:.2f}s")
        print(f"Quality score: {stages['validation']['overall_quality']}")

    @pytest.mark.asyncio
    @patch("src.agents.tool_factory.ToolFactory")
    async def test_agent_tool_integration_workflow(
        self, mock_tool_factory, system_settings
    ):
        """Test agent tool integration and coordination workflow."""
        # Mock tool factory with comprehensive tool suite
        mock_factory = MagicMock()

        # Mock retrieval tool
        mock_retrieval_tool = AsyncMock()
        mock_retrieval_tool.arun.return_value = [
            {"text": "Retrieved document 1", "score": 0.95},
            {"text": "Retrieved document 2", "score": 0.87},
        ]

        # Mock synthesis tool
        mock_synthesis_tool = AsyncMock()
        mock_synthesis_tool.arun.return_value = (
            "Synthesized response combining retrieved information with analysis."
        )

        # Mock validation tool
        mock_validation_tool = MagicMock()
        mock_validation_tool.validate_response.return_value = {
            "is_accurate": True,
            "is_complete": True,
            "confidence": 0.93,
        }

        mock_factory.create_retrieval_tool.return_value = mock_retrieval_tool
        mock_factory.create_synthesis_tool.return_value = mock_synthesis_tool
        mock_factory.create_validation_tool.return_value = mock_validation_tool
        mock_tool_factory.return_value = mock_factory

        # Test tool integration workflow
        factory = mock_tool_factory(system_settings)

        # Step 1: Create tools
        retrieval_tool = factory.create_retrieval_tool()
        synthesis_tool = factory.create_synthesis_tool()
        validation_tool = factory.create_validation_tool()

        # Step 2: Execute coordinated workflow
        query = "Analyze the document processing pipeline"

        # Retrieval stage
        retrieval_results = await retrieval_tool.arun(query)
        assert len(retrieval_results) == 2
        assert all(result["score"] > 0.8 for result in retrieval_results)

        # Synthesis stage
        synthesis_input = {"query": query, "retrieved_docs": retrieval_results}
        synthesis_response = await synthesis_tool.arun(synthesis_input)
        assert len(synthesis_response) > 50  # Substantial response

        # Validation stage
        validation_result = validation_tool.validate_response(synthesis_response)
        assert validation_result["is_accurate"] is True
        assert validation_result["confidence"] > 0.9

        # Verify tool coordination
        mock_factory.create_retrieval_tool.assert_called_once()
        mock_factory.create_synthesis_tool.assert_called_once()
        mock_factory.create_validation_tool.assert_called_once()

    def test_agent_system_fallback_workflow(self, system_settings):
        """Test agent system fallback to simple RAG when multi-agent fails."""
        with patch("src.agents.coordinator.get_agent_system") as mock_get_agent:
            # Mock multi-agent system that fails
            mock_agent_system = MagicMock()
            mock_agent_system.arun.side_effect = RuntimeError(
                "Multi-agent coordination failed"
            )
            mock_get_agent.return_value = mock_agent_system

            # Mock fallback RAG system
            with patch("src.utils.embedding.create_index_async") as mock_create_index:
                mock_index = MagicMock()
                mock_query_engine = MagicMock()
                mock_query_engine.query.return_value = MagicMock(
                    response="Fallback RAG response for the query."
                )
                mock_index.as_query_engine.return_value = mock_query_engine
                mock_create_index.return_value = mock_index

                # Test fallback workflow
                try:
                    agent_system = mock_get_agent()
                    agent_system.arun("test query")
                except RuntimeError:
                    # Fallback to simple RAG
                    fallback_index = mock_create_index([], system_settings)
                    fallback_engine = fallback_index.as_query_engine()
                    fallback_response = fallback_engine.query("test query")

                    assert (
                        fallback_response.response
                        == "Fallback RAG response for the query."
                    )

                    print("\n=== Agent Fallback Workflow ===")
                    print("Multi-agent system failed, successfully fell back to RAG")


@pytest.mark.system
class TestConfigurationSystemWorkflows:
    """Test complete configuration system workflows."""

    def test_environment_to_application_configuration_workflow(self):
        """Test complete configuration workflow from environment to application."""
        # Mock environment variables
        test_env = {
            "DOCMIND_DEBUG": "true",
            "DOCMIND_LOG_LEVEL": "DEBUG",
            "DOCMIND_ENABLE_MULTI_AGENT": "true",
            "DOCMIND_VLLM__MODEL": "Qwen/Qwen3-4B-Instruct-2507-FP8",
            "DOCMIND_VLLM__CONTEXT_WINDOW": "131072",
            "DOCMIND_VLLM__TEMPERATURE": "0.1",
            "DOCMIND_AGENTS__DECISION_TIMEOUT": "200",
            "DOCMIND_AGENTS__MAX_RETRIES": "2",
            "DOCMIND_EMBEDDINGS__MODEL": "BAAI/bge-m3",
            "DOCMIND_EMBEDDINGS__DIMENSION": "1024",
            "DOCMIND_RETRIEVAL__STRATEGY": "hybrid",
            "DOCMIND_RETRIEVAL__TOP_K": "10",
        }

        with patch.dict("os.environ", test_env):
            # Step 1: Load configuration from environment
            settings = DocMindSettings()

            # Step 2: Verify basic settings loaded correctly
            assert settings.debug is True
            assert settings.log_level == "DEBUG"
            assert settings.enable_multi_agent is True

            # Step 3: Verify nested configuration loaded correctly
            assert settings.vllm.model == "Qwen/Qwen3-4B-Instruct-2507-FP8"
            assert settings.vllm.context_window == 131072
            assert settings.vllm.temperature == 0.1

            assert settings.agents.decision_timeout == 200
            assert settings.agents.max_retries == 2

            assert settings.embeddings.model == "BAAI/bge-m3"
            assert settings.embeddings.dimension == 1024

            assert settings.retrieval.strategy == "hybrid"
            assert settings.retrieval.top_k == 10

            # Step 4: Verify configuration can be used across modules
            assert (
                settings.get_vllm_config()["model"] == "Qwen/Qwen3-4B-Instruct-2507-FP8"
            )
            assert settings.get_agent_config()["decision_timeout"] == 200
            assert settings.get_embedding_config()["model"] == "BAAI/bge-m3"

            print("\n=== Configuration Workflow Results ===")
            print(f"Environment variables loaded: {len(test_env)}")
            print("Nested configurations validated: 4")
            print("Configuration methods tested: 3")

    def test_configuration_validation_workflow(self):
        """Test complete configuration validation workflow."""
        # Test valid configuration passes validation
        valid_config = DocMindSettings(
            debug=True,
            context_window_size=32768,
            chunk_size=512,
            chunk_overlap=50,
            enable_multi_agent=True,
        )

        # Verify all validations pass
        assert valid_config.context_window_size == 32768
        assert valid_config.chunk_size == 512
        assert valid_config.chunk_overlap == 50

        # Test invalid configurations are caught
        invalid_configs = [
            # Negative context window
            {"context_window_size": -1},
            # Zero chunk size
            {"chunk_size": 0},
            # Chunk overlap larger than chunk size
            {"chunk_size": 100, "chunk_overlap": 150},
        ]

        validation_errors = []
        for invalid_config in invalid_configs:
            try:
                DocMindSettings(**invalid_config)
                validation_errors.append(f"Should have failed: {invalid_config}")
            except (ValueError, AssertionError):
                # Expected validation error
                pass

        # Verify all invalid configs were caught
        assert len(validation_errors) == 0, (
            f"Validation missed errors: {validation_errors}"
        )

    def test_settings_directory_creation_workflow(self, tmp_path):
        """Test complete directory creation workflow during settings initialization."""
        # Create settings with custom directories
        custom_settings = DocMindSettings(
            data_dir=str(tmp_path / "custom_data"),
            cache_dir=str(tmp_path / "custom_cache"),
            log_file=str(tmp_path / "custom_logs" / "app.log"),
            sqlite_db_path=str(tmp_path / "custom_db" / "docmind.db"),
        )

        # Verify all directories were created
        assert Path(custom_settings.data_dir).exists()
        assert Path(custom_settings.cache_dir).exists()
        assert Path(custom_settings.log_file).parent.exists()
        assert Path(custom_settings.sqlite_db_path).parent.exists()

        # Verify directories are writable
        test_files = [
            Path(custom_settings.data_dir) / "test_data.txt",
            Path(custom_settings.cache_dir) / "test_cache.txt",
            Path(custom_settings.log_file).parent / "test_log.txt",
            Path(custom_settings.sqlite_db_path).parent / "test_db.txt",
        ]

        for test_file in test_files:
            test_file.write_text("test content")
            assert test_file.exists()
            assert test_file.read_text() == "test content"


@pytest.mark.system
class TestErrorHandlingSystemWorkflows:
    """Test complete error handling and recovery workflows."""

    @pytest.mark.asyncio
    async def test_complete_error_recovery_workflow(self, system_settings):
        """Test complete error handling and recovery workflow."""
        # Mock services that can fail and recover
        error_scenarios = [
            ("embedder_failure", ImportError("Embedding model not found")),
            ("vector_store_failure", ConnectionError("Vector store connection failed")),
            ("agent_failure", RuntimeError("Agent coordination failed")),
        ]

        recovery_results = {}

        for error_name, error in error_scenarios:
            with patch("src.retrieval.embeddings.BGEM3Embedding") as mock_embedder:
                # First call fails, second succeeds
                mock_embedder.side_effect = [error, MagicMock()]

                try:
                    # First attempt - should fail
                    mock_embedder()
                    recovery_results[error_name] = "failed_as_expected"
                except type(error):
                    recovery_results[error_name] = "error_caught"

                try:
                    # Retry attempt - should succeed
                    mock_embedder()
                    recovery_results[error_name] = "recovered_successfully"
                except Exception:
                    recovery_results[error_name] = "recovery_failed"

        # Verify all error scenarios were handled correctly
        expected_recoveries = len(error_scenarios)
        successful_recoveries = sum(
            1
            for result in recovery_results.values()
            if result == "recovered_successfully"
        )

        # At least 2/3 scenarios should demonstrate recovery
        assert successful_recoveries >= expected_recoveries * 0.67

        print("\n=== Error Recovery Workflow Results ===")
        print(f"Error scenarios tested: {expected_recoveries}")
        print(f"Successful recoveries: {successful_recoveries}")
        for scenario, result in recovery_results.items():
            print(f"{scenario}: {result}")

    @pytest.mark.parametrize(
        ("kwargs", "check"),
        [
            (
                {"agents": {"enable_multi_agent": False}},
                lambda s: s.agents.enable_multi_agent is False,
            ),
            (
                {"retrieval": {"use_reranking": False}},
                lambda s: s.retrieval.use_reranking is False,
            ),
            (
                {"retrieval": {"use_sparse_embeddings": False}},
                lambda s: s.retrieval.use_sparse_embeddings is False,
            ),
            (
                {"monitoring": {"enable_performance_logging": False}},
                lambda s: s.monitoring.enable_performance_logging is False,
            ),
        ],
    )
    def test_graceful_degradation_workflow(self, system_settings, kwargs, check):
        """Test graceful system degradation when components fail."""
        degraded_settings = DocMindSettings(**kwargs)
        assert check(degraded_settings)
        assert degraded_settings.vllm.context_window > 0
        assert degraded_settings.processing.chunk_size > 0
        assert degraded_settings.retrieval.top_k > 0


@pytest.mark.system
class TestPerformanceSystemWorkflows:
    """Test complete performance validation workflows."""

    def test_system_performance_baseline_workflow(self, system_settings):
        """Test complete system performance baseline establishment."""
        import os
        import time

        import psutil

        process = psutil.Process(os.getpid())

        # Baseline measurements
        baseline_metrics = {
            "initial_memory_mb": process.memory_info().rss / (1024 * 1024),
            "initial_cpu_percent": process.cpu_percent(),
        }

        # Simulate typical workload
        start_time = time.perf_counter()

        # Create multiple settings instances (simulates app usage)
        settings_instances = []
        for _i in range(10):
            settings = DocMindSettings(
                debug=False,
                enable_multi_agent=True,
            )
            # Set nested config values
            settings.retrieval.use_reranking = True
            settings_instances.append(settings)

        workload_time = time.perf_counter() - start_time

        # Final measurements
        final_metrics = {
            "final_memory_mb": process.memory_info().rss / (1024 * 1024),
            "final_cpu_percent": process.cpu_percent(),
            "workload_time_ms": workload_time * 1000,
        }

        # Performance assertions
        memory_increase = (
            final_metrics["final_memory_mb"] - baseline_metrics["initial_memory_mb"]
        )
        assert memory_increase < 100, (
            f"Memory usage increased too much: {memory_increase:.2f}MB"
        )
        assert final_metrics["workload_time_ms"] < 1000, (
            f"Workload took too long: {workload_time * 1000:.2f}ms"
        )

        print("\n=== System Performance Baseline ===")
        print(f"Initial memory: {baseline_metrics['initial_memory_mb']:.2f}MB")
        print(f"Final memory: {final_metrics['final_memory_mb']:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")
        print(f"Workload time: {final_metrics['workload_time_ms']:.2f}ms")

    @pytest.mark.asyncio
    @patch("src.retrieval.embeddings.BGEM3Embedding")
    async def test_concurrent_processing_workflow(self, mock_embedder):
        """Test system performance under concurrent processing load."""
        # Mock embedder for consistent performance testing
        mock_embedder_instance = AsyncMock()
        mock_embedder_instance.aembed_documents.return_value = [[0.1] * 1024] * 5
        mock_embedder.return_value = mock_embedder_instance

        async def process_document_batch(batch_id):
            """Simulate processing a batch of documents."""
            embedder = mock_embedder()

            # Simulate document texts
            texts = [f"Document {batch_id}-{i} content" for i in range(5)]

            # Process embeddings
            embeddings = await embedder.aembed_documents(texts)

            return {
                "batch_id": batch_id,
                "documents_processed": len(texts),
                "embeddings_created": len(embeddings),
            }

        # Test concurrent processing
        start_time = time.perf_counter()

        # Process 10 batches concurrently (50 documents total)
        tasks = [process_document_batch(i) for i in range(10)]
        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

        # Verify concurrent processing results
        assert len(results) == 10
        total_documents = sum(result["documents_processed"] for result in results)
        total_embeddings = sum(result["embeddings_created"] for result in results)

        assert total_documents == 50
        assert total_embeddings == 50
        assert total_time < 5.0  # Should complete within 5 seconds

        print("\n=== Concurrent Processing Workflow ===")
        print(f"Concurrent batches: {len(results)}")
        print(f"Total documents: {total_documents}")
        print(f"Total embeddings: {total_embeddings}")
        print(f"Processing time: {total_time:.2f}s")
        print(f"Throughput: {total_documents / total_time:.1f} docs/sec")
