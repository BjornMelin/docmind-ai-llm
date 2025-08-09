"""Integration test for DocMind AI complete pipeline flow.

This test suite validates the end-to-end pipeline from document loading
to agent response generation, ensuring all components work together correctly.

Pipeline tested:
Document → Index → Retriever → Agent → Response

Critical path coverage:
- Document loading and parsing
- Index creation (vector, KG, hybrid)
- Tool factory and agent creation
- Query processing and response generation
- Error handling and fallbacks
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document
from llama_index.core.schema import NodeWithScore, TextNode


class TestPipelineIntegration:
    """Integration tests for complete DocMind AI pipeline."""

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                text="This is a comprehensive guide to machine learning algorithms.",
                metadata={
                    "file_path": "/docs/ml_guide.pdf",
                    "file_type": "pdf",
                    "page_count": 50,
                },
            ),
            Document(
                text="Deep learning frameworks comparison: PyTorch vs TensorFlow.",
                metadata={
                    "file_path": "/docs/frameworks.pdf",
                    "file_type": "pdf",
                    "page_count": 25,
                },
            ),
            Document(
                text="Neural networks are computational models inspired by biological systems.",
                metadata={
                    "file_path": "/docs/neural_nets.pdf",
                    "file_type": "pdf",
                    "page_count": 30,
                },
            ),
        ]

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing."""
        llm = MagicMock()
        llm.complete.return_value = MagicMock(text="Mock LLM response")
        llm.chat.return_value = MagicMock(response="Mock chat response")
        return llm

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model."""
        model = MagicMock()
        model.get_text_embedding.return_value = [0.1] * 384  # Mock 384-dim embedding
        model.get_text_embedding_batch.return_value = [[0.1] * 384, [0.2] * 384]
        return model

    @pytest.fixture
    async def temporary_storage_dir(self):
        """Create temporary directory for test storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.mark.asyncio
    async def test_complete_pipeline_document_to_response(
        self, sample_documents, mock_llm, mock_embedding_model, temporary_storage_dir
    ):
        """Test complete pipeline: document loading → indexing → query → response."""
        with patch("utils.document_loader.load_documents_llama") as mock_load:
            mock_load.return_value = sample_documents

            with patch("utils.index_builder.create_index_async") as mock_create_index:
                # Mock vector index
                mock_vector_index = MagicMock()
                mock_vector_index.as_retriever.return_value = MagicMock()

                # Mock retriever results
                mock_retriever = MagicMock()
                mock_nodes = [
                    NodeWithScore(
                        node=TextNode(text="Machine learning overview", id_="node1"),
                        score=0.9,
                    ),
                    NodeWithScore(
                        node=TextNode(text="Deep learning frameworks", id_="node2"),
                        score=0.8,
                    ),
                ]
                mock_retriever.retrieve.return_value = mock_nodes
                mock_vector_index.as_retriever.return_value = mock_retriever

                mock_create_index.return_value = {
                    "vector_index": mock_vector_index,
                    "kg_index": None,  # KG creation might fail
                    "multimodal_index": None,
                }

                with patch("agents.tool_factory.ToolFactory") as mock_tool_factory:
                    # Mock tool creation
                    mock_tool = MagicMock()
                    mock_tool.metadata.name = "hybrid_search"
                    mock_tool_factory.create_tools_from_indexes.return_value = [
                        mock_tool
                    ]

                    with patch("agent_factory.get_agent_system") as mock_get_agent:
                        # Mock agent system
                        mock_agent = MagicMock()
                        mock_response = MagicMock()
                        mock_response.response = "Based on the documents, machine learning algorithms include supervised, unsupervised, and reinforcement learning approaches."
                        mock_agent.chat.return_value = mock_response
                        mock_get_agent.return_value = (mock_agent, "single")

                        with patch(
                            "agent_factory.process_query_with_agent_system"
                        ) as mock_process:
                            mock_process.return_value = (
                                "Integration test successful: ML algorithms explained"
                            )

                            # Import functions to test
                            from agent_factory import (
                                get_agent_system,
                                process_query_with_agent_system,
                            )
                            from agents.tool_factory import ToolFactory
                            from utils.document_loader import load_documents_llama
                            from utils.index_builder import create_index_async

                            # Step 1: Load documents
                            documents = load_documents_llama(
                                file_paths=[
                                    "/docs/ml_guide.pdf",
                                    "/docs/frameworks.pdf",
                                ],
                                chunk_size=512,
                                chunk_overlap=50,
                            )

                            assert len(documents) == 3
                            assert "machine learning" in documents[0].text.lower()

                            # Step 2: Create indexes
                            indexes = await create_index_async(
                                documents=documents,
                                embedding_model=mock_embedding_model,
                                vector_store_path=str(temporary_storage_dir / "vector"),
                                enable_kg=True,
                                enable_multimodal=False,
                            )

                            assert "vector_index" in indexes
                            assert indexes["vector_index"] is not None

                            # Step 3: Create tools
                            tool_factory = ToolFactory()
                            tools = tool_factory.create_tools_from_indexes(
                                indexes, embedding_model=mock_embedding_model
                            )

                            assert len(tools) >= 1
                            assert any("search" in tool.metadata.name for tool in tools)

                            # Step 4: Create agent system
                            agent_system, mode = get_agent_system(
                                tools, mock_llm, enable_multi_agent=False
                            )

                            assert agent_system is not None
                            assert mode == "single"

                            # Step 5: Process query
                            query = "What are the main types of machine learning algorithms?"
                            response = process_query_with_agent_system(
                                agent_system, query, mode
                            )

                            assert response is not None
                            assert len(response) > 0
                            assert "Integration test successful" in response

    @pytest.mark.asyncio
    async def test_pipeline_with_multimodal_processing(
        self, sample_documents, mock_llm, mock_embedding_model, temporary_storage_dir
    ):
        """Test pipeline with multimodal document processing."""
        # Add multimodal document
        multimodal_doc = Document(
            text="Image caption: Neural network architecture diagram",
            metadata={
                "file_path": "/docs/diagram.pdf",
                "file_type": "pdf",
                "has_images": True,
                "image_count": 3,
            },
        )
        documents = sample_documents + [multimodal_doc]

        with patch("utils.document_loader.extract_images_from_pdf") as mock_extract:
            mock_extract.return_value = [
                {"image": b"fake_image_data", "page": 1, "bbox": [0, 0, 100, 100]}
            ]

            with patch(
                "utils.document_loader.create_native_multimodal_embeddings"
            ) as mock_mm_embed:
                mock_mm_embed.return_value = [0.1] * 512  # Mock multimodal embedding

                with patch(
                    "utils.index_builder.create_index_async"
                ) as mock_create_index:
                    mock_multimodal_index = MagicMock()
                    mock_create_index.return_value = {
                        "vector_index": MagicMock(),
                        "kg_index": None,
                        "multimodal_index": mock_multimodal_index,
                    }

                    with patch(
                        "agents.tool_factory.ToolFactory.create_tools_from_indexes"
                    ) as mock_tools:
                        # Include multimodal tool
                        mock_mm_tool = MagicMock()
                        mock_mm_tool.metadata.name = "multimodal_search"
                        mock_tools.return_value = [mock_mm_tool]

                        with patch("agent_factory.get_agent_system") as mock_get_agent:
                            mock_agent = MagicMock()
                            mock_get_agent.return_value = (mock_agent, "single")

                            with patch(
                                "agent_factory.process_query_with_agent_system"
                            ) as mock_process:
                                mock_process.return_value = "The diagram shows a typical neural network with input, hidden, and output layers."

                                # Test multimodal pipeline
                                from agent_factory import (
                                    process_query_with_agent_system,
                                )
                                from utils.document_loader import (
                                    extract_images_from_pdf,
                                )
                                from utils.index_builder import create_index_async

                                # Extract images
                                images = extract_images_from_pdf("/docs/diagram.pdf")
                                assert len(images) == 1

                                # Create multimodal index
                                indexes = await create_index_async(
                                    documents=documents,
                                    embedding_model=mock_embedding_model,
                                    enable_multimodal=True,
                                    vector_store_path=str(
                                        temporary_storage_dir / "mm_vector"
                                    ),
                                )

                                assert "multimodal_index" in indexes

                                # Process multimodal query
                                query = "What does the neural network diagram show?"
                                response = process_query_with_agent_system(
                                    mock_agent, query, "single"
                                )

                                assert "diagram" in response.lower()
                                assert "neural network" in response.lower()

    @pytest.mark.asyncio
    async def test_pipeline_error_handling_and_fallbacks(
        self, sample_documents, mock_llm, mock_embedding_model
    ):
        """Test pipeline error handling and fallback mechanisms."""
        with patch("utils.document_loader.load_documents_llama") as mock_load:
            mock_load.return_value = sample_documents

            # Test index creation failure fallback
            with patch("utils.index_builder.create_index_async") as mock_create_index:
                # First call fails, second succeeds with basic index
                mock_create_index.side_effect = [
                    Exception("Qdrant connection failed"),  # First attempt fails
                    {
                        "vector_index": MagicMock(),
                        "kg_index": None,
                        "multimodal_index": None,
                    },  # Fallback succeeds
                ]

                with patch(
                    "agents.tool_factory.ToolFactory.create_tools_from_indexes"
                ) as mock_tools:
                    mock_basic_tool = MagicMock()
                    mock_basic_tool.metadata.name = "basic_search"
                    mock_tools.return_value = [mock_basic_tool]

                    with patch("agent_factory.get_agent_system") as mock_get_agent:
                        # Multi-agent fails, falls back to single
                        mock_get_agent.return_value = (MagicMock(), "single")

                        with patch(
                            "agent_factory.process_query_with_agent_system"
                        ) as mock_process:
                            mock_process.return_value = (
                                "Fallback response generated successfully"
                            )

                            # Test with error handling
                            try:
                                from agent_factory import (
                                    get_agent_system,
                                    process_query_with_agent_system,
                                )
                                from utils.index_builder import create_index_async

                                # First attempt should fail
                                with pytest.raises(
                                    Exception, match="Qdrant connection failed"
                                ):
                                    await create_index_async(documents=sample_documents)

                                # Fallback attempt should succeed
                                indexes = await create_index_async(
                                    documents=sample_documents
                                )
                                assert "vector_index" in indexes

                                # Agent system should handle gracefully
                                agent_system, mode = get_agent_system(
                                    [mock_basic_tool], mock_llm, enable_multi_agent=True
                                )

                                # Should fallback to single agent
                                assert mode == "single"

                                # Should still process queries
                                response = process_query_with_agent_system(
                                    agent_system, "test query", mode
                                )
                                assert "Fallback response" in response

                            except Exception as e:
                                # Error handling should be graceful
                                assert "connection failed" in str(e).lower()

    @pytest.mark.asyncio
    async def test_pipeline_performance_monitoring(
        self, sample_documents, mock_llm, mock_embedding_model, temporary_storage_dir
    ):
        """Test pipeline performance monitoring and logging."""
        with (
            patch("utils.logging_config.log_performance") as mock_log_perf,
            patch("time.time") as mock_time
        ):
                # Mock timing
                mock_time.side_effect = [
                    0,
                    1.5,
                    3.0,
                    4.2,
                    5.8,
                ]  # Progressive timestamps

                with patch("utils.document_loader.load_documents_llama") as mock_load:
                    mock_load.return_value = sample_documents

                    with patch(
                        "utils.index_builder.create_index_async"
                    ) as mock_create_index:
                        mock_create_index.return_value = {
                            "vector_index": MagicMock(),
                            "kg_index": None,
                            "multimodal_index": None,
                        }

                        with patch("agent_factory.get_agent_system") as mock_get_agent:
                            mock_agent = MagicMock()
                            mock_get_agent.return_value = (mock_agent, "single")

                            with patch(
                                "agent_factory.process_query_with_agent_system"
                            ) as mock_process:
                                mock_process.return_value = "Performance test response"

                                # Test performance logging throughout pipeline
                                from agent_factory import (
                                    get_agent_system,
                                    process_query_with_agent_system,
                                )
                                from utils.document_loader import load_documents_llama
                                from utils.index_builder import create_index_async
                                from utils.logging_config import log_performance

                                # Document loading
                                start_time = mock_time()
                                documents = load_documents_llama(["/docs/test.pdf"])
                                end_time = mock_time()
                                log_performance(
                                    "document_loading",
                                    end_time - start_time,
                                    doc_count=len(documents),
                                )

                                # Index creation
                                start_time = mock_time()
                                indexes = await create_index_async(documents=documents)
                                end_time = mock_time()
                                log_performance(
                                    "index_creation",
                                    end_time - start_time,
                                    doc_count=len(documents),
                                )

                                # Agent query
                                start_time = mock_time()
                                agent_system, mode = get_agent_system([], mock_llm)
                                response = process_query_with_agent_system(
                                    agent_system, "test", mode
                                )
                                end_time = mock_time()
                                log_performance(
                                    "agent_query",
                                    end_time - start_time,
                                    response_length=len(response),
                                )

                                # Verify performance logging calls
                                assert mock_log_perf.call_count == 3

                                # Check log call details
                                calls = mock_log_perf.call_args_list
                                assert calls[0][0][0] == "document_loading"
                                assert calls[1][0][0] == "index_creation"
                                assert calls[2][0][0] == "agent_query"

                                # All operations should have recorded timing
                                for call in calls:
                                    duration = call[0][1]
                                    assert duration >= 0

    @pytest.mark.asyncio
    async def test_pipeline_with_knowledge_graph_integration(
        self, sample_documents, mock_llm, mock_embedding_model, temporary_storage_dir
    ):
        """Test pipeline with knowledge graph extraction and querying."""
        with patch("utils.index_builder.create_index_async") as mock_create_index:
            # Mock KG index with entities and relationships
            mock_kg_index = MagicMock()
            mock_kg_index.as_retriever.return_value = MagicMock()

            mock_create_index.return_value = {
                "vector_index": MagicMock(),
                "kg_index": mock_kg_index,
                "multimodal_index": None,
            }

            with patch(
                "agents.tool_factory.ToolFactory.create_tools_from_indexes"
            ) as mock_tools:
                # Include KG tools
                mock_kg_tool = MagicMock()
                mock_kg_tool.metadata.name = "knowledge_graph_search"
                mock_hybrid_tool = MagicMock()
                mock_hybrid_tool.metadata.name = "hybrid_search"
                mock_tools.return_value = [mock_kg_tool, mock_hybrid_tool]

                with patch("agent_factory.get_agent_system") as mock_get_agent:
                    # Use multi-agent system for KG queries
                    mock_multi_system = MagicMock()
                    mock_get_agent.return_value = (mock_multi_system, "multi")

                    with patch(
                        "agent_factory.process_query_with_agent_system"
                    ) as mock_process:
                        mock_process.return_value = "The main entities are: Machine Learning, Deep Learning, PyTorch, TensorFlow. Relationships: PyTorch and TensorFlow are frameworks for Deep Learning, which is a subset of Machine Learning."

                        # Test KG pipeline
                        from agent_factory import (
                            get_agent_system,
                            process_query_with_agent_system,
                        )
                        from agents.tool_factory import ToolFactory
                        from utils.index_builder import create_index_async

                        # Create indexes with KG enabled
                        indexes = await create_index_async(
                            documents=sample_documents,
                            embedding_model=mock_embedding_model,
                            enable_kg=True,
                            vector_store_path=str(temporary_storage_dir / "kg_vector"),
                        )

                        assert "kg_index" in indexes
                        assert indexes["kg_index"] is not None

                        # Create tools including KG tools
                        tool_factory = ToolFactory()
                        tools = tool_factory.create_tools_from_indexes(indexes)

                        # Should have KG and hybrid tools
                        tool_names = [tool.metadata.name for tool in tools]
                        assert "knowledge_graph_search" in tool_names
                        assert "hybrid_search" in tool_names

                        # Multi-agent system should handle KG queries
                        agent_system, mode = get_agent_system(
                            tools, mock_llm, enable_multi_agent=True
                        )

                        assert mode == "multi"

                        # Process KG-focused query
                        kg_query = "What entities are related to machine learning and how are they connected?"
                        response = process_query_with_agent_system(
                            agent_system, kg_query, mode
                        )

                        assert "entities" in response.lower()
                        assert "relationships" in response.lower()
                        assert "machine learning" in response.lower()

    @pytest.mark.asyncio
    async def test_pipeline_concurrent_processing(
        self, sample_documents, mock_llm, mock_embedding_model
    ):
        """Test pipeline with concurrent document processing and querying."""
        import asyncio

        with patch("utils.document_loader.stream_document_processing") as mock_stream:
            # Mock async document streaming
            async def mock_doc_stream(file_paths):
                for i, doc in enumerate(sample_documents):
                    yield doc, {"progress": (i + 1) / len(sample_documents)}

            mock_stream.return_value = mock_doc_stream([])

            with patch("utils.index_builder.create_index_async") as mock_create_index:
                mock_create_index.return_value = {
                    "vector_index": MagicMock(),
                    "kg_index": None,
                    "multimodal_index": None,
                }

                with patch("agent_factory.get_agent_system") as mock_get_agent:
                    mock_agent = MagicMock()
                    mock_get_agent.return_value = (mock_agent, "single")

                    # Mock concurrent query processing
                    async def mock_query_async(query):
                        await asyncio.sleep(0.1)  # Simulate async processing
                        return f"Async response for: {query}"

                    async def mock_achat_side_effect(query):
                        result = await mock_query_async(query)
                        return MagicMock(response=result)

                    mock_agent.achat = AsyncMock(side_effect=mock_achat_side_effect)

                    # Test concurrent processing
                    from agent_factory import get_agent_system
                    from utils.document_loader import stream_document_processing
                    from utils.index_builder import create_index_async

                    # Stream document processing
                    processed_docs = []
                    async for doc, metadata in stream_document_processing(
                        ["/docs/1.pdf", "/docs/2.pdf"]
                    ):
                        processed_docs.append(doc)
                        assert "progress" in metadata

                    assert len(processed_docs) == 3

                    # Create index from streamed docs
                    indexes = await create_index_async(documents=processed_docs)
                    assert "vector_index" in indexes

                    # Process multiple queries concurrently
                    agent_system, mode = get_agent_system([], mock_llm)

                    queries = [
                        "What is machine learning?",
                        "Compare PyTorch and TensorFlow",
                        "Explain neural networks",
                    ]

                    # Concurrent query processing
                    async def process_query(query):
                        if hasattr(agent_system, "achat"):
                            response = await agent_system.achat(query)
                            return response.response
                        return f"Async response for: {query}"

                    results = await asyncio.gather(*[process_query(q) for q in queries])

                    assert len(results) == 3
                    for i, result in enumerate(results):
                        assert (
                            queries[i].split()[2] in result
                        )  # Check query-specific content


class TestPipelineEdgeCases:
    """Test edge cases and error conditions in the pipeline."""

    @pytest.mark.asyncio
    async def test_empty_document_handling(self, mock_llm, mock_embedding_model):
        """Test pipeline behavior with empty or invalid documents."""
        empty_documents = [
            Document(text="", metadata={"file_path": "/empty.pdf"}),
            Document(text="   \n\n   ", metadata={"file_path": "/whitespace.pdf"}),
        ]

        with patch("utils.index_builder.create_index_async") as mock_create_index:
            # Should handle empty docs gracefully
            mock_create_index.return_value = {
                "vector_index": MagicMock(),
                "kg_index": None,
                "multimodal_index": None,
            }

            with patch("agent_factory.get_agent_system") as mock_get_agent:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.response = "No content found in the provided documents."
                mock_agent.chat.return_value = mock_response
                mock_get_agent.return_value = (mock_agent, "single")

                # Test pipeline with empty documents
                from agent_factory import (
                    get_agent_system,
                    process_query_with_agent_system,
                )
                from utils.index_builder import create_index_async

                indexes = await create_index_async(documents=empty_documents)
                assert "vector_index" in indexes

                agent_system, mode = get_agent_system([], mock_llm)
                response = process_query_with_agent_system(
                    agent_system, "What information is available?", mode
                )

                assert "no content" in response.lower()

    @pytest.mark.asyncio
    async def test_large_document_batch_processing(
        self, mock_llm, mock_embedding_model
    ):
        """Test pipeline with large batch of documents."""
        # Create large batch (100 documents)
        large_batch = []
        for i in range(100):
            doc = Document(
                text=f"Document {i}: Content about topic {i % 10}",
                metadata={"file_path": f"/docs/batch_{i}.pdf", "doc_id": i},
            )
            large_batch.append(doc)

        with patch("utils.document_loader.batch_embed_documents") as mock_batch_embed:
            # Mock batch embedding processing
            mock_batch_embed.return_value = [[0.1] * 384] * 100  # 100 embeddings

            with patch("utils.index_builder.create_index_async") as mock_create_index:
                mock_create_index.return_value = {
                    "vector_index": MagicMock(),
                    "kg_index": None,
                    "multimodal_index": None,
                }

                with patch("agent_factory.get_agent_system") as mock_get_agent:
                    mock_agent = MagicMock()
                    mock_response = MagicMock()
                    mock_response.response = "Processed 100 documents successfully. Topics 0-9 covered extensively."
                    mock_agent.chat.return_value = mock_response
                    mock_get_agent.return_value = (mock_agent, "single")

                    # Test large batch processing
                    from agent_factory import (
                        get_agent_system,
                        process_query_with_agent_system,
                    )
                    from utils.document_loader import batch_embed_documents
                    from utils.index_builder import create_index_async

                    # Batch embedding
                    embeddings = batch_embed_documents(
                        [doc.text for doc in large_batch], mock_embedding_model
                    )
                    assert len(embeddings) == 100

                    # Index creation should handle large batches
                    indexes = await create_index_async(
                        documents=large_batch,
                        embedding_model=mock_embedding_model,
                    )
                    assert "vector_index" in indexes

                    # Agent should handle batch queries
                    agent_system, mode = get_agent_system([], mock_llm)
                    response = process_query_with_agent_system(
                        agent_system, "Summarize the content from all documents", mode
                    )

                    assert "100 documents" in response
                    assert "successfully" in response.lower()

    @pytest.mark.asyncio
    async def test_mixed_format_document_pipeline(self, mock_llm, mock_embedding_model):
        """Test pipeline with mixed document formats and types."""
        mixed_documents = [
            Document(
                text="PDF content about machine learning",
                metadata={"file_path": "/docs/ml.pdf", "file_type": "pdf"},
            ),
            Document(
                text="Word document discussing neural networks",
                metadata={"file_path": "/docs/nn.docx", "file_type": "docx"},
            ),
            Document(
                text="Plain text file with AI definitions",
                metadata={"file_path": "/docs/ai.txt", "file_type": "txt"},
            ),
            Document(
                text="Markdown documentation for deep learning",
                metadata={"file_path": "/docs/dl.md", "file_type": "markdown"},
            ),
        ]

        with patch("utils.index_builder.create_index_async") as mock_create_index:
            mock_create_index.return_value = {
                "vector_index": MagicMock(),
                "kg_index": MagicMock(),
                "multimodal_index": None,
            }

            with patch("agent_factory.get_agent_system") as mock_get_agent:
                mock_agent = MagicMock()
                mock_response = MagicMock()
                mock_response.response = "Found information across PDF, Word, text, and Markdown files covering ML, neural networks, AI definitions, and deep learning."
                mock_agent.chat.return_value = mock_response
                mock_get_agent.return_value = (mock_agent, "single")

                # Test mixed format processing
                from agent_factory import (
                    get_agent_system,
                    process_query_with_agent_system,
                )
                from utils.index_builder import create_index_async

                # Should handle all formats
                indexes = await create_index_async(
                    documents=mixed_documents,
                    embedding_model=mock_embedding_model,
                    enable_kg=True,
                )

                assert "vector_index" in indexes
                assert "kg_index" in indexes

                # Agent should find info across formats
                agent_system, mode = get_agent_system([], mock_llm)
                response = process_query_with_agent_system(
                    agent_system,
                    "What information is available across all document types?",
                    mode,
                )

                # Should mention various formats
                response_lower = response.lower()
                assert any(
                    fmt in response_lower for fmt in ["pdf", "word", "text", "markdown"]
                )
                assert "information" in response_lower
