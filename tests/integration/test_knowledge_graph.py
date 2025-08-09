"""Comprehensive tests for Knowledge Graph creation, spaCy integration, and agent tool generation.

This module tests:
- SpaCy model management (loading and auto-download)
- Knowledge Graph index creation with entity extraction
- Agent tool creation with and without KG
- Graceful fallback when dependencies missing
- Async KG creation
- Entity extraction validation
- Tool metadata configuration

Following PyTestQA-Agent standards for comprehensive testing.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import asyncio
import logging
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import Document


class TestSpaCyIntegration:
    """Test spaCy model management and integration."""

    def test_ensure_spacy_model_already_installed(self):
        """Test when spaCy model is already available."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            from utils.utils import ensure_spacy_model

            nlp = ensure_spacy_model()

            assert nlp is not None
            assert nlp == mock_nlp
            mock_load.assert_called_once_with("en_core_web_sm")

    def test_ensure_spacy_model_auto_download_success(self):
        """Test automatic model download when missing."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            # First call fails (not installed), second succeeds
            mock_load.side_effect = [OSError("Model not found"), mock_nlp]

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                from utils.utils import ensure_spacy_model

                nlp = ensure_spacy_model()

                # Verify download was attempted
                mock_run.assert_called_once()
                args = mock_run.call_args[0][0]
                assert "python" in args
                assert "-m" in args
                assert "spacy" in args
                assert "download" in args
                assert "en_core_web_sm" in args
                assert mock_run.call_args[1]["check"] is True

                # Verify model loaded after download
                assert mock_load.call_count == 2
                assert nlp == mock_nlp

    def test_ensure_spacy_model_download_failure(self):
        """Test handling of download failure."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = OSError("Model not found")

            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = subprocess.CalledProcessError(
                    returncode=1, cmd=["spacy", "download"]
                )

                from utils.utils import ensure_spacy_model

                with pytest.raises(RuntimeError, match="Failed to load or download"):
                    ensure_spacy_model()

                # Verify download was attempted
                mock_run.assert_called_once()
                # Verify both load attempts failed
                assert mock_load.call_count == 1

    def test_ensure_spacy_model_import_error(self):
        """Test handling when spaCy is not installed."""
        with patch("spacy.load") as mock_load:
            mock_load.side_effect = ImportError("No module named 'spacy'")

            from utils.utils import ensure_spacy_model

            with pytest.raises(RuntimeError, match="spaCy is not installed"):
                ensure_spacy_model()

    def test_ensure_spacy_model_custom_model_name(self):
        """Test loading custom spaCy model."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_load.return_value = mock_nlp

            from utils.utils import ensure_spacy_model

            nlp = ensure_spacy_model("en_core_web_md")

            assert nlp is not None
            mock_load.assert_called_once_with("en_core_web_md")


class TestEntityExtraction:
    """Test entity extraction functionality for Knowledge Graph."""

    def test_entity_extraction(self):
        """Test spaCy entity extraction for KG."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()

            # Create mock entities
            mock_entity_1 = MagicMock()
            mock_entity_1.text = "Apple Inc."
            mock_entity_1.label_ = "ORG"

            mock_entity_2 = MagicMock()
            mock_entity_2.text = "Steve Jobs"
            mock_entity_2.label_ = "PERSON"

            mock_doc.ents = [mock_entity_1, mock_entity_2]
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp

            from utils.utils import ensure_spacy_model

            nlp = ensure_spacy_model()
            doc = nlp("Apple Inc. was founded by Steve Jobs")

            assert len(doc.ents) == 2
            assert any(ent.label_ == "ORG" for ent in doc.ents)
            assert any(ent.label_ == "PERSON" for ent in doc.ents)
            assert any(ent.text == "Apple Inc." for ent in doc.ents)
            assert any(ent.text == "Steve Jobs" for ent in doc.ents)

    def test_entity_extraction_no_entities(self):
        """Test entity extraction when no entities found."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()
            mock_doc.ents = []  # No entities
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp

            from utils.utils import ensure_spacy_model

            nlp = ensure_spacy_model()
            doc = nlp("This text has no named entities.")

            assert len(doc.ents) == 0


class TestKnowledgeGraphIndex:
    """Test Knowledge Graph index creation and management."""

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing."""
        return [
            Document(text="Apple Inc. was founded by Steve Jobs in 1976."),
            Document(text="Microsoft was founded by Bill Gates in 1975."),
            Document(text="Google was founded by Larry Page and Sergey Brin."),
        ]

    def test_kg_index_creation_success(self, sample_docs):
        """Test successful KG index creation with entity extraction."""
        from utils.index_builder import create_index

        # Mock all dependencies
        with patch("llama_index.core.KnowledgeGraphIndex.from_documents") as mock_kg:
            mock_kg_instance = MagicMock()
            mock_kg.return_value = mock_kg_instance

            with patch("utils.utils.ensure_spacy_model") as mock_spacy:
                mock_nlp = MagicMock()
                mock_spacy.return_value = mock_nlp

                with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
                    mock_llm = MagicMock()
                    mock_ollama.return_value = mock_llm

                    with patch(
                        "utils.index_builder.setup_hybrid_qdrant"
                    ) as mock_qdrant:
                        mock_vector_store = MagicMock()
                        mock_qdrant.return_value = mock_vector_store

                        with patch(
                            "llama_index.core.VectorStoreIndex.from_documents"
                        ) as mock_vector_index:
                            mock_vector_instance = MagicMock()
                            mock_vector_index.return_value = mock_vector_instance

                            result = create_index(sample_docs, use_gpu=False)

                            # Verify KG index was created
                            assert "kg" in result
                            assert result["kg"] == mock_kg_instance

                            # Verify dependencies were called correctly
                            mock_spacy.assert_called_once_with("en_core_web_sm")
                            mock_kg.assert_called_once()

                            # Check KG creation parameters
                            kg_call_args = mock_kg.call_args
                            assert kg_call_args[0][0] == sample_docs  # documents
                            assert "llm" in kg_call_args[1]
                            assert "embed_model" in kg_call_args[1]
                            assert "extractor" in kg_call_args[1]
                            assert "max_entities" in kg_call_args[1]

    def test_kg_index_creation_spacy_failure(self, sample_docs):
        """Test KG index creation when spaCy fails."""
        from utils.index_builder import create_index

        with patch("utils.utils.ensure_spacy_model") as mock_spacy:
            mock_spacy.side_effect = RuntimeError("spaCy model download failed")

            with patch("utils.index_builder.setup_hybrid_qdrant") as mock_qdrant:
                mock_vector_store = MagicMock()
                mock_qdrant.return_value = mock_vector_store

                with patch(
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_vector_index:
                    mock_vector_instance = MagicMock()
                    mock_vector_index.return_value = mock_vector_instance

                    result = create_index(sample_docs, use_gpu=False)

                    # Should still create vector index
                    assert "vector" in result
                    assert result["vector"] is not None
                    # KG should be None due to spaCy error
                    assert result.get("kg") is None

    def test_kg_index_creation_llm_unavailable(self, sample_docs):
        """Test graceful fallback when LLM is unavailable."""
        from utils.index_builder import create_index

        with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
            mock_ollama.side_effect = Exception("Ollama not available")

            with patch("utils.utils.ensure_spacy_model") as mock_spacy:
                mock_nlp = MagicMock()
                mock_spacy.return_value = mock_nlp

                with patch("utils.index_builder.setup_hybrid_qdrant") as mock_qdrant:
                    mock_vector_store = MagicMock()
                    mock_qdrant.return_value = mock_vector_store

                    with patch(
                        "llama_index.core.VectorStoreIndex.from_documents"
                    ) as mock_vector_index:
                        mock_vector_instance = MagicMock()
                        mock_vector_index.return_value = mock_vector_instance

                        result = create_index(sample_docs, use_gpu=False)

                        # Should still create vector index
                        assert "vector" in result
                        assert result["vector"] is not None
                        # KG should be None due to LLM error
                        assert result.get("kg") is None

    @pytest.mark.asyncio
    async def test_kg_async_creation_success(self, sample_docs):
        """Test successful async KG index creation."""
        from utils.index_builder import create_index_async

        # Mock all dependencies for async
        with patch("llama_index.core.KnowledgeGraphIndex.from_documents") as mock_kg:
            mock_kg_instance = MagicMock()
            mock_kg.return_value = mock_kg_instance

            with patch("utils.utils.ensure_spacy_model") as mock_spacy:
                mock_nlp = MagicMock()
                mock_spacy.return_value = mock_nlp

                with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
                    mock_llm = MagicMock()
                    mock_ollama.return_value = mock_llm

                    with patch(
                        "utils.index_builder.setup_hybrid_qdrant_async"
                    ) as mock_qdrant_async:
                        mock_vector_store = MagicMock()
                        mock_qdrant_async.return_value = mock_vector_store

                        with patch(
                            "llama_index.core.VectorStoreIndex.from_documents"
                        ) as mock_vector_index:
                            mock_vector_instance = MagicMock()
                            mock_vector_index.return_value = mock_vector_instance

                            with patch(
                                "utils.index_builder.create_hybrid_retriever"
                            ) as mock_retriever:
                                mock_hybrid_retriever = MagicMock()
                                mock_retriever.return_value = mock_hybrid_retriever

                                with patch(
                                    "utils.index_builder.verify_rrf_configuration"
                                ) as mock_verify_rrf:
                                    mock_verify_rrf.return_value = {
                                        "issues": [],
                                        "recommendations": [],
                                    }

                                    with patch(
                                        "qdrant_client.AsyncQdrantClient"
                                    ) as mock_async_client:
                                        mock_client_instance = AsyncMock()
                                        mock_async_client.return_value = (
                                            mock_client_instance
                                        )

                                        result = await create_index_async(
                                            sample_docs, use_gpu=False
                                        )

                                        # Verify KG index was created
                                        assert "kg" in result
                                        assert result["kg"] == mock_kg_instance

                                        # Verify successful async operation
                                        # Note: Client close is called in finally block, tested separately

    @pytest.mark.asyncio
    async def test_kg_async_creation_failure(self, sample_docs):
        """Test async KG creation with failures."""
        from utils.index_builder import create_index_async

        with patch("utils.utils.ensure_spacy_model") as mock_spacy:
            mock_spacy.side_effect = Exception("spaCy error")

            with patch(
                "utils.index_builder.setup_hybrid_qdrant_async"
            ) as mock_qdrant_async:
                mock_vector_store = MagicMock()
                mock_qdrant_async.return_value = mock_vector_store

                with patch(
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_vector_index:
                    mock_vector_instance = MagicMock()
                    mock_vector_index.return_value = mock_vector_instance

                    with patch(
                        "utils.index_builder.create_hybrid_retriever"
                    ) as mock_retriever:
                        mock_hybrid_retriever = MagicMock()
                        mock_retriever.return_value = mock_hybrid_retriever

                        with patch(
                            "utils.index_builder.verify_rrf_configuration"
                        ) as mock_verify_rrf:
                            mock_verify_rrf.return_value = {
                                "issues": [],
                                "recommendations": [],
                            }

                            with patch(
                                "qdrant_client.AsyncQdrantClient"
                            ) as mock_async_client:
                                mock_client_instance = AsyncMock()
                                mock_async_client.return_value = mock_client_instance

                                result = await create_index_async(
                                    sample_docs, use_gpu=False
                                )

                                # Should still create vector index
                                assert "vector" in result
                                # KG should be None due to error
                                assert result.get("kg") is None

                                # Verify async operation completed
                                # Note: Cleanup testing would require more complex mocking


class TestAgentTools:
    """Test agent tool creation and management."""

    @pytest.fixture
    def mock_index_data_with_kg(self):
        """Mock index data with vector, KG, and retriever."""
        vector_index = MagicMock()
        kg_index = MagicMock()
        hybrid_retriever = MagicMock()

        return {
            "vector": vector_index,
            "kg": kg_index,
            "retriever": hybrid_retriever,
        }

    @pytest.fixture
    def mock_index_data_without_kg(self):
        """Mock index data without KG index."""
        vector_index = MagicMock()
        hybrid_retriever = MagicMock()

        return {
            "vector": vector_index,
            "kg": None,
            "retriever": hybrid_retriever,
        }

    def test_create_tools_with_kg_and_retriever(self, mock_index_data_with_kg):
        """Test tool creation with KG index and hybrid retriever available."""
        from agents.agent_utils import create_tools_from_index

        with patch(
            "llama_index.postprocessor.colbert_rerank.ColbertRerank"
        ) as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            # Mock query engines
            mock_kg_query_engine = MagicMock()
            mock_index_data_with_kg[
                "kg"
            ].as_query_engine.return_value = mock_kg_query_engine

            tools = create_tools_from_index(mock_index_data_with_kg)

            # Should have multiple tools
            assert len(tools) >= 2

            # Check tool names
            tool_names = [t.metadata.name for t in tools]
            assert "hybrid_fusion_search" in tool_names
            assert "knowledge_graph_query" in tool_names

            # Verify hybrid fusion tool was created with retriever
            hybrid_tool = next(
                t for t in tools if t.metadata.name == "hybrid_fusion_search"
            )
            assert "QueryFusionRetriever" in hybrid_tool.metadata.description
            assert "RRF" in hybrid_tool.metadata.description

            # Verify KG tool was created
            kg_tool = next(
                t for t in tools if t.metadata.name == "knowledge_graph_query"
            )
            assert "entity" in kg_tool.metadata.description.lower()
            assert "relationship" in kg_tool.metadata.description.lower()

    def test_create_tools_without_kg(self, mock_index_data_without_kg):
        """Test tool creation when KG is None."""
        from agents.agent_utils import create_tools_from_index

        with patch(
            "llama_index.postprocessor.colbert_rerank.ColbertRerank"
        ) as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            tools = create_tools_from_index(mock_index_data_without_kg)

            # Should still create hybrid tool
            assert len(tools) >= 1

            # No KG tools
            tool_names = [t.metadata.name for t in tools]
            assert "knowledge_graph_query" not in tool_names
            assert "hybrid_fusion_search" in tool_names

    def test_create_tools_without_retriever_fallback(self):
        """Test fallback to vector search when no retriever."""
        from agents.agent_utils import create_tools_from_index

        # Mock index data without retriever
        mock_vector_index = MagicMock()
        mock_vector_query_engine = MagicMock()
        mock_vector_index.as_query_engine.return_value = mock_vector_query_engine

        index_data = {
            "vector": mock_vector_index,
            "kg": None,
            "retriever": None,  # No retriever
        }

        with patch(
            "llama_index.postprocessor.colbert_rerank.ColbertRerank"
        ) as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            tools = create_tools_from_index(index_data)

            # Should create fallback tool
            assert len(tools) >= 1

            tool_names = [t.metadata.name for t in tools]
            assert "hybrid_vector_search" in tool_names

            # Verify fallback tool description
            fallback_tool = next(
                t for t in tools if t.metadata.name == "hybrid_vector_search"
            )
            assert "BGE-Large" in fallback_tool.metadata.description
            assert "SPLADE++" in fallback_tool.metadata.description

    def test_tool_metadata_descriptions(self, mock_index_data_with_kg):
        """Test tool metadata and descriptions are properly configured."""
        from agents.agent_utils import create_tools_from_index

        with patch(
            "llama_index.postprocessor.colbert_rerank.ColbertRerank"
        ) as mock_reranker:
            mock_reranker_instance = MagicMock()
            mock_reranker.return_value = mock_reranker_instance

            # Mock query engines
            mock_kg_query_engine = MagicMock()
            mock_index_data_with_kg[
                "kg"
            ].as_query_engine.return_value = mock_kg_query_engine

            tools = create_tools_from_index(mock_index_data_with_kg)

            for tool in tools:
                # Each tool should have metadata
                assert hasattr(tool, "metadata")
                assert hasattr(tool.metadata, "name")
                assert hasattr(tool.metadata, "description")

                # Names should be descriptive
                assert len(tool.metadata.name) > 5
                assert "_" in tool.metadata.name  # snake_case naming

                # Descriptions should be meaningful and detailed
                assert len(tool.metadata.description) > 50
                assert "Best for:" in tool.metadata.description

    def test_colbert_reranker_integration(self, mock_index_data_with_kg):
        """Test ColBERT reranker is properly integrated."""
        from agents.agent_utils import create_tools_from_index

        # Mock query engines first
        mock_kg_query_engine = MagicMock()
        mock_index_data_with_kg[
            "kg"
        ].as_query_engine.return_value = mock_kg_query_engine

        # Mock app settings to enable reranker
        with patch("agents.agent_utils.settings") as mock_settings:
            mock_settings.reranker_model = "colbert-ir/colbertv2.0"
            mock_settings.reranking_top_k = 10

            with patch(
                "llama_index.postprocessor.colbert_rerank.ColbertRerank"
            ) as mock_reranker:
                mock_reranker_instance = MagicMock()
                mock_reranker.return_value = mock_reranker_instance

                tools = create_tools_from_index(mock_index_data_with_kg)

                # Verify reranker was created (if the model is set)
                if mock_settings.reranker_model:
                    mock_reranker.assert_called_once_with(
                        model="colbert-ir/colbertv2.0",
                        top_n=10,
                        keep_retrieval_score=True,
                    )

                # Should have multiple tools
                assert len(tools) >= 1

    def test_create_agent_with_tools_success(self, mock_index_data_with_kg):
        """Test successful ReActAgent creation with tools."""
        from agents.agent_utils import create_agent_with_tools

        mock_llm = MagicMock()

        with patch("agents.agent_utils.create_tools_from_index") as mock_create_tools:
            mock_tools = [MagicMock(), MagicMock()]
            for i, tool in enumerate(mock_tools):
                tool.metadata.name = f"test_tool_{i}"
            mock_create_tools.return_value = mock_tools

            with patch(
                "llama_index.core.agent.ReActAgent.from_tools"
            ) as mock_react_agent:
                mock_agent = MagicMock()
                mock_react_agent.return_value = mock_agent

                with patch(
                    "llama_index.core.memory.ChatMemoryBuffer.from_defaults"
                ) as mock_memory:
                    mock_memory_instance = MagicMock()
                    mock_memory.return_value = mock_memory_instance

                    agent = create_agent_with_tools(mock_index_data_with_kg, mock_llm)

                    # Verify agent creation
                    assert agent == mock_agent
                    mock_react_agent.assert_called_once()

                    # Check agent creation parameters
                    call_kwargs = mock_react_agent.call_args[1]
                    assert "tools" in call_kwargs
                    assert "llm" in call_kwargs
                    assert "verbose" in call_kwargs
                    assert "max_iterations" in call_kwargs
                    assert "memory" in call_kwargs
                    assert call_kwargs["verbose"] is True
                    assert call_kwargs["max_iterations"] == 10

    def test_create_agent_with_tools_fallback(self, mock_index_data_with_kg):
        """Test ReActAgent creation with fallback configuration."""
        from agents.agent_utils import create_agent_with_tools

        mock_llm = MagicMock()

        with patch("agents.agent_utils.create_tools_from_index") as mock_create_tools:
            mock_tools = [MagicMock()]
            mock_tools[0].metadata.name = "test_tool"
            mock_create_tools.return_value = mock_tools

            with patch(
                "llama_index.core.agent.ReActAgent.from_tools"
            ) as mock_react_agent:
                # First call fails (with memory), second succeeds (fallback)
                mock_agent = MagicMock()
                mock_react_agent.side_effect = [Exception("Memory error"), mock_agent]

                agent = create_agent_with_tools(mock_index_data_with_kg, mock_llm)

                # Verify fallback agent creation
                assert agent == mock_agent
                assert mock_react_agent.call_count == 2

                # Check fallback call had minimal parameters
                fallback_call = mock_react_agent.call_args_list[1]
                assert fallback_call[0][0] == mock_tools  # tools as positional arg
                assert fallback_call[1]["llm"] == mock_llm  # llm as keyword arg
                assert fallback_call[1]["verbose"] is True


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple components."""

    @pytest.fixture
    def sample_entity_docs(self):
        """Documents with clear entities for integration testing."""
        return [
            Document(
                text="Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California."
            ),
            Document(
                text="Microsoft Corporation was established by Bill Gates and Paul Allen in Albuquerque, New Mexico."
            ),
            Document(
                text="Google LLC was created by Larry Page and Sergey Brin at Stanford University."
            ),
        ]

    def test_end_to_end_kg_to_agent_flow(self, sample_entity_docs):
        """Test complete flow from spaCy model to agent creation."""
        from agents.agent_utils import create_agent_with_tools
        from utils.index_builder import create_index

        # Mock entire pipeline
        with patch("utils.utils.ensure_spacy_model") as mock_spacy:
            mock_nlp = MagicMock()
            mock_spacy.return_value = mock_nlp

            with patch(
                "llama_index.core.KnowledgeGraphIndex.from_documents"
            ) as mock_kg:
                mock_kg_instance = MagicMock()
                mock_kg.return_value = mock_kg_instance

                with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
                    mock_llm = MagicMock()
                    mock_ollama.return_value = mock_llm

                    with patch(
                        "utils.index_builder.setup_hybrid_qdrant"
                    ) as mock_qdrant:
                        mock_vector_store = MagicMock()
                        mock_qdrant.return_value = mock_vector_store

                        with patch(
                            "llama_index.core.VectorStoreIndex.from_documents"
                        ) as mock_vector_index:
                            mock_vector_instance = MagicMock()
                            mock_vector_index.return_value = mock_vector_instance

                            with patch(
                                "utils.index_builder.create_hybrid_retriever"
                            ) as mock_retriever:
                                mock_hybrid_retriever = MagicMock()
                                mock_retriever.return_value = mock_hybrid_retriever

                                with patch(
                                    "llama_index.core.agent.ReActAgent.from_tools"
                                ) as mock_react_agent:
                                    mock_agent = MagicMock()
                                    mock_react_agent.return_value = mock_agent

                                    # Create index with KG
                                    index_data = create_index(
                                        sample_entity_docs, use_gpu=False
                                    )

                                    # Create agent with tools
                                    agent = create_agent_with_tools(
                                        index_data, mock_llm
                                    )

                                    # Verify complete flow
                                    assert index_data["kg"] is not None
                                    assert agent is not None

                                    # Verify spaCy was used for KG
                                    mock_spacy.assert_called_once_with("en_core_web_sm")

                                    # Verify KG index was created with spaCy extractor
                                    mock_kg.assert_called_once()
                                    kg_kwargs = mock_kg.call_args[1]
                                    assert kg_kwargs["extractor"] == mock_nlp

    def test_error_resilience_flow(self, sample_entity_docs):
        """Test system resilience when components fail."""
        from agents.agent_utils import create_agent_with_tools
        from utils.index_builder import create_index

        # Simulate spaCy failure but vector success
        with patch("utils.utils.ensure_spacy_model") as mock_spacy:
            mock_spacy.side_effect = RuntimeError("spaCy download failed")

            with patch("utils.index_builder.setup_hybrid_qdrant") as mock_qdrant:
                mock_vector_store = MagicMock()
                mock_qdrant.return_value = mock_vector_store

                with patch(
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_vector_index:
                    mock_vector_instance = MagicMock()
                    mock_vector_index.return_value = mock_vector_instance

                    with patch(
                        "utils.index_builder.create_hybrid_retriever"
                    ) as mock_retriever:
                        mock_hybrid_retriever = MagicMock()
                        mock_retriever.return_value = mock_hybrid_retriever

                        with patch(
                            "llama_index.core.agent.ReActAgent.from_tools"
                        ) as mock_react_agent:
                            mock_agent = MagicMock()
                            mock_react_agent.return_value = mock_agent

                            with patch("llama_index.llms.ollama.Ollama") as mock_ollama:
                                mock_llm = MagicMock()
                                mock_ollama.return_value = mock_llm

                                # Create index (should succeed without KG)
                                index_data = create_index(
                                    sample_entity_docs, use_gpu=False
                                )

                                # Create agent (should work with vector only)
                                agent = create_agent_with_tools(index_data, mock_llm)

                                # Verify graceful degradation
                                assert index_data["vector"] is not None
                                assert (
                                    index_data["kg"] is None
                                )  # Failed but didn't crash
                                assert agent is not None  # Still functional

    def test_logging_behavior_integration(self, sample_entity_docs, caplog):
        """Test proper logging throughout the integration."""
        from utils.index_builder import create_index

        with caplog.at_level(logging.INFO):
            with patch("utils.utils.ensure_spacy_model") as mock_spacy:
                mock_nlp = MagicMock()
                mock_spacy.return_value = mock_nlp

                with patch(
                    "llama_index.core.KnowledgeGraphIndex.from_documents"
                ) as mock_kg:
                    mock_kg_instance = MagicMock()
                    mock_kg.return_value = mock_kg_instance

                    with patch("llama_index.llms.ollama.Ollama"):
                        with patch("utils.index_builder.setup_hybrid_qdrant"):
                            with patch(
                                "llama_index.core.VectorStoreIndex.from_documents"
                            ):
                                with patch(
                                    "utils.index_builder.create_hybrid_retriever"
                                ):
                                    create_index(sample_entity_docs, use_gpu=False)

                                    # Check for expected log messages
                                    log_messages = [
                                        record.message for record in caplog.records
                                    ]
                                    kg_logs = [
                                        msg
                                        for msg in log_messages
                                        if "Knowledge Graph" in msg
                                    ]
                                    assert len(kg_logs) > 0


class TestPerformanceAndEdgeCases:
    """Test performance characteristics and edge cases."""

    def test_empty_document_list(self):
        """Test handling of empty document list."""
        from utils.index_builder import create_index

        with patch("utils.index_builder.setup_hybrid_qdrant") as mock_qdrant:
            mock_vector_store = MagicMock()
            mock_qdrant.return_value = mock_vector_store

            with patch(
                "llama_index.core.VectorStoreIndex.from_documents"
            ) as mock_vector_index:
                mock_vector_instance = MagicMock()
                mock_vector_index.return_value = mock_vector_instance

                result = create_index([], use_gpu=False)

                # Should still attempt to create indexes
                assert "vector" in result
                mock_vector_index.assert_called_once()
                call_args = mock_vector_index.call_args
                assert call_args[0][0] == []  # empty docs
                assert "embed_model" in call_args[1]

    def test_large_entity_extraction(self):
        """Test entity extraction with large text."""
        with patch("spacy.load") as mock_load:
            mock_nlp = MagicMock()
            mock_doc = MagicMock()

            # Create many mock entities
            mock_entities = []
            for i in range(100):
                entity = MagicMock()
                entity.text = f"Entity{i}"
                entity.label_ = "ORG" if i % 2 == 0 else "PERSON"
                mock_entities.append(entity)

            mock_doc.ents = mock_entities
            mock_nlp.return_value = mock_doc
            mock_load.return_value = mock_nlp

            from utils.utils import ensure_spacy_model

            nlp = ensure_spacy_model()
            doc = nlp("Large text with many entities...")

            assert len(doc.ents) == 100
            org_entities = [ent for ent in doc.ents if ent.label_ == "ORG"]
            person_entities = [ent for ent in doc.ents if ent.label_ == "PERSON"]
            assert len(org_entities) == 50
            assert len(person_entities) == 50

    @pytest.mark.asyncio
    async def test_async_timeout_behavior(self):
        """Test async operations don't hang indefinitely."""
        from utils.index_builder import create_index_async

        sample_docs = [Document(text="Test document")]

        with patch("qdrant_client.AsyncQdrantClient") as mock_async_client:
            mock_client_instance = AsyncMock()

            # Simulate slow response
            async def slow_close():
                await asyncio.sleep(0.1)

            mock_client_instance.close = slow_close
            mock_async_client.return_value = mock_client_instance

            with patch("utils.index_builder.setup_hybrid_qdrant_async") as mock_qdrant:
                mock_vector_store = MagicMock()
                mock_qdrant.return_value = mock_vector_store

                with patch(
                    "llama_index.core.VectorStoreIndex.from_documents"
                ) as mock_vector_index:
                    mock_vector_instance = MagicMock()
                    mock_vector_index.return_value = mock_vector_instance

                    # Should complete within reasonable time
                    start_time = asyncio.get_event_loop().time()
                    result = await create_index_async(sample_docs, use_gpu=False)
                    end_time = asyncio.get_event_loop().time()

                    assert result is not None
                    assert (
                        end_time - start_time
                    ) < 5.0  # Should complete within 5 seconds


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
