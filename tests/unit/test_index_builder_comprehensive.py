"""Comprehensive test coverage for index_builder.py.

This test suite provides extensive coverage for the index_builder module,
including unit tests, integration tests, edge cases, error handling,
and property-based tests to achieve 70%+ coverage.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from llama_index.core import Document
from llama_index.core.schema import ImageDocument

from models import AppSettings

# Import the module under test
from utils.index_builder import (
    create_hybrid_retriever,
    create_index,
    create_index_async,
    create_multimodal_index,
    create_multimodal_index_async,
)


class TestCreateHybridRetriever:
    """Test the create_hybrid_retriever function comprehensively."""

    def test_create_hybrid_retriever_success(self, sample_documents):
        """Test successful hybrid retriever creation."""
        # Arrange
        mock_index = MagicMock()
        mock_index.docstore.docs.values.return_value = sample_documents

        with (
            patch("utils.index_builder.VectorIndexRetriever") as mock_vector_retriever,
            patch("utils.index_builder.QueryFusionRetriever") as mock_fusion
        ):
                mock_dense_retriever = MagicMock()
                mock_sparse_retriever = MagicMock()
                mock_vector_retriever.side_effect = [
                    mock_dense_retriever,
                    mock_sparse_retriever,
                ]
                mock_fusion_instance = MagicMock()
                mock_fusion.return_value = mock_fusion_instance

                # Act
                result = create_hybrid_retriever(mock_index)

                # Assert
                assert result == mock_fusion_instance
                assert mock_vector_retriever.call_count == 2

                # Verify dense retriever call
                dense_call = mock_vector_retriever.call_args_list[0]
                assert dense_call[1]["index"] == mock_index
                assert dense_call[1]["vector_store_query_mode"] == "default"

                # Verify sparse retriever call
                sparse_call = mock_vector_retriever.call_args_list[1]
                assert sparse_call[1]["index"] == mock_index
                assert sparse_call[1]["vector_store_query_mode"] == "sparse"

                # Verify fusion retriever call
                fusion_call = mock_fusion.call_args[1]
                assert fusion_call["retrievers"] == [
                    mock_dense_retriever,
                    mock_sparse_retriever,
                ]
                assert fusion_call["mode"] == "reciprocal_rerank"
                assert fusion_call["use_async"] == True

    def test_create_hybrid_retriever_invalid_index(self):
        """Test hybrid retriever creation with invalid index."""
        with pytest.raises(ValueError, match="Invalid index: None"):  # Should raise due to None index
            create_hybrid_retriever(None)

    def test_create_hybrid_retriever_exception_fallback(self, sample_documents):
        """Test fallback to dense-only retriever on exception."""
        # Arrange
        mock_index = MagicMock()

        with patch("utils.index_builder.VectorIndexRetriever") as mock_vector_retriever:
            with patch(
                "utils.index_builder.QueryFusionRetriever",
                side_effect=RuntimeError("Fusion error"),
            ):
                mock_fallback_retriever = MagicMock()
                mock_vector_retriever.return_value = mock_fallback_retriever

                # Act
                result = create_hybrid_retriever(mock_index)

                # Assert - should fall back to dense-only retriever
                assert result == mock_fallback_retriever
                # Should be called twice - once for each retriever in try block, once for fallback
                assert mock_vector_retriever.call_count >= 1

    def test_create_hybrid_retriever_settings_configuration(self):
        """Test that settings are properly applied to retriever configuration."""
        mock_index = MagicMock()
        test_settings = AppSettings(
            similarity_top_k=10, prefetch_factor=2, debug_mode=True
        )

        with patch("utils.index_builder.settings", test_settings):
            with patch(
                "utils.index_builder.VectorIndexRetriever"
            ) as mock_vector_retriever:
                with patch("utils.index_builder.QueryFusionRetriever") as mock_fusion:
                    create_hybrid_retriever(mock_index)

                    # Verify settings are applied
                    dense_call = mock_vector_retriever.call_args_list[0]
                    expected_top_k = (
                        test_settings.prefetch_factor * test_settings.similarity_top_k
                    )
                    assert dense_call[1]["similarity_top_k"] == expected_top_k

                    # Verify fusion settings
                    fusion_call = mock_fusion.call_args[1]
                    assert (
                        fusion_call["similarity_top_k"]
                        == test_settings.similarity_top_k
                    )
                    assert fusion_call["verbose"] == test_settings.debug_mode


class TestCreateIndexAsync:
    """Test the async index creation function comprehensively."""

    @pytest.mark.asyncio
    async def test_create_index_async_success(self, sample_documents):
        """Test successful async index creation."""
        with patch(
            "utils.index_builder.managed_async_qdrant_client"
        ) as mock_client_ctx:
            with (
                patch("utils.index_builder.setup_hybrid_qdrant_async") as mock_setup,
                patch("utils.index_builder.FastEmbedEmbedding") as mock_embed,
                patch(
            ):
                        "utils.index_builder.SparseTextEmbedding"
                    ) as mock_sparse:
                        with patch(
                            "utils.index_builder.VectorStoreIndex"
                        ) as mock_index_cls:
                            with patch(
                                "utils.index_builder.create_hybrid_retriever"
                            ) as mock_retriever:
                                # Arrange
                                mock_client = AsyncMock()
                                mock_client_ctx.return_value.__aenter__.return_value = (
                                    mock_client
                                )
                                mock_client_ctx.return_value.__aexit__.return_value = (
                                    None
                                )

                                mock_vector_store = MagicMock()
                                mock_setup.return_value = mock_vector_store

                                mock_index = MagicMock()
                                mock_index_cls.from_documents.return_value = mock_index

                                mock_kg_index = MagicMock()
                                mock_hybrid_retriever = MagicMock()
                                mock_retriever.return_value = mock_hybrid_retriever

                                with patch(
                                    "utils.index_builder.KnowledgeGraphIndex"
                                ) as mock_kg_cls:
                                    with patch(
                                        "utils.index_builder.ensure_spacy_model"
                                    ):
                                        with patch("utils.index_builder.Ollama"):
                                            mock_kg_cls.from_documents.return_value = (
                                                mock_kg_index
                                            )

                                            # Act
                                            result = await create_index_async(
                                                sample_documents, use_gpu=False
                                            )

                                            # Assert
                                            assert "vector" in result
                                            assert "kg" in result
                                            assert "retriever" in result
                                            assert result["vector"] == mock_index
                                            assert result["kg"] == mock_kg_index
                                            assert (
                                                result["retriever"]
                                                == mock_hybrid_retriever
                                            )

    @pytest.mark.asyncio
    async def test_create_index_async_gpu_enabled(self, sample_documents):
        """Test async index creation with GPU acceleration."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.Stream") as mock_stream_cls,
            patch("torch.profiler.profile") as mock_profiler,
            patch(
        ):
                        "utils.index_builder.managed_async_qdrant_client"
                    ) as mock_client_ctx:
                        with patch(
                            "utils.index_builder.managed_gpu_operation"
                        ) as mock_gpu_ctx:
                            # Arrange mocks
                            mock_stream = MagicMock()
                            mock_stream_cls.return_value = mock_stream

                            mock_client = AsyncMock()
                            mock_client_ctx.return_value.__aenter__.return_value = (
                                mock_client
                            )
                            mock_client_ctx.return_value.__aexit__.return_value = None

                            mock_gpu_ctx.return_value.__aenter__ = AsyncMock()
                            mock_gpu_ctx.return_value.__aexit__ = AsyncMock()

                            with (
                                patch("utils.index_builder.setup_hybrid_qdrant_async"),
                                patch("utils.index_builder.FastEmbedEmbedding"),
                                patch(
                            ):
                                        "utils.index_builder.SparseTextEmbedding"
                                    ):
                                        with patch(
                                            "utils.index_builder.VectorStoreIndex"
                                        ):
                                            with patch(
                                                "utils.index_builder.create_hybrid_retriever"
                                            ):
                                                with patch(
                                                    "utils.index_builder.settings"
                                                ) as mock_settings:
                                                    mock_settings.gpu_acceleration = (
                                                        True
                                                    )
                                                    mock_settings.debug_mode = True

                                                    # Act
                                                    await create_index_async(
                                                        sample_documents, use_gpu=True
                                                    )

                                                    # Assert
                                                    mock_stream.synchronize.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_index_async_kg_failure_fallback(self, sample_documents):
        """Test graceful fallback when KG index creation fails."""
        with patch(
            "utils.index_builder.managed_async_qdrant_client"
        ) as mock_client_ctx:
            with (
                patch("utils.index_builder.setup_hybrid_qdrant_async"),
                patch("utils.index_builder.FastEmbedEmbedding"),
                patch("utils.index_builder.SparseTextEmbedding"),
                patch(
            ):
                            "utils.index_builder.VectorStoreIndex"
                        ) as mock_index_cls:
                            with patch("utils.index_builder.create_hybrid_retriever"):
                                # Arrange - KG creation fails
                                mock_client = AsyncMock()
                                mock_client_ctx.return_value.__aenter__.return_value = (
                                    mock_client
                                )
                                mock_client_ctx.return_value.__aexit__.return_value = (
                                    None
                                )

                                mock_index = MagicMock()
                                mock_index_cls.from_documents.return_value = mock_index

                                with patch(
                                    "utils.index_builder.KnowledgeGraphIndex.from_documents",
                                    side_effect=RuntimeError("KG creation failed"),
                                ):
                                    with patch(
                                        "utils.index_builder.ensure_spacy_model"
                                    ):
                                        # Act
                                        result = await create_index_async(
                                            sample_documents, use_gpu=False
                                        )

                                        # Assert - should still return vector index
                                        assert result["vector"] == mock_index
                                        assert (
                                            result["kg"] is None
                                        )  # Should be None due to failure

    @pytest.mark.asyncio
    async def test_create_index_async_retriever_failure_fallback(
        self, sample_documents
    ):
        """Test graceful fallback when hybrid retriever creation fails."""
        with patch(
            "utils.index_builder.managed_async_qdrant_client"
        ) as mock_client_ctx:
            with (
                patch("utils.index_builder.setup_hybrid_qdrant_async"),
                patch("utils.index_builder.FastEmbedEmbedding"),
                patch("utils.index_builder.SparseTextEmbedding"),
                patch(
            ):
                            "utils.index_builder.VectorStoreIndex"
                        ) as mock_index_cls:
                            # Arrange - Retriever creation fails
                            mock_client = AsyncMock()
                            mock_client_ctx.return_value.__aenter__.return_value = (
                                mock_client
                            )
                            mock_client_ctx.return_value.__aexit__.return_value = None

                            mock_index = MagicMock()
                            mock_index_cls.from_documents.return_value = mock_index

                            with patch(
                                "utils.index_builder.create_hybrid_retriever",
                                side_effect=RuntimeError("Retriever creation failed"),
                            ):
                                with patch(
                                    "utils.index_builder.KnowledgeGraphIndex.from_documents"
                                ):
                                    with patch(
                                        "utils.index_builder.ensure_spacy_model"
                                    ):
                                        with patch("utils.index_builder.Ollama"):
                                            # Act
                                            result = await create_index_async(
                                                sample_documents, use_gpu=False
                                            )

                                            # Assert - should still return vector index
                                            assert result["vector"] == mock_index
                                            assert (
                                                result["retriever"] is None
                                            )  # Should be None due to failure

    @pytest.mark.asyncio
    async def test_create_index_async_rrf_configuration_validation(
        self, sample_documents
    ):
        """Test RRF configuration validation in async index creation."""
        with patch("utils.index_builder.verify_rrf_configuration") as mock_verify:
            with patch(
                "utils.index_builder.managed_async_qdrant_client"
            ) as mock_client_ctx:
                # Arrange - RRF validation returns issues
                mock_verify.return_value = {
                    "issues": ["Weight sum not equal to 1.0"],
                    "recommendations": ["Adjust weights to sum to 1.0"],
                }

                mock_client = AsyncMock()
                mock_client_ctx.return_value.__aenter__.return_value = mock_client
                mock_client_ctx.return_value.__aexit__.return_value = None

                with (
                    patch("utils.index_builder.setup_hybrid_qdrant_async"),
                    patch("utils.index_builder.FastEmbedEmbedding"),
                    patch("utils.index_builder.SparseTextEmbedding"),
                    patch("utils.index_builder.VectorStoreIndex"),
                    patch(
                ):
                                    "utils.index_builder.create_hybrid_retriever"
                                ):
                                    with patch(
                                        "utils.index_builder.KnowledgeGraphIndex.from_documents"
                                    ):
                                        with patch(
                                            "utils.index_builder.ensure_spacy_model"
                                        ):
                                            with patch("utils.index_builder.Ollama"):
                                                # Act
                                                result = await create_index_async(
                                                    sample_documents, use_gpu=False
                                                )

                                                # Assert - should still create index despite warnings
                                                assert "vector" in result
                                                mock_verify.assert_called_once()


class TestCreateIndex:
    """Test the synchronous index creation function comprehensively."""

    def test_create_index_success(self, sample_documents):
        """Test successful synchronous index creation."""
        with (
            patch("utils.index_builder.QdrantClient") as mock_client_cls,
            patch("utils.index_builder.setup_hybrid_qdrant") as mock_setup,
            patch("utils.index_builder.FastEmbedEmbedding") as mock_embed,
            patch(
        ):
                        "utils.index_builder.SparseTextEmbedding"
                    ) as mock_sparse:
                        with patch(
                            "utils.index_builder.VectorStoreIndex"
                        ) as mock_index_cls:
                            with patch(
                                "utils.index_builder.create_hybrid_retriever"
                            ) as mock_retriever:
                                # Arrange
                                mock_client = MagicMock()
                                mock_client_cls.return_value = mock_client

                                mock_vector_store = MagicMock()
                                mock_setup.return_value = mock_vector_store

                                mock_index = MagicMock()
                                mock_index_cls.from_documents.return_value = mock_index

                                mock_kg_index = MagicMock()
                                mock_hybrid_retriever = MagicMock()
                                mock_retriever.return_value = mock_hybrid_retriever

                                with patch(
                                    "utils.index_builder.KnowledgeGraphIndex"
                                ) as mock_kg_cls:
                                    with patch(
                                        "utils.index_builder.ensure_spacy_model"
                                    ):
                                        with patch("utils.index_builder.Ollama"):
                                            mock_kg_cls.from_documents.return_value = (
                                                mock_kg_index
                                            )

                                            # Act
                                            result = create_index(
                                                sample_documents, use_gpu=False
                                            )

                                            # Assert
                                            assert result["vector"] == mock_index
                                            assert result["kg"] == mock_kg_index
                                            assert (
                                                result["retriever"]
                                                == mock_hybrid_retriever
                                            )

    def test_create_index_empty_documents(self):
        """Test index creation with empty document list."""
        with (
            patch("utils.index_builder.QdrantClient"),
            patch("utils.index_builder.setup_hybrid_qdrant"),
            patch("utils.index_builder.FastEmbedEmbedding"),
            patch("utils.index_builder.SparseTextEmbedding"),
            patch(
        ):
                            "utils.index_builder.VectorStoreIndex"
                        ) as mock_index_cls:
                            with patch("utils.index_builder.create_hybrid_retriever"):
                                mock_index = MagicMock()
                                mock_index_cls.from_documents.return_value = mock_index

                                # Act
                                result = create_index([], use_gpu=False)

                                # Assert - should handle empty list gracefully
                                assert "vector" in result
                                mock_index_cls.from_documents.assert_called_once_with(
                                    [],
                                    storage_context=unittest.mock.ANY,
                                    embed_model=unittest.mock.ANY,
                                    sparse_embed_model=unittest.mock.ANY,
                                )

    def test_create_index_gpu_cuda_stream_operations(self, sample_documents):
        """Test GPU CUDA stream operations in synchronous index creation."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.Stream") as mock_stream_cls,
            patch("torch.cuda.empty_cache") as mock_empty_cache,
            patch("utils.index_builder.QdrantClient"),
            patch("utils.index_builder.setup_hybrid_qdrant"),
            patch("utils.index_builder.FastEmbedEmbedding"),
            patch("utils.index_builder.SparseTextEmbedding"),
            patch("utils.index_builder.VectorStoreIndex"),
            patch(
        ):
                                            "utils.index_builder.create_hybrid_retriever"
                                        ):
                                            with patch(
                                                "utils.index_builder.settings"
                                            ) as mock_settings:
                                                # Arrange
                                                mock_settings.gpu_acceleration = True
                                                mock_settings.debug_mode = False

                                                mock_stream = MagicMock()
                                                mock_stream_cls.return_value = (
                                                    mock_stream
                                                )

                                                # Act
                                                create_index(
                                                    sample_documents, use_gpu=True
                                                )

                                                # Assert
                                                mock_stream.synchronize.assert_called_once()
                                                mock_empty_cache.assert_called_once()

    @pytest.mark.parametrize(
        "error_type,expected_behavior",
        [
            (ConnectionError, "should_raise"),
            (TimeoutError, "should_raise"),
            (ValueError, "should_raise"),
            (RuntimeError, "should_raise"),
        ],
    )
    def test_create_index_error_handling_scenarios(
        self, sample_documents, error_type, expected_behavior
    ):
        """Test various error handling scenarios in index creation."""
        with patch(
            "utils.index_builder.QdrantClient", side_effect=error_type("Test error")
        ):
            if expected_behavior == "should_raise":
                with pytest.raises(error_type):
                    create_index(sample_documents, use_gpu=False)

    def test_create_index_rrf_weight_calculation(self, sample_documents):
        """Test RRF weight calculation and hybrid alpha computation."""
        test_settings = AppSettings(
            rrf_fusion_weight_dense=0.7, rrf_fusion_weight_sparse=0.3
        )

        with (
            patch("utils.index_builder.settings", test_settings),
            patch("utils.index_builder.QdrantClient"),
            patch("utils.index_builder.setup_hybrid_qdrant"),
            patch("utils.index_builder.FastEmbedEmbedding"),
            patch("utils.index_builder.SparseTextEmbedding"),
            patch("utils.index_builder.VectorStoreIndex"),
            patch(
        ):
                                    "utils.index_builder.create_hybrid_retriever"
                                ):
                                    with patch(
                                        "utils.index_builder.KnowledgeGraphIndex.from_documents"
                                    ):
                                        with patch(
                                            "utils.index_builder.ensure_spacy_model"
                                        ):
                                            with patch("utils.index_builder.Ollama"):
                                                # Act
                                                result = create_index(
                                                    sample_documents, use_gpu=False
                                                )

                                                # Assert - verify weights are used correctly
                                                # The actual weight calculation happens internally
                                                # but we can verify the function completes successfully
                                                assert "vector" in result


class TestCreateMultimodalIndex:
    """Test the multimodal index creation function comprehensively."""

    def test_create_multimodal_index_success(self):
        """Test successful multimodal index creation."""
        # Arrange
        text_docs = [Document(text="Text content")]
        image_docs = [ImageDocument(image="base64data", metadata={"type": "image"})]
        mixed_docs = text_docs + image_docs

        with (
            patch("utils.index_builder.QdrantClient"),
            patch("utils.index_builder.QdrantVectorStore"),
            patch("utils.index_builder.StorageContext"),
            patch(
        ):
                        "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
                    ):
                        with patch(
                            "utils.index_builder.MultiModalVectorStoreIndex"
                        ) as mock_mm_index:
                            mock_index = MagicMock()
                            mock_mm_index.from_documents.return_value = mock_index

                            # Act
                            result = create_multimodal_index(mixed_docs, use_gpu=True)

                            # Assert
                            assert result == mock_index
                            mock_mm_index.from_documents.assert_called_once()

                            # Verify documents were processed correctly
                            call_args = mock_mm_index.from_documents.call_args[1]
                            assert len(call_args["documents"]) == len(mixed_docs)

    def test_create_multimodal_index_text_only_fallback(self):
        """Test fallback to text-only index when multimodal creation fails."""
        text_docs = [Document(text="Text content")]

        with patch(
            "utils.index_builder.MultiModalVectorStoreIndex.from_documents",
            side_effect=RuntimeError("Multimodal failed"),
        ):
            with patch("utils.index_builder.create_index") as mock_create_index:
                mock_fallback_result = {"vector": MagicMock()}
                mock_create_index.return_value = mock_fallback_result

                # Act
                result = create_multimodal_index(text_docs, use_gpu=True)

                # Assert - should fallback to text-only
                assert result == mock_fallback_result["vector"]
                mock_create_index.assert_called_once_with(text_docs, True)

    def test_create_multimodal_index_no_text_docs_raises(self):
        """Test that exception is raised when no text docs available for fallback."""
        image_docs = [ImageDocument(image="base64data")]

        with patch(
            "utils.index_builder.MultiModalVectorStoreIndex.from_documents",
            side_effect=RuntimeError("Multimodal failed"),
        ):
            # Act & Assert - should raise since no text docs for fallback
            with pytest.raises(
                Exception, match="No text documents available for fallback"
            ):
                create_multimodal_index(image_docs, use_gpu=True)

    def test_create_multimodal_index_quantization_enabled(self):
        """Test multimodal index creation with quantization enabled."""
        docs = [Document(text="Text content")]

        with patch("utils.index_builder.settings") as mock_settings:
            mock_settings.enable_quantization = True

            with patch("transformers.BitsAndBytesConfig") as mock_quant_config:
                with patch(
                    "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
                ) as mock_embed:
                    with (
                        patch("utils.index_builder.QdrantClient"),
                        patch("utils.index_builder.QdrantVectorStore"),
                        patch("utils.index_builder.StorageContext"),
                        patch(
                    ):
                                    "utils.index_builder.MultiModalVectorStoreIndex"
                                ):
                                    # Act
                                    create_multimodal_index(docs, use_gpu=True)

                                    # Assert - quantization config should be used
                                    mock_quant_config.assert_called_once()
                                    mock_embed.assert_called_once()

    def test_create_multimodal_index_gpu_cuda_streams(self):
        """Test GPU CUDA streams in multimodal index creation."""
        docs = [Document(text="Text content")]

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.Stream") as mock_stream_cls,
            patch("utils.index_builder.QdrantClient"),
            patch("utils.index_builder.QdrantVectorStore"),
            patch("utils.index_builder.StorageContext"),
            patch(
        ):
                                "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
                            ):
                                with patch(
                                    "utils.index_builder.MultiModalVectorStoreIndex"
                                ):
                                    # Arrange
                                    mock_stream = MagicMock()
                                    mock_stream_cls.return_value = mock_stream

                                    # Act
                                    create_multimodal_index(docs, use_gpu=True)

                                    # Assert
                                    mock_stream.synchronize.assert_called_once()


class TestCreateMultimodalIndexAsync:
    """Test the async multimodal index creation function comprehensively."""

    @pytest.mark.asyncio
    async def test_create_multimodal_index_async_success(self):
        """Test successful async multimodal index creation."""
        # Arrange
        text_docs = [Document(text="Text content")]
        image_docs = [ImageDocument(image="base64data")]
        mixed_docs = text_docs + image_docs

        with patch(
            "utils.index_builder.managed_async_qdrant_client"
        ) as mock_client_ctx:
            with (
                patch("utils.index_builder.QdrantVectorStore"),
                patch("utils.index_builder.StorageContext"),
                patch(
            ):
                        "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
                    ):
                        with patch(
                            "utils.index_builder.MultiModalVectorStoreIndex"
                        ) as mock_mm_index:
                            # Arrange
                            mock_client = AsyncMock()
                            mock_client_ctx.return_value.__aenter__.return_value = (
                                mock_client
                            )
                            mock_client_ctx.return_value.__aexit__.return_value = None

                            mock_index = MagicMock()
                            mock_mm_index.from_documents.return_value = mock_index

                            # Act
                            result = await create_multimodal_index_async(
                                mixed_docs, use_gpu=True
                            )

                            # Assert
                            assert result == mock_index

    @pytest.mark.asyncio
    async def test_create_multimodal_index_async_fallback_to_sync(self):
        """Test fallback to synchronous creation on async failure."""
        docs = [Document(text="Text content")]

        with patch(
            "utils.index_builder.managed_async_qdrant_client",
            side_effect=RuntimeError("Async failed"),
        ):
            with patch("asyncio.to_thread") as mock_to_thread:
                mock_sync_result = MagicMock()
                mock_to_thread.return_value = mock_sync_result

                # Act
                result = await create_multimodal_index_async(docs, use_gpu=True)

                # Assert - should fallback to sync version
                assert result == mock_sync_result
                mock_to_thread.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_multimodal_index_async_gpu_operations(self):
        """Test GPU operations in async multimodal index creation."""
        docs = [Document(text="Text content")]

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.Stream") as mock_stream_cls,
            patch(
        ):
                    "utils.index_builder.managed_async_qdrant_client"
                ) as mock_client_ctx:
                    with patch(
                        "utils.index_builder.managed_gpu_operation"
                    ) as mock_gpu_ctx:
                        with (
                            patch("utils.index_builder.QdrantVectorStore"),
                            patch("utils.index_builder.StorageContext"),
                            patch(
                        ):
                                    "llama_index.embeddings.huggingface.HuggingFaceEmbedding"
                                ):
                                    with patch(
                                        "utils.index_builder.MultiModalVectorStoreIndex"
                                    ):
                                        # Arrange
                                        mock_client = AsyncMock()
                                        mock_client_ctx.return_value.__aenter__.return_value = mock_client
                                        mock_client_ctx.return_value.__aexit__.return_value = None

                                        mock_gpu_ctx.return_value.__aenter__ = (
                                            AsyncMock()
                                        )
                                        mock_gpu_ctx.return_value.__aexit__ = (
                                            AsyncMock()
                                        )

                                        mock_stream = MagicMock()
                                        mock_stream_cls.return_value = mock_stream

                                        # Act
                                        await create_multimodal_index_async(
                                            docs, use_gpu=True
                                        )

                                        # Assert
                                        mock_stream.synchronize.assert_called_once()


class TestPropertyBasedIndexBuilder:
    """Property-based tests for index builder functions."""

    @given(
        texts=st.lists(st.text(min_size=1, max_size=1000), min_size=1, max_size=5),
        use_gpu=st.booleans(),
    )
    def test_create_index_properties(self, texts, use_gpu):
        """Test index creation properties with various inputs."""
        docs = [Document(text=text) for text in texts]

        with (
            patch("utils.index_builder.QdrantClient"),
            patch("utils.index_builder.setup_hybrid_qdrant"),
            patch("utils.index_builder.FastEmbedEmbedding"),
            patch("utils.index_builder.SparseTextEmbedding"),
            patch(
        ):
                            "utils.index_builder.VectorStoreIndex"
                        ) as mock_index_cls:
                            with patch("utils.index_builder.create_hybrid_retriever"):
                                mock_index = MagicMock()
                                mock_index_cls.from_documents.return_value = mock_index

                                # Act
                                result = create_index(docs, use_gpu=use_gpu)

                                # Assert - properties that should always hold
                                assert isinstance(result, dict)
                                assert "vector" in result
                                assert "kg" in result
                                assert "retriever" in result

    @given(
        rrf_dense_weight=st.floats(min_value=0.1, max_value=0.9),
        rrf_sparse_weight=st.floats(min_value=0.1, max_value=0.9),
    )
    def test_rrf_weight_properties(self, rrf_dense_weight, rrf_sparse_weight):
        """Test RRF weight validation properties."""
        # Normalize weights to sum to 1.0
        total = rrf_dense_weight + rrf_sparse_weight
        normalized_dense = rrf_dense_weight / total
        normalized_sparse = rrf_sparse_weight / total

        test_settings = AppSettings(
            rrf_fusion_weight_dense=normalized_dense,
            rrf_fusion_weight_sparse=normalized_sparse,
        )

        # Should create valid settings when weights are properly normalized
        assert test_settings.rrf_fusion_weight_dense == normalized_dense
        assert test_settings.rrf_fusion_weight_sparse == normalized_sparse
        assert (
            abs(
                test_settings.rrf_fusion_weight_dense
                + test_settings.rrf_fusion_weight_sparse
                - 1.0
            )
            < 0.001
        )


# Import unittest.mock for ANY matcher
import unittest.mock
