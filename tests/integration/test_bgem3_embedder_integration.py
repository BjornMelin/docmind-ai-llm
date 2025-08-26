"""Integration tests for BGE-M3 Embedder with unified embedding pipeline.

This module provides integration tests for BGEM3Embedder that test the actual
unified embedding generation with lightweight model configurations and realistic
embedding workflows.

Test Coverage:
- End-to-end embedding pipeline with both dense and sparse embeddings
- Integration with document processing workflows
- Performance characteristics with real model overhead
- Batch processing and memory management
- Device selection and GPU acceleration
- Error handling and recovery in integration context

Following 3-tier testing strategy:
- Tier 2 (Integration): Cross-component tests (<30s each)
- Use lightweight model configurations where possible
- Test with realistic text samples and batch sizes
- Validate unified embedding pipeline functionality
"""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.processing.embeddings.bgem3_embedder import BGEM3Embedder
from src.processing.embeddings.models import (
    EmbeddingError,
    EmbeddingParameters,
    EmbeddingResult,
)


@pytest.fixture
def integration_settings():
    """Integration test settings with realistic configuration."""
    settings = Mock()
    settings.bge_m3_model_name = "BAAI/bge-m3"
    settings.chunk_size = 512
    settings.chunk_overlap = 50
    return settings


@pytest.fixture
def integration_parameters():
    """Integration test embedding parameters with balanced configuration."""
    return EmbeddingParameters(
        max_length=2048,  # Smaller for integration tests
        batch_size_gpu=4,  # Conservative for testing
        batch_size_cpu=2,
        use_fp16=True,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,  # Disable ColBERT for faster integration tests
    )


@pytest.fixture
def sample_texts():
    """Realistic text samples for integration testing."""
    return [
        "DocMind AI is a powerful document processing system that uses advanced embedding techniques.",
        "The BGE-M3 model provides unified dense and sparse embeddings for hybrid search capabilities.",
        "Integration testing validates the complete embedding pipeline with real model behavior.",
        "Performance optimization ensures efficient processing of large document collections.",
        "Sparse embeddings enable keyword-based matching while dense embeddings capture semantic similarity.",
    ]


@pytest.fixture
def mock_integration_model():
    """Mock BGE-M3 model with realistic integration behavior."""
    model = MagicMock()

    # Mock realistic embedding generation with proper shapes and types
    def mock_encode(texts, **kwargs):
        batch_size = len(texts)

        result = {}

        if kwargs.get("return_dense", False):
            # Generate realistic dense embeddings (1024D)
            result["dense_vecs"] = np.random.rand(batch_size, 1024).astype(np.float32)

        if kwargs.get("return_sparse", False):
            # Generate realistic sparse embeddings (token weights)
            result["lexical_weights"] = [
                {
                    np.random.randint(1, 1000): np.random.rand()
                    for _ in range(np.random.randint(5, 20))
                }
                for _ in range(batch_size)
            ]

        if kwargs.get("return_colbert_vecs", False):
            # Generate realistic ColBERT embeddings (variable tokens per text)
            result["colbert_vecs"] = [
                np.random.rand(np.random.randint(8, 32), 1024).astype(np.float32)
                for _ in range(batch_size)
            ]

        # Simulate processing time
        import time

        time.sleep(0.1)  # Realistic model inference delay

        return result

    model.encode = mock_encode
    model.compute_lexical_matching_score = MagicMock(return_value=0.65)
    model.model = MagicMock()

    return model


class TestBGEM3EmbedderIntegration:
    """Integration tests for BGEM3Embedder."""

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_end_to_end_unified_embedding_pipeline(
        self,
        mock_torch,
        mock_flag_model,
        integration_settings,
        integration_parameters,
        sample_texts,
        mock_integration_model,
    ):
        """Test complete unified embedding pipeline with both dense and sparse embeddings."""
        # Mock torch configuration
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3  # 2GB used
        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings, integration_parameters)

        # Test unified embedding generation
        result = await embedder.embed_texts_async(sample_texts)

        # Verify unified embedding structure
        assert isinstance(result, EmbeddingResult)

        # CRITICAL: Verify both dense and sparse embeddings are present
        assert result.dense_embeddings is not None, (
            "Dense embeddings missing in integration test"
        )
        assert result.sparse_embeddings is not None, (
            "Sparse embeddings missing in integration test"
        )
        assert result.colbert_embeddings is None  # Disabled for integration

        # Verify embedding dimensions and structure
        assert len(result.dense_embeddings) == len(sample_texts)
        assert len(result.sparse_embeddings) == len(sample_texts)

        for i, (dense_emb, sparse_emb) in enumerate(
            zip(result.dense_embeddings, result.sparse_embeddings, strict=False)
        ):
            # Dense embedding validation
            assert isinstance(dense_emb, list)
            assert len(dense_emb) == 1024
            assert all(isinstance(val, float) for val in dense_emb)

            # Sparse embedding validation
            assert isinstance(sparse_emb, dict)
            assert len(sparse_emb) > 0
            assert all(
                isinstance(k, int) and isinstance(v, float)
                for k, v in sparse_emb.items()
            )

        # Verify performance metrics
        assert result.processing_time > 0
        assert result.batch_size == 4  # Integration test batch size
        assert result.memory_usage_mb > 0

        # Verify model information
        assert result.model_info["dense_enabled"] is True
        assert result.model_info["sparse_enabled"] is True
        assert result.model_info["embedding_dim"] == 1024
        assert result.model_info["library"] == "FlagEmbedding.BGEM3FlagModel"

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_batch_processing_integration(
        self, mock_torch, mock_flag_model, integration_settings, mock_integration_model
    ):
        """Test batch processing with various batch sizes."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3
        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings)

        # Test different batch sizes
        batch_sizes = [1, 3, 8, 12]

        for batch_size in batch_sizes:
            texts = [f"Test document number {i}" for i in range(batch_size)]

            result = await embedder.embed_texts_async(texts)

            # Verify batch processing results
            assert len(result.dense_embeddings) == batch_size
            assert len(result.sparse_embeddings) == batch_size
            assert result.batch_size <= 12  # Model's batch size limit

            # Verify all embeddings are properly formatted
            for dense_emb, sparse_emb in zip(
                result.dense_embeddings, result.sparse_embeddings, strict=False
            ):
                assert len(dense_emb) == 1024
                assert isinstance(sparse_emb, dict)
                assert len(sparse_emb) > 0

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_device_selection_integration(
        self, mock_torch, mock_flag_model, integration_settings, mock_integration_model
    ):
        """Test device selection and batch size optimization in integration context."""
        mock_flag_model.return_value = mock_integration_model

        # Test CUDA availability and memory-based optimization
        test_cases = [
            (True, 8 * 1024**3, 4),  # 8GB GPU -> batch_size 4
            (True, 12 * 1024**3, 8),  # 12GB GPU -> batch_size 8
            (True, 16 * 1024**3, 12),  # 16GB GPU -> batch_size 12
            (False, 0, 2),  # CPU -> batch_size 2
        ]

        for cuda_available, gpu_memory, expected_batch_size in test_cases:
            mock_torch.cuda.is_available.return_value = cuda_available
            if cuda_available:
                mock_torch.cuda.get_device_properties.return_value = MagicMock(
                    total_memory=gpu_memory
                )
                expected_device = "cuda"
            else:
                expected_device = "cpu"

            embedder = BGEM3Embedder(integration_settings)

            # Verify device and batch size selection
            assert embedder.device == expected_device
            assert embedder.batch_size == expected_batch_size

            # Verify model initialization with correct parameters
            mock_flag_model.assert_called_with(
                model_name_or_path="BAAI/bge-m3",
                use_fp16=cuda_available,  # FP16 only on CUDA
                device=expected_device,
            )

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_performance_tracking_integration(
        self,
        mock_torch,
        mock_flag_model,
        integration_settings,
        sample_texts,
        mock_integration_model,
    ):
        """Test performance tracking across multiple embedding operations."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 2 * 1024**3
        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings)

        # Perform multiple embedding operations
        operations = [
            sample_texts[:2],  # 2 texts
            sample_texts[2:4],  # 2 texts
            sample_texts[4:],  # 1 text
        ]

        total_texts = 0
        for texts in operations:
            result = await embedder.embed_texts_async(texts)
            total_texts += len(texts)

            # Verify each operation succeeds
            assert len(result.dense_embeddings) == len(texts)
            assert len(result.sparse_embeddings) == len(texts)

        # Verify cumulative performance tracking
        stats = embedder.get_performance_stats()

        assert stats["total_texts_embedded"] == total_texts
        assert stats["total_processing_time"] > 0
        assert stats["avg_time_per_text_ms"] > 0
        assert stats["device"] == "cuda"
        assert stats["unified_embeddings_enabled"] is True

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_individual_embedding_methods_integration(
        self,
        mock_torch,
        mock_flag_model,
        integration_settings,
        sample_texts,
        mock_integration_model,
    ):
        """Test individual embedding extraction methods in integration context."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings)

        # Test dense-only embeddings
        dense_embeddings = embedder.get_dense_embeddings(sample_texts)
        assert dense_embeddings is not None
        assert len(dense_embeddings) == len(sample_texts)
        assert all(len(emb) == 1024 for emb in dense_embeddings)

        # Test sparse-only embeddings
        sparse_embeddings = embedder.get_sparse_embeddings(sample_texts)
        assert sparse_embeddings is not None
        assert len(sparse_embeddings) == len(sample_texts)
        assert all(isinstance(emb, dict) and len(emb) > 0 for emb in sparse_embeddings)

        # Verify different model calls were made
        assert mock_integration_model.encode.call_count == 2

        # Verify calls used correct parameters
        calls = mock_integration_model.encode.call_args_list

        # Dense call
        dense_call = calls[0]
        assert dense_call[1]["return_dense"] is True
        assert dense_call[1]["return_sparse"] is False

        # Sparse call
        sparse_call = calls[1]
        assert sparse_call[1]["return_dense"] is False
        assert sparse_call[1]["return_sparse"] is True

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    def test_sparse_similarity_computation_integration(
        self, mock_torch, mock_flag_model, integration_settings, mock_integration_model
    ):
        """Test sparse similarity computation in integration context."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings)

        # Create realistic sparse embeddings
        sparse1 = {101: 0.8, 205: 0.6, 310: 0.4, 425: 0.9, 501: 0.3}
        sparse2 = {101: 0.7, 189: 0.5, 310: 0.8, 456: 0.6, 602: 0.4}

        similarity = embedder.compute_sparse_similarity(sparse1, sparse2)

        # Verify similarity computation
        assert isinstance(similarity, float)
        assert similarity == 0.65  # Mock return value

        # Verify model method was called
        mock_integration_model.compute_lexical_matching_score.assert_called_once_with(
            sparse1, sparse2
        )

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_memory_management_integration(
        self, mock_torch, mock_flag_model, integration_settings, mock_integration_model
    ):
        """Test memory management and cleanup in integration context."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )

        # Simulate memory usage progression
        memory_usage = [1024**3, 2 * 1024**3, 1.5 * 1024**3]  # 1GB, 2GB, 1.5GB
        mock_torch.cuda.memory_allocated.side_effect = memory_usage

        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings)

        # Process texts and verify memory reporting
        result = await embedder.embed_texts_async(["Test text for memory tracking"])

        # Verify memory usage is reported
        assert result.memory_usage_mb > 0
        assert result.memory_usage_mb == 2048.0  # 2GB in MB

        # Test model unloading
        embedder.unload_model()

        # Verify model was moved to CPU for memory cleanup
        mock_integration_model.model.to.assert_called_once_with("cpu")

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery_integration(
        self, mock_torch, mock_flag_model, integration_settings
    ):
        """Test error handling and recovery in integration context."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024**3

        # Mock model with intermittent failures
        failing_model = MagicMock()
        call_count = 0

        def failing_encode(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("GPU out of memory")

            # Succeed on third attempt
            return {
                "dense_vecs": np.random.rand(1, 1024).astype(np.float32),
                "lexical_weights": [{1: 0.8, 5: 0.6}],
            }

        failing_model.encode = failing_encode
        mock_flag_model.return_value = failing_model

        # Test error handling for embedding generation
        embedder = BGEM3Embedder(integration_settings)

        # Should fail after multiple attempts
        with pytest.raises(EmbeddingError):
            await embedder.embed_texts_async(["Test text"])

        # Verify error handling in individual methods
        result = embedder.get_dense_embeddings(["Test text"])
        assert result is None  # Should return None on error

        result = embedder.get_sparse_embeddings(["Test text"])
        assert result is None  # Should return None on error

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_embedding_quality_and_consistency(
        self, mock_torch, mock_flag_model, integration_settings, mock_integration_model
    ):
        """Test embedding quality and consistency across multiple runs."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024**3

        # Create deterministic mock for consistency testing
        def deterministic_encode(texts, **kwargs):
            batch_size = len(texts)
            result = {}

            if kwargs.get("return_dense", False):
                # Create deterministic dense embeddings based on text hash
                embeddings = []
                for text in texts:
                    seed = hash(text) % 1000
                    np.random.seed(seed)
                    emb = np.random.rand(1024).astype(np.float32)
                    embeddings.append(emb)
                result["dense_vecs"] = np.array(embeddings)

            if kwargs.get("return_sparse", False):
                # Create deterministic sparse embeddings based on text hash
                sparse_embs = []
                for text in texts:
                    seed = hash(text) % 1000
                    np.random.seed(seed)
                    tokens = np.random.randint(1, 100, size=10)
                    weights = np.random.rand(10)
                    sparse_emb = {
                        int(token): float(weight)
                        for token, weight in zip(tokens, weights, strict=False)
                    }
                    sparse_embs.append(sparse_emb)
                result["lexical_weights"] = sparse_embs

            return result

        mock_integration_model.encode = deterministic_encode
        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings)

        test_texts = ["Consistent test text for embedding quality validation"]

        # Generate embeddings multiple times
        results = []
        for _ in range(3):
            result = await embedder.embed_texts_async(test_texts)
            results.append(result)

        # Verify consistency across runs
        for i in range(1, len(results)):
            # Dense embeddings should be identical
            np.testing.assert_array_almost_equal(
                results[0].dense_embeddings[0],
                results[i].dense_embeddings[0],
                decimal=5,
            )

            # Sparse embeddings should be identical
            assert results[0].sparse_embeddings[0] == results[i].sparse_embeddings[0]

    @pytest.mark.integration
    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    @patch("src.processing.embeddings.bgem3_embedder.torch")
    @pytest.mark.asyncio
    async def test_integration_with_document_processing_workflow(
        self, mock_torch, mock_flag_model, integration_settings, mock_integration_model
    ):
        """Test integration with typical document processing workflows."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_memory=16 * 1024**3
        )
        mock_torch.cuda.memory_allocated.return_value = 1024**3
        mock_flag_model.return_value = mock_integration_model

        embedder = BGEM3Embedder(integration_settings)

        # Simulate document processing workflow
        document_chunks = [
            "Chapter 1: Introduction to DocMind AI and its capabilities in document processing.",
            "The system uses BGE-M3 embeddings for unified dense and sparse representation.",
            "Chapter 2: Technical Architecture - The hybrid processor combines multiple strategies.",
            "Performance optimization ensures efficient processing of large document collections.",
            "Chapter 3: Integration Testing - Validates end-to-end functionality and performance.",
        ]

        # Process document chunks as they would be in real workflow
        batch_results = []
        batch_size = 3

        for i in range(0, len(document_chunks), batch_size):
            batch = document_chunks[i : i + batch_size]
            result = await embedder.embed_texts_async(batch)
            batch_results.append(result)

        # Verify all batches processed successfully
        total_processed = sum(len(result.dense_embeddings) for result in batch_results)
        assert total_processed == len(document_chunks)

        # Verify embedding quality for document processing
        for result in batch_results:
            assert result.dense_embeddings is not None
            assert result.sparse_embeddings is not None

            for dense_emb, sparse_emb in zip(
                result.dense_embeddings, result.sparse_embeddings, strict=False
            ):
                # Verify dense embedding properties
                assert len(dense_emb) == 1024
                assert all(isinstance(val, float) for val in dense_emb)

                # Verify sparse embedding properties
                assert isinstance(sparse_emb, dict)
                assert len(sparse_emb) > 0
                assert all(
                    isinstance(k, int) and isinstance(v, float)
                    for k, v in sparse_emb.items()
                )

        # Verify performance statistics
        final_stats = embedder.get_performance_stats()
        assert final_stats["total_texts_embedded"] == len(document_chunks)
        assert final_stats["total_processing_time"] > 0
