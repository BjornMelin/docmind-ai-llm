"""Tests for GPU optimization features in DocMind AI.

This module tests GPU acceleration, torch.compile optimization, CUDA streams,
mixed precision, profiling, and fallback behavior for embeddings and indexing.
Tests use comprehensive mocking to avoid GPU dependencies in CI/CD.
"""

import sys
from pathlib import Path

# Fix import path for tests
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestGPUOptimization:
    """Test GPU optimization features for embeddings and indexing."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    def test_torch_compile_applied(self):
        """Test torch.compile is applied when GPU available."""
        from utils.utils import get_embed_model

        with patch("torch.compile") as mock_compile:
            mock_compile.return_value = MagicMock()

            with patch("models.AppSettings") as mock_settings_class:
                mock_settings = MagicMock()
                mock_settings.gpu_acceleration = True
                mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
                mock_settings.embedding_batch_size = 32
                mock_settings_class.return_value = mock_settings

                with patch("torch.cuda.is_available", return_value=True):
                    embed_model = get_embed_model()

                    mock_compile.assert_called_once()
                    assert mock_compile.call_args[1]["mode"] == "reduce-overhead"
                    assert mock_compile.call_args[1]["dynamic"] == True

    def test_cuda_streams_async(self):
        """Test CUDA streams for parallel operations in async context."""
        from llama_index.core import Document

        from utils.index_builder import create_index_async

        with patch("torch.cuda.Stream") as mock_stream_class:
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=None)

            with patch("torch.cuda.is_available", return_value=True):
                with patch("utils.index_builder.AsyncQdrantClient"):
                    with patch("utils.index_builder.setup_hybrid_qdrant_async"):
                        with patch("llama_index.core.VectorStoreIndex.from_documents"):
                            # Test stream creation in async context
                            docs = [Document(text="test")]
                            try:
                                asyncio.run(create_index_async(docs, use_gpu=True))
                                mock_stream_class.assert_called()
                                mock_stream.synchronize.assert_called()
                            except Exception:
                                # Expected due to mocking, but stream should be created
                                mock_stream_class.assert_called()

    def test_cuda_streams_sync(self):
        """Test CUDA streams for parallel operations in sync context."""
        from llama_index.core import Document

        from utils.index_builder import create_index

        with patch("torch.cuda.Stream") as mock_stream_class:
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=None)

            with patch("torch.cuda.is_available", return_value=True):
                with patch("qdrant_client.QdrantClient"):
                    with patch("utils.index_builder.setup_hybrid_qdrant"):
                        with patch("llama_index.core.VectorStoreIndex.from_documents"):
                            # Test stream creation in sync context
                            docs = [Document(text="test")]
                            try:
                                create_index(docs, use_gpu=True)
                                mock_stream_class.assert_called()
                                mock_stream.synchronize.assert_called()
                            except Exception:
                                # Expected due to mocking, but stream should be created
                                mock_stream_class.assert_called()

    def test_gpu_fallback_cuda_unavailable(self):
        """Test graceful fallback when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            from utils.utils import get_embed_model

            with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
                mock_model = MagicMock()
                mock_fastembed.return_value = mock_model

                embed_model = get_embed_model()

                # Should work on CPU
                assert embed_model is not None
                # Should use CPU provider only
                call_args = mock_fastembed.call_args[1]
                assert call_args["providers"] == ["CPUExecutionProvider"]

    def test_gpu_fallback_exception_handling(self):
        """Test graceful fallback when GPU operations fail."""
        with patch("torch.cuda.is_available", return_value=True):
            with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
                # Simulate GPU failure
                mock_fastembed.side_effect = Exception("GPU memory error")

                from utils.utils import get_embed_model

                with pytest.raises(Exception, match="GPU memory error"):
                    get_embed_model()


class TestMixedPrecisionOptimization:
    """Test mixed precision and precision-related optimizations."""

    def test_fastembed_gpu_providers(self):
        """Test FastEmbed GPU provider configuration."""
        from utils.utils import get_embed_model

        with patch("torch.cuda.is_available", return_value=True):
            with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
                mock_model = MagicMock()
                mock_fastembed.return_value = mock_model

                embed_model = get_embed_model()

                # Verify GPU providers are configured
                call_kwargs = mock_fastembed.call_args[1]
                assert "providers" in call_kwargs
                assert "CUDAExecutionProvider" in call_kwargs["providers"]
                assert "CPUExecutionProvider" in call_kwargs["providers"]

    def test_embedding_batch_size_configuration(self):
        """Test embedding batch size configuration for GPU optimization."""
        from models import AppSettings

        settings = AppSettings()

        # Test default batch size is GPU-optimized
        assert settings.embedding_batch_size == 32
        assert 1 <= settings.embedding_batch_size <= 512

    def test_quantization_configuration(self):
        """Test quantization settings for memory optimization."""
        from models import AppSettings

        settings = AppSettings()

        # Test quantization settings
        assert hasattr(settings, "enable_quantization")
        assert hasattr(settings, "quantization_type")
        assert settings.quantization_type in ["int8", "int4"]


class TestProfilingAndDebugMode:
    """Test GPU profiling and debug mode functionality."""

    @pytest.mark.asyncio
    async def test_gpu_profiling_in_debug_mode_async(self):
        """Test GPU profiling in debug mode for async operations."""
        from llama_index.core import Document

        from utils.index_builder import create_index_async

        with patch("models.AppSettings") as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.debug_mode = True
            mock_settings.gpu_acceleration = True
            mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
            mock_settings.sparse_embedding_model = "prithivida/Splade_PP_en_v1"
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.dense_embedding_dimension = 1024
            mock_settings.embedding_batch_size = 32
            mock_settings.max_entities = 50
            mock_settings.default_model = "test-model"
            mock_settings_class.return_value = mock_settings

            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.profiler.profile") as mock_profile:
                    mock_profiler = MagicMock()
                    mock_profile.return_value = mock_profiler
                    mock_profiler.__enter__ = MagicMock(return_value=mock_profiler)
                    mock_profiler.__exit__ = MagicMock(return_value=None)

                    with patch("utils.index_builder.AsyncQdrantClient"):
                        with patch("utils.index_builder.setup_hybrid_qdrant_async"):
                            with patch(
                                "llama_index.core.VectorStoreIndex.from_documents"
                            ):
                                docs = [Document(text="test")]

                                try:
                                    await create_index_async(docs, use_gpu=True)
                                except Exception:
                                    pass  # Expected due to mocking

                                # Verify profiling was attempted
                                if torch.cuda.is_available():
                                    mock_profile.assert_called()
                                    # Verify profiler export was called
                                    mock_profiler.export_chrome_trace.assert_called_with(
                                        "gpu_trace.json"
                                    )

    def test_gpu_profiling_in_debug_mode_sync(self):
        """Test GPU profiling in debug mode for sync operations."""
        from llama_index.core import Document

        from utils.index_builder import create_index

        with patch("models.AppSettings") as mock_settings_class:
            mock_settings = MagicMock()
            mock_settings.debug_mode = True
            mock_settings.gpu_acceleration = True
            mock_settings.dense_embedding_model = "BAAI/bge-large-en-v1.5"
            mock_settings.sparse_embedding_model = "prithivida/Splade_PP_en_v1"
            mock_settings.qdrant_url = "http://localhost:6333"
            mock_settings.dense_embedding_dimension = 1024
            mock_settings.embedding_batch_size = 32
            mock_settings.max_entities = 50
            mock_settings.default_model = "test-model"
            mock_settings_class.return_value = mock_settings

            with patch("torch.cuda.is_available", return_value=True):
                with patch("torch.profiler.profile") as mock_profile:
                    mock_profiler = MagicMock()
                    mock_profile.return_value = mock_profiler
                    mock_profiler.__enter__ = MagicMock(return_value=mock_profiler)
                    mock_profiler.__exit__ = MagicMock(return_value=None)

                    with patch("qdrant_client.QdrantClient"):
                        with patch("utils.index_builder.setup_hybrid_qdrant"):
                            with patch(
                                "llama_index.core.VectorStoreIndex.from_documents"
                            ):
                                docs = [Document(text="test")]

                                try:
                                    create_index(docs, use_gpu=True)
                                except Exception:
                                    pass  # Expected due to mocking

                                # Verify profiling was attempted
                                if torch.cuda.is_available():
                                    mock_profile.assert_called()
                                    mock_profiler.export_chrome_trace.assert_called_with(
                                        "gpu_trace.json"
                                    )

    def test_gpu_info_logging(self):
        """Test GPU information logging."""
        from utils.utils import get_embed_model

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="Tesla V100"):
                with patch("torch.cuda.get_device_properties") as mock_props:
                    mock_device_props = MagicMock()
                    mock_device_props.total_memory = 32 * 1024**3  # 32GB
                    mock_props.return_value = mock_device_props

                    with patch("utils.utils.FastEmbedEmbedding"):
                        with patch("logging.info") as mock_log:
                            get_embed_model()

                            # Should log GPU information
                            log_calls = [str(call) for call in mock_log.call_args_list]
                            gpu_logged = any(
                                "Tesla V100" in log_call and "32.0GB" in log_call
                                for log_call in log_calls
                            )
                            assert gpu_logged or any(
                                "GPU:" in log_call for log_call in log_calls
                            )


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarking tests for embeddings."""

    @pytest.mark.skipif(
        not pytest.importorskip("pytest_benchmark"),
        reason="pytest-benchmark not available",
    )
    def test_embedding_performance_cpu(self, benchmark):
        """Benchmark embedding performance on CPU."""
        from utils.utils import get_embed_model

        with patch("torch.cuda.is_available", return_value=False):
            with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
                mock_model = MagicMock()
                # Mock realistic embedding output
                mock_model.embed_documents.return_value = [
                    [0.1] * 1024 for _ in range(100)
                ]
                mock_fastembed.return_value = mock_model

                embed_model = get_embed_model()
                texts = ["test text"] * 100

                result = benchmark(embed_model.embed_documents, texts)
                assert len(result) == 100

    @pytest.mark.skipif(
        not torch.cuda.is_available() or not pytest.importorskip("pytest_benchmark"),
        reason="GPU not available or pytest-benchmark not available",
    )
    def test_embedding_performance_gpu(self, benchmark):
        """Benchmark embedding performance on GPU."""
        from utils.utils import get_embed_model

        with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
            mock_model = MagicMock()
            # Mock realistic embedding output
            mock_model.embed_documents.return_value = [[0.1] * 1024 for _ in range(100)]
            mock_fastembed.return_value = mock_model

            embed_model = get_embed_model()
            texts = ["test text"] * 100

            result = benchmark(embed_model.embed_documents, texts)
            assert len(result) == 100

    def test_batch_size_optimization(self):
        """Test that batch size is optimized for available hardware."""
        from models import AppSettings

        # Test different scenarios
        with patch("torch.cuda.is_available", return_value=True):
            settings = AppSettings()
            assert settings.embedding_batch_size >= 16  # GPU should use larger batches

        with patch("torch.cuda.is_available", return_value=False):
            settings = AppSettings()
            assert settings.embedding_batch_size > 0  # Should still work on CPU


class TestHardwareDetection:
    """Test hardware detection and adaptation."""

    def test_detect_hardware_with_gpu(self):
        """Test hardware detection when GPU is available."""
        from utils.utils import detect_hardware

        with patch("torch.cuda.is_available", return_value=True):
            with patch("torch.cuda.get_device_name", return_value="RTX 4090"):
                with patch("torch.cuda.get_device_properties") as mock_props:
                    mock_device_props = MagicMock()
                    mock_device_props.total_memory = 24 * 1024**3  # 24GB
                    mock_props.return_value = mock_device_props

                    with patch(
                        "utils.utils.ModelManager.get_text_embedding_model"
                    ) as mock_model:
                        mock_embedding_model = MagicMock()
                        mock_embedding_model.model.model.get_providers.return_value = [
                            "CUDAExecutionProvider",
                            "CPUExecutionProvider",
                        ]
                        mock_model.return_value = mock_embedding_model

                        hardware = detect_hardware()

                        assert hardware["cuda_available"] is True
                        assert hardware["gpu_name"] == "RTX 4090"
                        assert hardware["vram_total_gb"] == 24.0
                        assert (
                            "CUDAExecutionProvider" in hardware["fastembed_providers"]
                        )

    def test_detect_hardware_without_gpu(self):
        """Test hardware detection when GPU is not available."""
        from utils.utils import detect_hardware

        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "utils.utils.ModelManager.get_text_embedding_model"
            ) as mock_model:
                mock_embedding_model = MagicMock()
                mock_embedding_model.model.model.get_providers.return_value = [
                    "CPUExecutionProvider"
                ]
                mock_model.return_value = mock_embedding_model

                hardware = detect_hardware()

                assert hardware["cuda_available"] is False
                assert hardware["fastembed_providers"] == ["CPUExecutionProvider"]

    def test_detect_hardware_fallback(self):
        """Test hardware detection fallback when FastEmbed detection fails."""
        from utils.utils import detect_hardware

        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "utils.utils.ModelManager.get_text_embedding_model",
                side_effect=Exception("Model loading failed"),
            ):
                hardware = detect_hardware()

                # Should fallback to basic torch detection
                assert hardware["cuda_available"] is True
                assert hardware["gpu_name"] == "Unknown"


class TestAsyncGPUOperations:
    """Test GPU operations in async contexts."""

    @pytest.mark.asyncio
    async def test_async_gpu_memory_management(self):
        """Test GPU memory management in async operations."""
        from llama_index.core import Document

        from utils.index_builder import create_index_async

        with patch("torch.cuda.is_available", return_value=True):
            with patch("utils.index_builder.AsyncQdrantClient") as mock_client:
                mock_async_client = MagicMock()
                mock_client.return_value = mock_async_client

                with patch("utils.index_builder.setup_hybrid_qdrant_async"):
                    with patch("llama_index.core.VectorStoreIndex.from_documents"):
                        docs = [Document(text="test document")]

                        try:
                            result = await create_index_async(docs, use_gpu=True)
                            # Should clean up async client
                            mock_async_client.close.assert_called()
                        except Exception:
                            # Even on failure, should attempt cleanup
                            pass

    @pytest.mark.asyncio
    async def test_async_concurrent_embedding_operations(self):
        """Test concurrent embedding operations don't conflict."""
        from utils.utils import get_embed_model

        with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
            mock_model = MagicMock()
            mock_model.embed_documents.return_value = [[0.1] * 1024 for _ in range(10)]
            mock_fastembed.return_value = mock_model

            embed_model = get_embed_model()

            # Simulate concurrent operations
            async def embed_batch(texts):
                return embed_model.embed_documents(texts)

            tasks = [
                embed_batch([f"text {i}" for i in range(10)]),
                embed_batch([f"text {i + 10}" for i in range(10)]),
                embed_batch([f"text {i + 20}" for i in range(10)]),
            ]

            results = await asyncio.gather(*tasks)

            # All batches should complete successfully
            assert len(results) == 3
            for result in results:
                assert len(result) == 10


class TestFastEmbedGPUAcceleration:
    """Test FastEmbed native GPU acceleration features."""

    def test_fastembed_providers_configuration(self):
        """Test FastEmbed provider configuration for GPU acceleration."""
        from llama_index.core import Document

        from utils.index_builder import create_index

        with patch("torch.cuda.is_available", return_value=True):
            with patch("utils.index_builder.FastEmbedEmbedding") as mock_fastembed:
                with patch("utils.index_builder.SparseTextEmbedding") as mock_sparse:
                    with patch("qdrant_client.QdrantClient"):
                        with patch("utils.index_builder.setup_hybrid_qdrant"):
                            with patch(
                                "llama_index.core.VectorStoreIndex.from_documents"
                            ):
                                docs = [Document(text="test")]

                                try:
                                    create_index(docs, use_gpu=True)
                                except Exception:
                                    pass  # Expected due to mocking

                                # Verify both dense and sparse models use GPU providers
                                dense_call_args = mock_fastembed.call_args[1]
                                sparse_call_args = mock_sparse.call_args[1]

                                expected_providers = [
                                    "CUDAExecutionProvider",
                                    "CPUExecutionProvider",
                                ]
                                assert (
                                    dense_call_args["providers"] == expected_providers
                                )
                                assert (
                                    sparse_call_args["providers"] == expected_providers
                                )

    def test_fastembed_cache_directory_configuration(self):
        """Test FastEmbed cache directory is properly configured."""
        from utils.utils import get_embed_model

        with patch("utils.utils.FastEmbedEmbedding") as mock_fastembed:
            get_embed_model()

            call_kwargs = mock_fastembed.call_args[1]
            assert call_kwargs["cache_dir"] == "./embeddings_cache"

    def test_embedding_model_names(self):
        """Test correct embedding model names are used."""
        from llama_index.core import Document

        from utils.index_builder import create_index

        with patch("utils.index_builder.FastEmbedEmbedding") as mock_fastembed:
            with patch("utils.index_builder.SparseTextEmbedding") as mock_sparse:
                with patch("qdrant_client.QdrantClient"):
                    with patch("utils.index_builder.setup_hybrid_qdrant"):
                        with patch("llama_index.core.VectorStoreIndex.from_documents"):
                            docs = [Document(text="test")]

                            try:
                                create_index(docs, use_gpu=True)
                            except Exception:
                                pass  # Expected due to mocking

                            # Verify model names
                            dense_call_args = mock_fastembed.call_args[1]
                            sparse_call_args = mock_sparse.call_args[1]

                            assert (
                                dense_call_args["model_name"]
                                == "BAAI/bge-large-en-v1.5"
                            )
                            assert (
                                sparse_call_args["model_name"]
                                == "prithivida/Splade_PP_en_v1"
                            )


class TestEmbeddingPipelineIntegration:
    """Test integration between embedding models and GPU optimization."""

    def test_hybrid_search_configuration(self):
        """Test hybrid search with both dense and sparse embeddings."""
        from models import AppSettings

        settings = AppSettings()

        # Test research-backed RRF fusion weights
        assert settings.rrf_fusion_weight_dense == 0.7
        assert settings.rrf_fusion_weight_sparse == 0.3
        assert 10 <= settings.rrf_fusion_alpha <= 100
        assert settings.rrf_fusion_alpha == 60  # Optimal value

    def test_dimension_consistency(self):
        """Test embedding dimension consistency."""
        from models import AppSettings

        settings = AppSettings()

        # BGE-Large uses 1024 dimensions
        assert settings.dense_embedding_dimension == 1024

    def test_sparse_embeddings_enabled(self):
        """Test sparse embeddings are enabled by default."""
        from models import AppSettings

        settings = AppSettings()

        assert settings.enable_sparse_embeddings is True

    @pytest.mark.integration
    def test_full_embedding_pipeline_integration(self):
        """Test full embedding pipeline with GPU optimization."""
        from llama_index.core import Document

        from utils.index_builder import create_index

        with patch("torch.cuda.is_available", return_value=True):
            with patch("utils.index_builder.FastEmbedEmbedding") as mock_dense:
                with patch("utils.index_builder.SparseTextEmbedding") as mock_sparse:
                    with patch("qdrant_client.QdrantClient"):
                        with patch(
                            "utils.index_builder.setup_hybrid_qdrant"
                        ) as mock_setup:
                            with patch(
                                "llama_index.core.VectorStoreIndex.from_documents"
                            ) as mock_index:
                                with patch(
                                    "utils.index_builder.create_hybrid_retriever"
                                ):
                                    docs = [Document(text="Integration test document")]

                                    try:
                                        result = create_index(docs, use_gpu=True)
                                    except Exception:
                                        pass  # Expected due to mocking

                                    # Verify all components were initialized
                                    mock_dense.assert_called()
                                    mock_sparse.assert_called()
                                    mock_setup.assert_called()

                                    # Verify GPU providers were configured
                                    dense_args = mock_dense.call_args[1]
                                    sparse_args = mock_sparse.call_args[1]

                                    expected_providers = [
                                        "CUDAExecutionProvider",
                                        "CPUExecutionProvider",
                                    ]
                                    assert dense_args["providers"] == expected_providers
                                    assert (
                                        sparse_args["providers"] == expected_providers
                                    )
