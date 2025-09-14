"""Unit tests for models in embeddings.py.

Covers embedding parameters, results, and error handling.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.models.embeddings import EmbeddingParameters, EmbeddingResult


@pytest.mark.unit
class TestEmbeddingParametersAdditional:
    """Additional tests for EmbeddingParameters model."""

    def test_pooling_and_device_validation(self):
        """Test pooling method and device validation in EmbeddingParameters."""
        p = EmbeddingParameters(pooling_method="mean", device="cpu")
        assert p.pooling_method == "mean"
        assert p.device == "cpu"

    def test_serialization_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        p = EmbeddingParameters(max_length=4096, return_colbert=True)
        d = p.model_dump()
        restored = EmbeddingParameters.model_validate(d)
        assert restored.max_length == 4096
        assert restored.return_colbert is True


@pytest.mark.unit
class TestEmbeddingResultAdditional:
    """Additional tests for EmbeddingResult model."""

    def test_various_embeddings(self):
        """Test EmbeddingResult with various embedding types and formats."""
        dense = [[0.1] * 4, [0.2] * 4]
        sparse = [{100: 0.8}, {200: 0.7}]
        colbert = [np.random.randn(2, 4), np.random.randn(1, 4)]
        r = EmbeddingResult(
            dense_embeddings=dense,
            sparse_embeddings=sparse,
            colbert_embeddings=colbert,
            processing_time=1.0,
            batch_size=2,
            memory_usage_mb=256.0,
        )
        assert len(r.dense_embeddings) == 2
        assert len(r.sparse_embeddings) == 2
        assert len(r.colbert_embeddings) == 2


class TestEmbeddingParameters:
    """Test suite for EmbeddingParameters model."""

    @pytest.mark.unit
    def test_embedding_parameters_creation_default(self):
        """Test EmbeddingParameters creation with default values."""
        params = EmbeddingParameters()

        assert params.max_length == 8192
        assert params.use_fp16 is True
        assert params.normalize_embeddings is True
        assert params.return_dense is True
        assert params.return_sparse is True
        assert params.return_colbert is False
        assert params.device == "cuda"
        assert params.pooling_method == "cls"
        assert params.weights_for_different_modes == [0.4, 0.2, 0.4]
        assert params.return_numpy is False

    @pytest.mark.unit
    def test_embedding_parameters_creation_custom(self):
        """Test EmbeddingParameters creation with custom values."""
        params = EmbeddingParameters(
            max_length=4096,
            use_fp16=False,
            normalize_embeddings=False,
            return_dense=False,
            return_sparse=False,
            return_colbert=True,
            device="cpu",
            pooling_method="mean",
            weights_for_different_modes=[0.5, 0.3, 0.2],
            return_numpy=True,
        )

        assert params.max_length == 4096
        assert params.use_fp16 is False
        assert params.normalize_embeddings is False
        assert params.return_dense is False
        assert params.return_sparse is False
        assert params.return_colbert is True
        assert params.device == "cpu"
        assert params.pooling_method == "mean"
        assert params.weights_for_different_modes == [0.5, 0.3, 0.2]
        assert params.return_numpy is True

    @pytest.mark.unit
    @pytest.mark.parametrize("max_length", [512, 8192, 16384])
    def test_embedding_parameters_valid_max_length_boundary(self, max_length):
        """Test EmbeddingParameters max_length boundary values."""
        params = EmbeddingParameters(max_length=max_length)
        assert params.max_length == max_length

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_length", [511, 256, 16385, 32768])
    def test_embedding_parameters_invalid_max_length(self, invalid_length):
        """Test EmbeddingParameters with invalid max_length values."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingParameters(max_length=invalid_length)

        errors = exc_info.value.errors()
        assert any(
            error["type"] in ("greater_than_equal", "less_than_equal")
            for error in errors
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cuda", "cpu", "auto"])
    def test_embedding_parameters_valid_devices(self, device):
        """Test EmbeddingParameters with various device configurations."""
        params = EmbeddingParameters(device=device)
        assert params.device == device

    @pytest.mark.unit
    @pytest.mark.parametrize("device", ["cuda:0", "cuda:1", "mps", "tpu"])
    def test_embedding_parameters_specialized_devices(self, device):
        """Test EmbeddingParameters with specialized device configurations."""
        params = EmbeddingParameters(device=device)
        assert params.device == device

    @pytest.mark.unit
    @pytest.mark.parametrize("pooling", ["cls", "mean"])
    def test_embedding_parameters_valid_pooling_methods(self, pooling):
        """Test EmbeddingParameters with valid pooling methods."""
        params = EmbeddingParameters(pooling_method=pooling)
        assert params.pooling_method == pooling

    @pytest.mark.unit
    def test_embedding_parameters_weights_custom_valid(self):
        """Test EmbeddingParameters with custom valid weight configurations."""
        test_weights = [
            [1.0, 0.0, 0.0],  # Dense only
            [0.0, 1.0, 0.0],  # Sparse only
            [0.0, 0.0, 1.0],  # ColBERT only
            [0.33, 0.33, 0.34],  # Balanced
            [0.6, 0.3, 0.1],  # Dense heavy
        ]

        for weights in test_weights:
            params = EmbeddingParameters(weights_for_different_modes=weights)
            assert params.weights_for_different_modes == weights

    @pytest.mark.unit
    def test_embedding_parameters_weights_empty_list(self):
        """Test EmbeddingParameters with empty weights list."""
        params = EmbeddingParameters(weights_for_different_modes=[])
        assert params.weights_for_different_modes == []

    @pytest.mark.unit
    def test_embedding_parameters_weights_single_value(self):
        """Test EmbeddingParameters with single weight value."""
        params = EmbeddingParameters(weights_for_different_modes=[1.0])
        assert params.weights_for_different_modes == [1.0]

    @pytest.mark.unit
    def test_embedding_parameters_weights_large_list(self):
        """Test EmbeddingParameters with large weight list."""
        large_weights = [0.1] * 10
        params = EmbeddingParameters(weights_for_different_modes=large_weights)
        assert params.weights_for_different_modes == large_weights

    @pytest.mark.unit
    def test_embedding_parameters_boolean_combinations(self):
        """Test EmbeddingParameters with various boolean combinations."""
        # All embeddings disabled
        params1 = EmbeddingParameters(
            return_dense=False, return_sparse=False, return_colbert=False
        )
        assert not any(
            [params1.return_dense, params1.return_sparse, params1.return_colbert]
        )

        # All embeddings enabled
        params2 = EmbeddingParameters(
            return_dense=True, return_sparse=True, return_colbert=True
        )
        assert all(
            [params2.return_dense, params2.return_sparse, params2.return_colbert]
        )

        # Mixed configurations
        params3 = EmbeddingParameters(
            return_dense=True, return_sparse=False, return_colbert=True
        )
        assert params3.return_dense
        assert not params3.return_sparse
        assert params3.return_colbert

    @pytest.mark.unit
    def test_embedding_parameters_serialization(self):
        """Test EmbeddingParameters serialization and deserialization."""
        original = EmbeddingParameters(
            max_length=4096,
            use_fp16=False,
            normalize_embeddings=True,
            return_dense=True,
            return_sparse=False,
            return_colbert=True,
            device="cpu",
            pooling_method="mean",
            weights_for_different_modes=[0.6, 0.2, 0.2],
            return_numpy=True,
        )

        # Serialize and deserialize
        json_data = original.model_dump()
        restored = EmbeddingParameters.model_validate(json_data)

        assert restored.max_length == original.max_length
        assert restored.use_fp16 == original.use_fp16
        assert restored.device == original.device
        assert (
            restored.weights_for_different_modes == original.weights_for_different_modes
        )

    @pytest.mark.unit
    def test_embedding_parameters_validation_types(self):
        """Test EmbeddingParameters validation with invalid types."""
        with pytest.raises(ValidationError):
            EmbeddingParameters(max_length="invalid")  # Should be int

        with pytest.raises(ValidationError):
            EmbeddingParameters(use_fp16=[1, 2, 3])  # Should be bool, not list

        with pytest.raises(ValidationError):
            EmbeddingParameters(weights_for_different_modes="invalid")  # Should be list

        with pytest.raises(ValidationError):
            EmbeddingParameters(device=123)  # Should be string


class TestEmbeddingResult:
    """Test suite for EmbeddingResult model."""

    @pytest.mark.unit
    def test_embedding_result_creation_minimal(self):
        """Test EmbeddingResult creation with minimal required fields."""
        result = EmbeddingResult(
            processing_time=1.25, batch_size=8, memory_usage_mb=1024.0
        )

        assert result.dense_embeddings is None
        assert result.sparse_embeddings is None
        assert result.colbert_embeddings is None
        assert result.processing_time == 1.25
        assert result.batch_size == 8
        assert result.memory_usage_mb == 1024.0
        assert result.model_info == {}

    @pytest.mark.unit
    def test_embedding_result_creation_full(self):
        """Test EmbeddingResult creation with all fields."""
        dense_embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        sparse_embs = [{1: 0.5, 5: 0.3}, {2: 0.7, 10: 0.2}]
        colbert_embs = [
            np.array([[0.1, 0.2], [0.3, 0.4]]),
            np.array([[0.5, 0.6], [0.7, 0.8]]),
        ]

        result = EmbeddingResult(
            dense_embeddings=dense_embs,
            sparse_embeddings=sparse_embs,
            colbert_embeddings=colbert_embs,
            processing_time=2.5,
            batch_size=16,
            memory_usage_mb=2048.5,
            model_info={"model_name": "bge-m3", "dimension": 1024},
        )

        assert result.dense_embeddings == dense_embs
        assert result.sparse_embeddings == sparse_embs
        assert len(result.colbert_embeddings) == 2
        assert isinstance(result.colbert_embeddings[0], np.ndarray)
        assert result.processing_time == 2.5
        assert result.batch_size == 16
        assert result.memory_usage_mb == 2048.5
        assert result.model_info["model_name"] == "bge-m3"

    @pytest.mark.unit
    def test_embedding_result_dense_embeddings_various_shapes(self):
        """Test EmbeddingResult with various dense embedding shapes."""
        # Single embedding
        single = [[0.1] * 1024]
        result1 = EmbeddingResult(
            dense_embeddings=single,
            processing_time=1.0,
            batch_size=1,
            memory_usage_mb=512.0,
        )
        assert len(result1.dense_embeddings) == 1
        assert len(result1.dense_embeddings[0]) == 1024

        # Multiple embeddings
        multiple = [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
        result2 = EmbeddingResult(
            dense_embeddings=multiple,
            processing_time=2.0,
            batch_size=3,
            memory_usage_mb=1536.0,
        )
        assert len(result2.dense_embeddings) == 3

        # Empty embeddings list
        result3 = EmbeddingResult(
            dense_embeddings=[],
            processing_time=0.1,
            batch_size=0,
            memory_usage_mb=0.0,
        )
        assert result3.dense_embeddings == []

    @pytest.mark.unit
    def test_embedding_result_sparse_embeddings_various_formats(self):
        """Test EmbeddingResult with various sparse embedding formats."""
        # Dense sparse (many indices)
        dense_sparse = [dict.fromkeys(range(100), 0.1)]
        result1 = EmbeddingResult(
            sparse_embeddings=dense_sparse,
            processing_time=1.5,
            batch_size=1,
            memory_usage_mb=256.0,
        )
        assert len(result1.sparse_embeddings[0]) == 100

        # Sparse sparse (few indices)
        sparse_sparse = [{10: 0.5, 50: 0.3, 100: 0.2}]
        result2 = EmbeddingResult(
            sparse_embeddings=sparse_sparse,
            processing_time=0.5,
            batch_size=1,
            memory_usage_mb=128.0,
        )
        assert len(result2.sparse_embeddings[0]) == 3

        # Empty sparse embeddings
        empty_sparse = [{}]
        result3 = EmbeddingResult(
            sparse_embeddings=empty_sparse,
            processing_time=0.1,
            batch_size=1,
            memory_usage_mb=64.0,
        )
        assert result3.sparse_embeddings[0] == {}

    @pytest.mark.unit
    def test_embedding_result_colbert_embeddings_numpy_arrays(self):
        """Test EmbeddingResult with ColBERT numpy arrays."""
        # Various numpy array shapes
        arrays = [
            np.random.rand(10, 1024),  # 10 tokens, 1024 dimensions
            np.random.rand(5, 1024),  # 5 tokens, 1024 dimensions
            np.random.rand(1, 1024),  # 1 token, 1024 dimensions
        ]

        result = EmbeddingResult(
            colbert_embeddings=arrays,
            processing_time=3.0,
            batch_size=3,
            memory_usage_mb=4096.0,
        )

        assert len(result.colbert_embeddings) == 3
        assert result.colbert_embeddings[0].shape == (10, 1024)
        assert result.colbert_embeddings[1].shape == (5, 1024)
        assert result.colbert_embeddings[2].shape == (1, 1024)

    @pytest.mark.unit
    def test_embedding_result_performance_metrics(self):
        """Test EmbeddingResult performance and memory tracking."""
        # Test various performance scenarios
        scenarios = [
            {
                "processing_time": 0.1,
                "batch_size": 1,
                "memory_usage_mb": 256.0,
            },  # Fast, small batch
            {
                "processing_time": 5.0,
                "batch_size": 32,
                "memory_usage_mb": 8192.0,
            },  # Slow, large batch
            {
                "processing_time": 0.0,
                "batch_size": 0,
                "memory_usage_mb": 0.0,
            },  # Edge case - no processing
        ]

        for scenario in scenarios:
            result = EmbeddingResult(**scenario)
            assert result.processing_time == scenario["processing_time"]
            assert result.batch_size == scenario["batch_size"]
            assert result.memory_usage_mb == scenario["memory_usage_mb"]

    @pytest.mark.unit
    def test_embedding_result_model_info_various_formats(self):
        """Test EmbeddingResult with various model_info formats."""
        # Comprehensive model info
        comprehensive_info = {
            "model_name": "BAAI/bge-m3",
            "model_version": "1.0.0",
            "dimension": 1024,
            "max_length": 8192,
            "device": "cuda:0",
            "precision": "fp16",
            "parameters": 568_000_000,
        }

        result1 = EmbeddingResult(
            processing_time=1.0,
            batch_size=1,
            memory_usage_mb=1024.0,
            model_info=comprehensive_info,
        )
        assert result1.model_info["model_name"] == "BAAI/bge-m3"
        assert result1.model_info["parameters"] == 568_000_000

        # Minimal model info
        result2 = EmbeddingResult(
            processing_time=1.0,
            batch_size=1,
            memory_usage_mb=1024.0,
            model_info={"name": "simple"},
        )
        assert result2.model_info == {"name": "simple"}

        # Empty model info (default)
        result3 = EmbeddingResult(
            processing_time=1.0, batch_size=1, memory_usage_mb=1024.0
        )
        assert result3.model_info == {}

    @pytest.mark.unit
    def test_embedding_result_serialization_without_numpy(self):
        """Test EmbeddingResult serialization without numpy arrays."""
        result = EmbeddingResult(
            dense_embeddings=[[0.1, 0.2], [0.3, 0.4]],
            sparse_embeddings=[{1: 0.5}, {2: 0.7}],
            processing_time=1.5,
            batch_size=2,
            memory_usage_mb=512.0,
            model_info={"test": "value"},
        )

        # Serialize and deserialize
        json_data = result.model_dump()
        restored = EmbeddingResult.model_validate(json_data)

        assert restored.dense_embeddings == result.dense_embeddings
        assert restored.sparse_embeddings == result.sparse_embeddings
        assert restored.processing_time == result.processing_time
        assert restored.model_info == result.model_info

    @pytest.mark.unit
    def test_embedding_result_validation_types(self):
        """Test EmbeddingResult validation with invalid types."""
        with pytest.raises(ValidationError):
            EmbeddingResult(
                processing_time="invalid",  # Should be float
                batch_size=1,
                memory_usage_mb=512.0,
            )

        with pytest.raises(ValidationError):
            EmbeddingResult(
                processing_time=1.0,
                batch_size="invalid",  # Should be int
                memory_usage_mb=512.0,
            )

        with pytest.raises(ValidationError):
            EmbeddingResult(
                processing_time=1.0,
                batch_size=1,
                memory_usage_mb="invalid",  # Should be float
            )

    @pytest.mark.unit
    def test_embedding_result_negative_values(self):
        """Test EmbeddingResult with negative values."""
        # Negative values should be allowed for some fields (no explicit constraints)
        result = EmbeddingResult(
            processing_time=-1.0,  # Could represent error condition
            batch_size=-1,  # Could represent error condition
            memory_usage_mb=-100.0,  # Could represent error condition
        )

        assert result.processing_time == -1.0
        assert result.batch_size == -1
        assert result.memory_usage_mb == -100.0

    @pytest.mark.unit
    def test_embedding_result_zero_values(self):
        """Test EmbeddingResult with zero values."""
        result = EmbeddingResult(
            processing_time=0.0,
            batch_size=0,
            memory_usage_mb=0.0,
        )

        assert result.processing_time == 0.0
        assert result.batch_size == 0
        assert result.memory_usage_mb == 0.0
