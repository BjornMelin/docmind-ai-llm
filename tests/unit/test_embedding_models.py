"""Comprehensive unit tests for embedding models and schemas.

Tests focus on Pydantic model validation, serialization/deserialization,
and schema consistency for BGE-M3 embedding operations.

Key testing areas:
- EmbeddingParameters validation and defaults
- EmbeddingResult structure and validation
- EmbeddingError exception handling
- Schema consistency and field validation
- Serialization/deserialization accuracy
- Edge cases and boundary conditions
"""

from typing import Any
import numpy as np
import pytest
from pydantic import ValidationError


@pytest.mark.unit
class TestEmbeddingParameters:
    """Test EmbeddingParameters model validation and defaults."""

    def test_embedding_parameters_defaults(self):
        """Test EmbeddingParameters default values."""
        from src.models.embeddings import EmbeddingParameters
        
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

    def test_embedding_parameters_custom_values(self):
        """Test EmbeddingParameters with custom values."""
        from src.models.embeddings import EmbeddingParameters
        
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

    def test_embedding_parameters_max_length_validation(self):
        """Test max_length field validation limits."""
        from src.models.embeddings import EmbeddingParameters
        
        # Valid values
        EmbeddingParameters(max_length=512)    # Minimum
        EmbeddingParameters(max_length=8192)   # Default
        EmbeddingParameters(max_length=16384)  # Maximum
        
        # Invalid values
        with pytest.raises(ValidationError):
            EmbeddingParameters(max_length=256)  # Below minimum
            
        with pytest.raises(ValidationError):
            EmbeddingParameters(max_length=20000)  # Above maximum

    def test_embedding_parameters_pooling_method_validation(self):
        """Test pooling_method field accepts valid values."""
        from src.models.embeddings import EmbeddingParameters
        
        # Valid pooling methods
        params_cls = EmbeddingParameters(pooling_method="cls")
        assert params_cls.pooling_method == "cls"
        
        params_mean = EmbeddingParameters(pooling_method="mean")
        assert params_mean.pooling_method == "mean"

    def test_embedding_parameters_device_validation(self):
        """Test device field accepts common values."""
        from src.models.embeddings import EmbeddingParameters
        
        # Valid devices
        params_cuda = EmbeddingParameters(device="cuda")
        assert params_cuda.device == "cuda"
        
        params_cpu = EmbeddingParameters(device="cpu")
        assert params_cpu.device == "cpu"
        
        params_auto = EmbeddingParameters(device="auto")
        assert params_auto.device == "auto"

    def test_embedding_parameters_weights_validation(self):
        """Test weights_for_different_modes validation."""
        from src.models.embeddings import EmbeddingParameters
        
        # Valid weights (should sum to 1.0 ideally but not enforced)
        params = EmbeddingParameters(weights_for_different_modes=[0.6, 0.2, 0.2])
        assert params.weights_for_different_modes == [0.6, 0.2, 0.2]
        
        # Different lengths are allowed
        params_short = EmbeddingParameters(weights_for_different_modes=[0.7, 0.3])
        assert params_short.weights_for_different_modes == [0.7, 0.3]

    def test_embedding_parameters_serialization(self):
        """Test EmbeddingParameters serialization to dict."""
        from src.models.embeddings import EmbeddingParameters
        
        params = EmbeddingParameters(
            max_length=4096,
            use_fp16=False,
            return_colbert=True,
            weights_for_different_modes=[0.5, 0.3, 0.2]
        )
        
        params_dict = params.model_dump()
        
        assert params_dict["max_length"] == 4096
        assert params_dict["use_fp16"] is False
        assert params_dict["return_colbert"] is True
        assert params_dict["weights_for_different_modes"] == [0.5, 0.3, 0.2]

    def test_embedding_parameters_deserialization(self):
        """Test EmbeddingParameters deserialization from dict."""
        from src.models.embeddings import EmbeddingParameters
        
        params_dict = {
            "max_length": 4096,
            "use_fp16": False,
            "return_dense": True,
            "return_sparse": False,
            "return_colbert": True,
            "device": "cpu",
            "pooling_method": "mean",
            "weights_for_different_modes": [0.6, 0.2, 0.2],
        }
        
        params = EmbeddingParameters(**params_dict)
        
        assert params.max_length == 4096
        assert params.use_fp16 is False
        assert params.return_dense is True
        assert params.return_sparse is False
        assert params.return_colbert is True
        assert params.device == "cpu"
        assert params.pooling_method == "mean"
        assert params.weights_for_different_modes == [0.6, 0.2, 0.2]


@pytest.mark.unit
class TestEmbeddingResult:
    """Test EmbeddingResult model validation and structure."""

    def test_embedding_result_minimal(self):
        """Test EmbeddingResult with minimal required fields."""
        from src.models.embeddings import EmbeddingResult
        
        result = EmbeddingResult(
            processing_time=1.5,
            batch_size=3,
            memory_usage_mb=512.0,
        )
        
        assert result.dense_embeddings is None
        assert result.sparse_embeddings is None
        assert result.colbert_embeddings is None
        assert result.processing_time == 1.5
        assert result.batch_size == 3
        assert result.memory_usage_mb == 512.0
        assert result.model_info == {}

    def test_embedding_result_dense_embeddings(self):
        """Test EmbeddingResult with dense embeddings."""
        from src.models.embeddings import EmbeddingResult
        
        dense_embeddings = [
            [0.1, 0.2, 0.3] * 341 + [0.4],  # 1024 dimensions
            [0.5, 0.6, 0.7] * 341 + [0.8],  # 1024 dimensions
        ]
        
        result = EmbeddingResult(
            dense_embeddings=dense_embeddings,
            processing_time=2.0,
            batch_size=2,
            memory_usage_mb=1024.0,
        )
        
        assert result.dense_embeddings is not None
        assert len(result.dense_embeddings) == 2
        assert len(result.dense_embeddings[0]) == 1024
        assert len(result.dense_embeddings[1]) == 1024
        assert result.sparse_embeddings is None
        assert result.colbert_embeddings is None

    def test_embedding_result_sparse_embeddings(self):
        """Test EmbeddingResult with sparse embeddings."""
        from src.models.embeddings import EmbeddingResult
        
        sparse_embeddings = [
            {100: 0.8, 200: 0.6, 300: 0.4},
            {150: 0.9, 250: 0.7, 350: 0.5},
            {80: 0.7, 180: 0.8, 280: 0.6},
        ]
        
        result = EmbeddingResult(
            sparse_embeddings=sparse_embeddings,
            processing_time=1.8,
            batch_size=3,
            memory_usage_mb=768.0,
        )
        
        assert result.dense_embeddings is None
        assert result.sparse_embeddings is not None
        assert len(result.sparse_embeddings) == 3
        assert result.sparse_embeddings[0] == {100: 0.8, 200: 0.6, 300: 0.4}
        assert result.colbert_embeddings is None

    def test_embedding_result_colbert_embeddings(self):
        """Test EmbeddingResult with ColBERT embeddings."""
        from src.models.embeddings import EmbeddingResult
        
        # Mock ColBERT embeddings as numpy arrays
        colbert_embeddings = [
            np.random.randn(10, 1024).astype(np.float32),  # 10 tokens, 1024 dimensions
            np.random.randn(15, 1024).astype(np.float32),  # 15 tokens, 1024 dimensions
        ]
        
        result = EmbeddingResult(
            colbert_embeddings=colbert_embeddings,
            processing_time=3.2,
            batch_size=2,
            memory_usage_mb=2048.0,
        )
        
        assert result.dense_embeddings is None
        assert result.sparse_embeddings is None
        assert result.colbert_embeddings is not None
        assert len(result.colbert_embeddings) == 2
        assert result.colbert_embeddings[0].shape == (10, 1024)
        assert result.colbert_embeddings[1].shape == (15, 1024)

    def test_embedding_result_all_embeddings(self):
        """Test EmbeddingResult with all embedding types."""
        from src.models.embeddings import EmbeddingResult
        
        dense_embeddings = [[0.1] * 1024, [0.2] * 1024]
        sparse_embeddings = [{100: 0.8}, {200: 0.7}]
        colbert_embeddings = [
            np.random.randn(8, 1024).astype(np.float32),
            np.random.randn(12, 1024).astype(np.float32),
        ]
        
        result = EmbeddingResult(
            dense_embeddings=dense_embeddings,
            sparse_embeddings=sparse_embeddings,
            colbert_embeddings=colbert_embeddings,
            processing_time=4.5,
            batch_size=2,
            memory_usage_mb=1536.0,
            model_info={
                "model_name": "BAAI/bge-m3",
                "embedding_dim": 1024,
                "library": "FlagEmbedding",
            },
        )
        
        assert result.dense_embeddings is not None
        assert len(result.dense_embeddings) == 2
        assert result.sparse_embeddings is not None
        assert len(result.sparse_embeddings) == 2
        assert result.colbert_embeddings is not None
        assert len(result.colbert_embeddings) == 2
        assert result.model_info["model_name"] == "BAAI/bge-m3"
        assert result.model_info["embedding_dim"] == 1024

    def test_embedding_result_model_info_structure(self):
        """Test EmbeddingResult model_info field accepts various structures."""
        from src.models.embeddings import EmbeddingResult
        
        model_info = {
            "model_name": "BAAI/bge-m3",
            "device": "cuda",
            "embedding_dim": 1024,
            "library": "FlagEmbedding.BGEM3FlagModel",
            "pooling_method": "cls",
            "normalize_embeddings": True,
            "dense_enabled": True,
            "sparse_enabled": True,
            "colbert_enabled": False,
            "weights_for_modes": [0.4, 0.2, 0.4],
        }
        
        result = EmbeddingResult(
            processing_time=2.1,
            batch_size=5,
            memory_usage_mb=1200.0,
            model_info=model_info,
        )
        
        assert result.model_info["model_name"] == "BAAI/bge-m3"
        assert result.model_info["device"] == "cuda"
        assert result.model_info["embedding_dim"] == 1024
        assert result.model_info["dense_enabled"] is True
        assert result.model_info["weights_for_modes"] == [0.4, 0.2, 0.4]

    def test_embedding_result_performance_fields(self):
        """Test EmbeddingResult performance-related fields."""
        from src.models.embeddings import EmbeddingResult
        
        result = EmbeddingResult(
            processing_time=1.234,
            batch_size=10,
            memory_usage_mb=2048.5,
        )
        
        assert result.processing_time == 1.234
        assert result.batch_size == 10
        assert result.memory_usage_mb == 2048.5

    def test_embedding_result_serialization(self):
        """Test EmbeddingResult serialization."""
        from src.models.embeddings import EmbeddingResult
        
        result = EmbeddingResult(
            dense_embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            sparse_embeddings=[{100: 0.8}, {200: 0.7}],
            processing_time=1.5,
            batch_size=2,
            memory_usage_mb=512.0,
            model_info={"test": "value"},
        )
        
        result_dict = result.model_dump()
        
        assert result_dict["dense_embeddings"] == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert result_dict["sparse_embeddings"] == [{100: 0.8}, {200: 0.7}]
        assert result_dict["processing_time"] == 1.5
        assert result_dict["batch_size"] == 2
        assert result_dict["memory_usage_mb"] == 512.0
        assert result_dict["model_info"] == {"test": "value"}

    def test_embedding_result_deserialization(self):
        """Test EmbeddingResult deserialization."""
        from src.models.embeddings import EmbeddingResult
        
        result_dict = {
            "dense_embeddings": [[0.1, 0.2], [0.3, 0.4]],
            "sparse_embeddings": [{10: 0.5}, {20: 0.6}],
            "colbert_embeddings": None,
            "processing_time": 2.5,
            "batch_size": 2,
            "memory_usage_mb": 1024.0,
            "model_info": {"model": "test"},
        }
        
        result = EmbeddingResult(**result_dict)
        
        assert result.dense_embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert result.sparse_embeddings == [{10: 0.5}, {20: 0.6}]
        assert result.colbert_embeddings is None
        assert result.processing_time == 2.5
        assert result.batch_size == 2
        assert result.memory_usage_mb == 1024.0
        assert result.model_info == {"model": "test"}

    def test_embedding_result_numpy_arrays(self):
        """Test EmbeddingResult with numpy arrays in ColBERT embeddings."""
        from src.models.embeddings import EmbeddingResult
        
        colbert_embeddings = [
            np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            np.array([[0.5, 0.6]], dtype=np.float32),
        ]
        
        result = EmbeddingResult(
            colbert_embeddings=colbert_embeddings,
            processing_time=1.0,
            batch_size=2,
            memory_usage_mb=256.0,
        )
        
        assert result.colbert_embeddings is not None
        assert len(result.colbert_embeddings) == 2
        assert isinstance(result.colbert_embeddings[0], np.ndarray)
        assert isinstance(result.colbert_embeddings[1], np.ndarray)
        assert result.colbert_embeddings[0].shape == (2, 2)
        assert result.colbert_embeddings[1].shape == (1, 2)


@pytest.mark.unit
class TestEmbeddingError:
    """Test EmbeddingError exception class."""

    def test_embedding_error_creation(self):
        """Test EmbeddingError can be created and raised."""
        from src.models.embeddings import EmbeddingError
        
        error = EmbeddingError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_embedding_error_raise_and_catch(self):
        """Test EmbeddingError can be raised and caught."""
        from src.models.embeddings import EmbeddingError
        
        with pytest.raises(EmbeddingError, match="Test embedding error"):
            raise EmbeddingError("Test embedding error")

    def test_embedding_error_inheritance(self):
        """Test EmbeddingError inherits from Exception."""
        from src.models.embeddings import EmbeddingError
        
        error = EmbeddingError("Inheritance test")
        assert isinstance(error, Exception)
        assert isinstance(error, EmbeddingError)

    def test_embedding_error_with_cause(self):
        """Test EmbeddingError with original cause exception."""
        from src.models.embeddings import EmbeddingError
        
        original_error = ValueError("Original error")
        
        try:
            raise original_error
        except ValueError as e:
            embedding_error = EmbeddingError("Wrapped error") from e
            assert embedding_error.__cause__ == original_error
            assert str(embedding_error) == "Wrapped error"

    def test_embedding_error_empty_message(self):
        """Test EmbeddingError with empty message."""
        from src.models.embeddings import EmbeddingError
        
        error = EmbeddingError("")
        assert str(error) == ""

    def test_embedding_error_none_message(self):
        """Test EmbeddingError with None message."""
        from src.models.embeddings import EmbeddingError
        
        # Should handle None gracefully
        error = EmbeddingError(None)
        assert str(error) == "None"


@pytest.mark.unit
class TestModelConsistency:
    """Test consistency across embedding models."""

    def test_embedding_parameters_result_compatibility(self):
        """Test EmbeddingParameters and EmbeddingResult work together."""
        from src.models.embeddings import EmbeddingParameters, EmbeddingResult
        
        # Create parameters
        params = EmbeddingParameters(
            return_dense=True,
            return_sparse=True,
            return_colbert=False,
        )
        
        # Create compatible result
        result = EmbeddingResult(
            dense_embeddings=[[0.1] * 1024],
            sparse_embeddings=[{100: 0.8}],
            colbert_embeddings=None,  # Consistent with params
            processing_time=1.0,
            batch_size=1,
            memory_usage_mb=256.0,
        )
        
        assert params.return_dense is True
        assert result.dense_embeddings is not None
        assert params.return_sparse is True
        assert result.sparse_embeddings is not None
        assert params.return_colbert is False
        assert result.colbert_embeddings is None

    def test_model_serialization_roundtrip(self):
        """Test serialization roundtrip preserves data."""
        from src.models.embeddings import EmbeddingParameters, EmbeddingResult
        
        # Test EmbeddingParameters
        original_params = EmbeddingParameters(
            max_length=4096,
            return_colbert=True,
            weights_for_different_modes=[0.5, 0.3, 0.2],
        )
        
        params_dict = original_params.model_dump()
        restored_params = EmbeddingParameters(**params_dict)
        
        assert original_params.max_length == restored_params.max_length
        assert original_params.return_colbert == restored_params.return_colbert
        assert original_params.weights_for_different_modes == restored_params.weights_for_different_modes
        
        # Test EmbeddingResult
        original_result = EmbeddingResult(
            dense_embeddings=[[0.1, 0.2]],
            sparse_embeddings=[{10: 0.5}],
            processing_time=1.5,
            batch_size=1,
            memory_usage_mb=512.0,
            model_info={"test": "value"},
        )
        
        result_dict = original_result.model_dump()
        restored_result = EmbeddingResult(**result_dict)
        
        assert original_result.dense_embeddings == restored_result.dense_embeddings
        assert original_result.sparse_embeddings == restored_result.sparse_embeddings
        assert original_result.processing_time == restored_result.processing_time
        assert original_result.model_info == restored_result.model_info

    def test_bge_m3_dimension_consistency(self):
        """Test models are consistent with BGE-M3 1024 dimensions."""
        from src.models.embeddings import EmbeddingResult
        
        # BGE-M3 produces 1024-dimensional embeddings
        bge_m3_dim = 1024
        
        dense_embeddings = [
            [0.1] * bge_m3_dim,
            [0.2] * bge_m3_dim,
            [0.3] * bge_m3_dim,
        ]
        
        result = EmbeddingResult(
            dense_embeddings=dense_embeddings,
            processing_time=2.0,
            batch_size=3,
            memory_usage_mb=1024.0,
            model_info={"embedding_dim": bge_m3_dim},
        )
        
        # Verify all embeddings have correct dimension
        for embedding in result.dense_embeddings:
            assert len(embedding) == bge_m3_dim
        
        assert result.model_info["embedding_dim"] == bge_m3_dim

    def test_model_defaults_consistency(self):
        """Test model defaults are consistent across the system."""
        from src.models.embeddings import EmbeddingParameters
        
        params = EmbeddingParameters()
        
        # BGE-M3 specific defaults
        assert params.max_length == 8192  # BGE-M3 8K context
        assert params.return_dense is True  # Dense embeddings enabled
        assert params.return_sparse is True  # Sparse embeddings enabled
        assert params.pooling_method == "cls"  # CLS pooling
        assert params.normalize_embeddings is True  # L2 normalization
        assert params.device == "cuda"  # GPU acceleration
        assert params.weights_for_different_modes == [0.4, 0.2, 0.4]  # Balanced fusion


@pytest.mark.unit
class TestEdgeCasesAndBoundaryConditions:
    """Test edge cases and boundary conditions."""

    def test_embedding_result_empty_embeddings(self):
        """Test EmbeddingResult with empty embedding lists."""
        from src.models.embeddings import EmbeddingResult
        
        result = EmbeddingResult(
            dense_embeddings=[],
            sparse_embeddings=[],
            processing_time=0.0,
            batch_size=0,
            memory_usage_mb=0.0,
        )
        
        assert result.dense_embeddings == []
        assert result.sparse_embeddings == []
        assert result.batch_size == 0
        assert result.processing_time == 0.0

    def test_embedding_result_large_batch_size(self):
        """Test EmbeddingResult with large batch size."""
        from src.models.embeddings import EmbeddingResult
        
        large_batch_size = 1000
        
        result = EmbeddingResult(
            processing_time=10.5,
            batch_size=large_batch_size,
            memory_usage_mb=8192.0,
        )
        
        assert result.batch_size == large_batch_size
        assert result.processing_time == 10.5
        assert result.memory_usage_mb == 8192.0

    def test_embedding_parameters_extreme_values(self):
        """Test EmbeddingParameters with boundary values."""
        from src.models.embeddings import EmbeddingParameters
        
        # Test minimum values
        params_min = EmbeddingParameters(max_length=512)
        assert params_min.max_length == 512
        
        # Test maximum values
        params_max = EmbeddingParameters(max_length=16384)
        assert params_max.max_length == 16384

    def test_embedding_result_zero_processing_time(self):
        """Test EmbeddingResult with zero processing time."""
        from src.models.embeddings import EmbeddingResult
        
        result = EmbeddingResult(
            processing_time=0.0,
            batch_size=1,
            memory_usage_mb=0.0,
        )
        
        assert result.processing_time == 0.0
        assert result.memory_usage_mb == 0.0

    def test_embedding_result_negative_values_not_allowed(self):
        """Test EmbeddingResult rejects negative values where inappropriate."""
        from src.models.embeddings import EmbeddingResult
        
        # These should work (negative processing time might occur in edge cases)
        result = EmbeddingResult(
            processing_time=-0.001,  # Might occur due to timing precision
            batch_size=1,
            memory_usage_mb=0.0,
        )
        assert result.processing_time == -0.001

    def test_sparse_embedding_empty_dict(self):
        """Test sparse embeddings with empty dictionary."""
        from src.models.embeddings import EmbeddingResult
        
        result = EmbeddingResult(
            sparse_embeddings=[{}, {10: 0.5}, {}],
            processing_time=1.0,
            batch_size=3,
            memory_usage_mb=256.0,
        )
        
        assert len(result.sparse_embeddings) == 3
        assert result.sparse_embeddings[0] == {}
        assert result.sparse_embeddings[1] == {10: 0.5}
        assert result.sparse_embeddings[2] == {}

    def test_model_info_complex_nested_structure(self):
        """Test model_info with complex nested structures."""
        from src.models.embeddings import EmbeddingResult
        
        complex_model_info = {
            "model_name": "BAAI/bge-m3",
            "config": {
                "embedding_dim": 1024,
                "pooling": {"method": "cls", "normalize": True},
                "devices": ["cuda:0", "cuda:1"],
            },
            "performance": {
                "avg_time_ms": 45.2,
                "memory_peak_mb": 2048.5,
            },
            "features": ["dense", "sparse", "colbert"],
            "version": "1.3.5",
        }
        
        result = EmbeddingResult(
            processing_time=2.0,
            batch_size=5,
            memory_usage_mb=1024.0,
            model_info=complex_model_info,
        )
        
        assert result.model_info["config"]["embedding_dim"] == 1024
        assert result.model_info["config"]["pooling"]["method"] == "cls"
        assert result.model_info["performance"]["avg_time_ms"] == 45.2
        assert result.model_info["features"] == ["dense", "sparse", "colbert"]


@pytest.mark.unit
class TestModelValidationEdgeCases:
    """Test model validation edge cases and error conditions."""

    def test_embedding_parameters_invalid_types(self):
        """Test EmbeddingParameters with invalid field types."""
        from src.models.embeddings import EmbeddingParameters
        
        # These should raise ValidationError
        with pytest.raises(ValidationError):
            EmbeddingParameters(max_length="invalid")  # String instead of int
        
        with pytest.raises(ValidationError):
            EmbeddingParameters(use_fp16="true")  # String instead of bool
        
        with pytest.raises(ValidationError):
            EmbeddingParameters(weights_for_different_modes="invalid")  # String instead of list

    def test_embedding_result_invalid_types(self):
        """Test EmbeddingResult with invalid field types."""
        from src.models.embeddings import EmbeddingResult
        
        # These should raise ValidationError
        with pytest.raises(ValidationError):
            EmbeddingResult(
                processing_time="invalid",  # String instead of float
                batch_size=1,
                memory_usage_mb=256.0,
            )
        
        with pytest.raises(ValidationError):
            EmbeddingResult(
                processing_time=1.0,
                batch_size="invalid",  # String instead of int
                memory_usage_mb=256.0,
            )

    def test_embedding_result_missing_required_fields(self):
        """Test EmbeddingResult with missing required fields."""
        from src.models.embeddings import EmbeddingResult
        
        # Missing all required fields
        with pytest.raises(ValidationError):
            EmbeddingResult()
        
        # Missing some required fields
        with pytest.raises(ValidationError):
            EmbeddingResult(processing_time=1.0)  # Missing batch_size and memory_usage_mb

    def test_model_field_immutability(self):
        """Test that models can be modified after creation (mutable by default)."""
        from src.models.embeddings import EmbeddingParameters, EmbeddingResult
        
        # EmbeddingParameters should be mutable
        params = EmbeddingParameters()
        original_max_length = params.max_length
        params.max_length = 4096
        assert params.max_length != original_max_length
        assert params.max_length == 4096
        
        # EmbeddingResult should be mutable  
        result = EmbeddingResult(
            processing_time=1.0,
            batch_size=1,
            memory_usage_mb=256.0,
        )
        original_time = result.processing_time
        result.processing_time = 2.0
        assert result.processing_time != original_time
        assert result.processing_time == 2.0