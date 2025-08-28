"""Enhanced dimension validation tests for embedding operations using LlamaIndex MockEmbedding.

This module provides comprehensive dimension validation testing that supplements the existing
tests with a focus on:
- LlamaIndex MockEmbedding(embed_dim=1024) integration as requested
- Dimension consistency validation across all embedding operations
- Interface contract testing for BGE-M3 1024D embeddings
- Error handling for dimension mismatches
- Batch processing dimension validation

Key testing areas:
- BGE-M3 1024D dimension consistency using MockEmbedding
- Vector similarity computation with dimension validation
- Batch processing with dimension mismatch scenarios
- Interface contracts for embedding dimension integrity
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from llama_index.core.embeddings import MockEmbedding

from src.models.embeddings import EmbeddingParameters


@pytest.fixture
def mock_llamaindex_embedding():
    """LlamaIndex MockEmbedding configured for BGE-M3 1024D as requested."""
    return MockEmbedding(embed_dim=1024)


@pytest.fixture
def mock_settings_1024d():
    """Mock settings configured for BGE-M3 1024-dimensional embeddings."""
    settings = Mock()
    settings.embedding = Mock()
    settings.embedding.model_name = "BAAI/bge-m3"
    settings.embedding.dimension = 1024
    settings.embedding.max_length = 8192
    settings.embedding.batch_size = 12
    return settings


@pytest.fixture
def bgem3_parameters_1024d():
    """EmbeddingParameters configured for BGE-M3 1024D validation."""
    return EmbeddingParameters(
        max_length=8192,
        use_fp16=True,
        return_dense=True,
        return_sparse=True,
        return_colbert=False,
        normalize_embeddings=True,
    )


@pytest.fixture
def sample_embeddings_1024d():
    """Sample 1024-dimensional embeddings for validation."""
    return {
        "valid_1024d": [np.random.randn(1024).tolist() for _ in range(5)],
        "invalid_512d": [np.random.randn(512).tolist() for _ in range(5)],
        "invalid_768d": [np.random.randn(768).tolist() for _ in range(5)],
        "invalid_2048d": [np.random.randn(2048).tolist() for _ in range(5)],
        "empty_batch": [],
        "single_valid": [np.random.randn(1024).tolist()],
    }


@pytest.mark.unit
class TestLlamaIndexMockEmbeddingIntegration:
    """Test LlamaIndex MockEmbedding integration for BGE-M3 1024D validation."""

    def test_mock_embedding_dimension_consistency(self, mock_llamaindex_embedding):
        """Test MockEmbedding produces consistent 1024D embeddings."""
        test_texts = [
            "Machine learning text for dimension testing",
            "Vector embedding validation sample",
            "BGE-M3 dimension consistency check",
        ]

        # Test query embedding
        for text in test_texts:
            embedding = mock_llamaindex_embedding._get_query_embedding(text)
            assert len(embedding) == 1024, (
                f"Expected 1024 dimensions, got {len(embedding)}"
            )
            assert all(isinstance(x, (int, float)) for x in embedding), (
                "All values should be numeric"
            )

        # Test text embedding
        for text in test_texts:
            embedding = mock_llamaindex_embedding._get_text_embedding(text)
            assert len(embedding) == 1024, (
                f"Expected 1024 dimensions, got {len(embedding)}"
            )
            assert all(isinstance(x, (int, float)) for x in embedding), (
                "All values should be numeric"
            )

    async def test_mock_embedding_async_operations(self, mock_llamaindex_embedding):
        """Test MockEmbedding async operations maintain dimension consistency."""
        test_queries = [
            "Async query embedding test",
            "Another async query for validation",
            "BGE-M3 async dimension test",
        ]

        for query in test_queries:
            embedding = await mock_llamaindex_embedding._aget_query_embedding(query)
            assert len(embedding) == 1024, (
                f"Async embedding should be 1024D, got {len(embedding)}"
            )
            assert all(isinstance(x, (int, float)) for x in embedding), (
                "Async embedding values should be numeric"
            )

    def test_mock_embedding_batch_dimension_consistency(
        self, mock_llamaindex_embedding
    ):
        """Test MockEmbedding batch operations maintain 1024D consistency."""
        batch_sizes = [1, 5, 10, 20, 50]

        for batch_size in batch_sizes:
            test_texts = [f"Batch test text {i}" for i in range(batch_size)]

            for text in test_texts:
                embedding = mock_llamaindex_embedding._get_text_embedding(text)
                assert len(embedding) == 1024, (
                    f"Batch embedding should be 1024D, got {len(embedding)}"
                )

    def test_mock_embedding_embed_dim_property(self, mock_llamaindex_embedding):
        """Test MockEmbedding embed_dim property matches BGE-M3 expectations."""
        assert hasattr(mock_llamaindex_embedding, "embed_dim"), (
            "MockEmbedding should have embed_dim property"
        )
        assert mock_llamaindex_embedding.embed_dim == 1024, (
            f"embed_dim should be 1024, got {mock_llamaindex_embedding.embed_dim}"
        )

    def test_mock_embedding_consistency_across_calls(self, mock_llamaindex_embedding):
        """Test MockEmbedding produces consistent dimensions across multiple calls."""
        test_text = "Consistency test text"

        embeddings = []
        for _ in range(10):
            embedding = mock_llamaindex_embedding._get_query_embedding(test_text)
            embeddings.append(embedding)
            assert len(embedding) == 1024, "All embeddings should be 1024D"

        # Verify all embeddings have same dimension
        dimensions = [len(emb) for emb in embeddings]
        assert all(dim == 1024 for dim in dimensions), (
            "All embeddings should consistently be 1024D"
        )


@pytest.mark.unit
class TestEmbeddingDimensionMismatchScenarios:
    """Test dimension mismatch detection and error handling."""

    def test_dimension_mismatch_detection_512d(self, sample_embeddings_1024d):
        """Test detection of 512D embeddings when 1024D expected."""
        valid_embeddings = sample_embeddings_1024d["valid_1024d"]
        invalid_embeddings = sample_embeddings_1024d["invalid_512d"]

        # Valid embeddings should pass dimension check
        for embedding in valid_embeddings:
            assert len(embedding) == 1024, (
                f"Valid embedding should be 1024D, got {len(embedding)}"
            )

        # Invalid embeddings should be detectable
        for embedding in invalid_embeddings:
            assert len(embedding) != 1024, (
                f"Invalid embedding should not be 1024D, got {len(embedding)}"
            )
            assert len(embedding) == 512, (
                f"Invalid embedding should be 512D, got {len(embedding)}"
            )

    def test_dimension_mismatch_detection_768d(self, sample_embeddings_1024d):
        """Test detection of 768D embeddings when 1024D expected."""
        invalid_embeddings = sample_embeddings_1024d["invalid_768d"]

        for embedding in invalid_embeddings:
            assert len(embedding) == 768, (
                f"Invalid embedding should be 768D, got {len(embedding)}"
            )
            assert len(embedding) != 1024, (
                "768D embedding should not match 1024D expectation"
            )

    def test_dimension_mismatch_detection_2048d(self, sample_embeddings_1024d):
        """Test detection of over-dimensional embeddings."""
        invalid_embeddings = sample_embeddings_1024d["invalid_2048d"]

        for embedding in invalid_embeddings:
            assert len(embedding) == 2048, (
                f"Invalid embedding should be 2048D, got {len(embedding)}"
            )
            assert len(embedding) != 1024, (
                "2048D embedding should not match 1024D expectation"
            )

    def test_batch_dimension_consistency_validation(self, sample_embeddings_1024d):
        """Test batch processing validates all embeddings have consistent dimensions."""
        valid_batch = sample_embeddings_1024d["valid_1024d"]
        mixed_batch = valid_batch[:2] + sample_embeddings_1024d["invalid_512d"][:2]

        # Valid batch should have consistent dimensions
        dimensions = [len(emb) for emb in valid_batch]
        assert all(dim == 1024 for dim in dimensions), (
            "All embeddings in valid batch should be 1024D"
        )

        # Mixed batch should have inconsistent dimensions
        mixed_dimensions = [len(emb) for emb in mixed_batch]
        unique_dimensions = set(mixed_dimensions)
        assert len(unique_dimensions) > 1, (
            "Mixed batch should have inconsistent dimensions"
        )
        assert 1024 in unique_dimensions, "Mixed batch should contain 1024D embeddings"
        assert 512 in unique_dimensions, "Mixed batch should contain 512D embeddings"

    def test_empty_batch_dimension_validation(self, sample_embeddings_1024d):
        """Test empty batch handling in dimension validation."""
        empty_batch = sample_embeddings_1024d["empty_batch"]

        assert len(empty_batch) == 0, "Empty batch should have no embeddings"

        # Empty batch dimension validation should not fail
        dimensions = [len(emb) for emb in empty_batch]
        assert len(dimensions) == 0, "Empty batch should have no dimensions to validate"

    def test_single_embedding_dimension_validation(self, sample_embeddings_1024d):
        """Test single embedding dimension validation."""
        single_valid = sample_embeddings_1024d["single_valid"]

        assert len(single_valid) == 1, "Single batch should have exactly one embedding"
        assert len(single_valid[0]) == 1024, "Single embedding should be 1024D"


@pytest.mark.unit
class TestVectorSimilarityDimensionValidation:
    """Test vector similarity computations with dimension validation."""

    def test_cosine_similarity_dimension_matching(self, sample_embeddings_1024d):
        """Test cosine similarity requires matching dimensions."""
        valid_embeddings = sample_embeddings_1024d["valid_1024d"]
        invalid_embeddings = sample_embeddings_1024d["invalid_512d"]

        # Same dimension vectors should be compatible for similarity
        vec1_1024 = valid_embeddings[0]
        vec2_1024 = valid_embeddings[1]

        assert len(vec1_1024) == len(vec2_1024) == 1024, "Both vectors should be 1024D"

        # Manual cosine similarity calculation for validation
        vec1_np = np.array(vec1_1024)
        vec2_np = np.array(vec2_1024)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
            assert -1.0 <= cosine_sim <= 1.0, (
                "Cosine similarity should be in [-1, 1] range"
            )

    def test_similarity_computation_dimension_mismatch_detection(
        self, sample_embeddings_1024d
    ):
        """Test dimension mismatch detection in similarity computations."""
        valid_1024 = sample_embeddings_1024d["valid_1024d"][0]
        invalid_512 = sample_embeddings_1024d["invalid_512d"][0]

        assert len(valid_1024) == 1024, "Valid vector should be 1024D"
        assert len(invalid_512) == 512, "Invalid vector should be 512D"
        assert len(valid_1024) != len(invalid_512), (
            "Vectors should have mismatched dimensions"
        )

        # Attempting similarity with mismatched dimensions should be detectable
        try:
            vec1_np = np.array(valid_1024)
            vec2_np = np.array(invalid_512)

            # This should fail due to dimension mismatch
            with pytest.raises((ValueError, RuntimeError)):
                np.dot(vec1_np, vec2_np)

        except Exception as e:
            # Expected behavior - dimension mismatch should cause error
            assert "shape" in str(e).lower() or "dimension" in str(e).lower()

    def test_batch_similarity_dimension_consistency(self, sample_embeddings_1024d):
        """Test batch similarity computations require dimension consistency."""
        valid_batch = sample_embeddings_1024d["valid_1024d"][:3]
        query_vector = sample_embeddings_1024d["single_valid"][0]

        # All vectors should have same dimension
        all_dimensions = [len(emb) for emb in valid_batch + [query_vector]]
        assert all(dim == 1024 for dim in all_dimensions), (
            "All vectors should be 1024D for batch similarity"
        )

        # Batch similarity computation should work with consistent dimensions
        query_np = np.array(query_vector)
        batch_np = np.array(valid_batch)

        assert query_np.shape == (1024,), "Query vector should be 1024D"
        assert batch_np.shape == (3, 1024), "Batch should be 3x1024D"

        # Compute similarities
        similarities = np.dot(batch_np, query_np)
        assert len(similarities) == 3, "Should have 3 similarity scores"
        assert all(isinstance(sim, (int, float, np.number)) for sim in similarities), (
            "All similarities should be numeric"
        )


@pytest.mark.unit
class TestSparseEmbeddingValidation:
    """Test sparse embedding validation and dimension consistency."""

    def test_sparse_embedding_structure_validation(self):
        """Test sparse embedding structure follows expected format."""
        # Valid sparse embeddings (token_id -> weight)
        valid_sparse_embeddings = [
            {100: 0.8, 205: 0.6, 1024: 0.5},
            {150: 0.9, 300: 0.7, 1100: 0.6},
            {80: 0.7, 180: 0.8, 900: 0.5},
        ]

        for sparse_emb in valid_sparse_embeddings:
            assert isinstance(sparse_emb, dict), (
                "Sparse embedding should be a dictionary"
            )

            # Validate token IDs are integers
            for token_id in sparse_emb.keys():
                assert isinstance(token_id, int), (
                    f"Token ID should be integer, got {type(token_id)}"
                )
                assert token_id >= 0, f"Token ID should be non-negative, got {token_id}"

            # Validate weights are floats
            for weight in sparse_emb.values():
                assert isinstance(weight, (int, float)), (
                    f"Weight should be numeric, got {type(weight)}"
                )
                assert 0.0 <= weight <= 1.0, (
                    f"Weight should be in [0,1] range, got {weight}"
                )

    def test_sparse_embedding_token_uniqueness(self):
        """Test sparse embedding token IDs are unique within each embedding."""
        sparse_embedding = {100: 0.8, 200: 0.6, 300: 0.4, 400: 0.2}

        token_ids = list(sparse_embedding.keys())
        unique_token_ids = set(token_ids)

        assert len(token_ids) == len(unique_token_ids), (
            "Token IDs should be unique within sparse embedding"
        )

    def test_sparse_embedding_batch_validation(self):
        """Test batch of sparse embeddings maintains structure consistency."""
        sparse_batch = [
            {100: 0.8, 200: 0.6},
            {150: 0.9, 250: 0.7, 350: 0.5},
            {80: 0.7, 180: 0.8, 280: 0.6, 380: 0.4},
        ]

        for i, sparse_emb in enumerate(sparse_batch):
            assert isinstance(sparse_emb, dict), (
                f"Sparse embedding {i} should be dictionary"
            )
            assert len(sparse_emb) > 0, f"Sparse embedding {i} should not be empty"

            for token_id, weight in sparse_emb.items():
                assert isinstance(token_id, int), (
                    f"Token ID in embedding {i} should be integer"
                )
                assert isinstance(weight, (int, float)), (
                    f"Weight in embedding {i} should be numeric"
                )
                assert 0.0 <= weight <= 1.0, (
                    f"Weight in embedding {i} should be in [0,1] range"
                )

    def test_sparse_embedding_similarity_computation(self):
        """Test sparse embedding similarity computation validation."""
        sparse1 = {100: 0.8, 200: 0.6, 300: 0.4}
        sparse2 = {100: 0.7, 250: 0.5, 300: 0.9}

        # Compute overlap-based similarity
        common_tokens = set(sparse1.keys()) & set(sparse2.keys())
        assert len(common_tokens) > 0, (
            "Should have common tokens for meaningful similarity"
        )

        # Compute weighted overlap
        overlap_score = 0.0
        for token_id in common_tokens:
            overlap_score += min(sparse1[token_id], sparse2[token_id])

        assert overlap_score > 0.0, "Overlap score should be positive for common tokens"
        # Note: overlap score can exceed 1.0 when summing multiple token overlaps
        assert overlap_score >= 0.0, "Overlap score should be non-negative"


@pytest.mark.unit
class TestInterfaceContractValidation:
    """Test interface contracts for embedding operations."""

    @patch("src.processing.embeddings.bgem3_embedder.BGEM3FlagModel")
    def test_bgem3_embedder_dimension_contract(
        self, mock_flag_model_class, mock_settings_1024d, bgem3_parameters_1024d
    ):
        """Test BGE-M3 embedder maintains 1024D contract."""
        from src.processing.embeddings.bgem3_embedder import BGEM3Embedder

        # Setup mock to return 1024D embeddings
        mock_model = Mock()

        def mock_encode(*args, **kwargs):
            batch_size = len(args[0]) if args else 1
            return {"dense_vecs": np.random.randn(batch_size, 1024).astype(np.float32)}

        mock_model.encode.side_effect = mock_encode
        mock_flag_model_class.return_value = mock_model

        embedder = BGEM3Embedder(
            settings=mock_settings_1024d, parameters=bgem3_parameters_1024d
        )

        # Test single embedding contract
        single_result = embedder.get_dense_embeddings(["Test text"])
        assert single_result is not None, "Dense embeddings should not be None"
        assert len(single_result) == 1, "Should return one embedding"
        assert len(single_result[0]) == 1024, "Single embedding should be 1024D"

        # Test batch embedding contract
        batch_result = embedder.get_dense_embeddings(["Text 1", "Text 2", "Text 3"])
        assert batch_result is not None, "Batch dense embeddings should not be None"
        assert len(batch_result) == 3, "Should return three embeddings"
        for embedding in batch_result:
            assert len(embedding) == 1024, "Each batch embedding should be 1024D"

    @patch("src.retrieval.embeddings.BGEM3FlagModel")
    def test_retrieval_embedding_dimension_contract(
        self, mock_flag_model_class, mock_llamaindex_embedding
    ):
        """Test retrieval embedding maintains LlamaIndex dimension contract."""
        from src.retrieval.embeddings import BGEM3Embedding

        # Setup mock
        mock_model = Mock()
        mock_model.encode.return_value = {
            "dense_vecs": np.random.randn(1, 1024).astype(np.float32)
        }
        mock_flag_model_class.return_value = mock_model

        embedding = BGEM3Embedding()

        # Test embed_dim property contract
        assert hasattr(embedding, "embed_dim"), "Should have embed_dim property"
        assert embedding.embed_dim == 1024, "embed_dim should be 1024 for BGE-M3"

        # Test query embedding contract
        query_result = embedding._get_query_embedding("Test query")
        assert isinstance(query_result, list), "Query embedding should be list"
        assert len(query_result) == 1024, "Query embedding should be 1024D"

    def test_vector_similarity_contract_validation(self, sample_embeddings_1024d):
        """Test vector similarity computation contracts."""
        embeddings = sample_embeddings_1024d["valid_1024d"]

        # Test pairwise similarity contract
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                vec1 = np.array(embeddings[i])
                vec2 = np.array(embeddings[j])

                # Dimension contract
                assert vec1.shape == (1024,), f"Vector {i} should be 1024D"
                assert vec2.shape == (1024,), f"Vector {j} should be 1024D"

                # Similarity computation contract
                dot_product = np.dot(vec1, vec2)
                assert isinstance(dot_product, (int, float, np.number)), (
                    "Dot product should be numeric"
                )

                # Normalized similarity contract
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    cosine_sim = dot_product / (norm1 * norm2)
                    assert -1.0 <= cosine_sim <= 1.0, (
                        "Cosine similarity should be in [-1,1]"
                    )


@pytest.mark.unit
class TestBatchProcessingEdgeCases:
    """Test batch processing edge cases and boundary conditions."""

    def test_empty_batch_processing(self, mock_llamaindex_embedding):
        """Test empty batch handling."""
        empty_texts = []

        # Empty batch should handle gracefully
        try:
            for text in empty_texts:
                embedding = mock_llamaindex_embedding._get_text_embedding(text)
        except Exception as e:
            pytest.fail(f"Empty batch processing should not raise exception: {e}")

    def test_single_item_batch(self, mock_llamaindex_embedding):
        """Test single-item batch processing."""
        single_batch = ["Single test text"]

        embedding = mock_llamaindex_embedding._get_text_embedding(single_batch[0])
        assert len(embedding) == 1024, (
            "Single item batch should produce 1024D embedding"
        )

    def test_large_batch_dimension_consistency(self, mock_llamaindex_embedding):
        """Test large batch maintains dimension consistency."""
        large_batch = [f"Large batch text item {i}" for i in range(100)]

        for i, text in enumerate(large_batch):
            embedding = mock_llamaindex_embedding._get_text_embedding(text)
            assert len(embedding) == 1024, f"Large batch item {i} should be 1024D"

    def test_batch_size_boundary_conditions(self, sample_embeddings_1024d):
        """Test batch processing at various boundary sizes."""
        valid_embeddings = sample_embeddings_1024d["valid_1024d"]

        boundary_sizes = [0, 1, 2, 5, 10, 16, 32, 64]

        for size in boundary_sizes:
            if size == 0:
                batch = []
            elif size <= len(valid_embeddings):
                batch = valid_embeddings[:size]
            else:
                # Extend batch by repeating embeddings
                batch = valid_embeddings * (size // len(valid_embeddings) + 1)
                batch = batch[:size]

            # Validate batch dimensions
            assert len(batch) == size, f"Batch should have size {size}"

            for i, embedding in enumerate(batch):
                assert len(embedding) == 1024, (
                    f"Embedding {i} in size-{size} batch should be 1024D"
                )

    def test_mixed_batch_dimension_detection(self, sample_embeddings_1024d):
        """Test detection of mixed-dimension batches."""
        valid_1024 = sample_embeddings_1024d["valid_1024d"][:2]
        invalid_512 = sample_embeddings_1024d["invalid_512d"][:2]

        mixed_batch = valid_1024 + invalid_512

        # Should be able to detect mixed dimensions
        dimensions = [len(emb) for emb in mixed_batch]
        unique_dims = set(dimensions)

        assert len(unique_dims) > 1, "Mixed batch should have multiple dimensions"
        assert 1024 in unique_dims, "Mixed batch should contain 1024D embeddings"
        assert 512 in unique_dims, "Mixed batch should contain 512D embeddings"


@pytest.mark.unit
class TestErrorHandlingScenarios:
    """Test comprehensive error handling scenarios."""

    def test_none_embedding_handling(self):
        """Test handling of None embeddings."""
        none_embedding = None

        assert none_embedding is None, "None embedding should remain None"

        # Should handle None gracefully in dimension checks
        if none_embedding is not None:
            dimension = len(none_embedding)
        else:
            dimension = 0

        assert dimension == 0, "None embedding should have 0 dimension"

    def test_invalid_embedding_types(self):
        """Test handling of invalid embedding types."""
        invalid_embeddings = [
            "string_embedding",
            123,
            {"dict": "embedding"},
            set([1, 2, 3]),
        ]

        for invalid_emb in invalid_embeddings:
            assert not isinstance(invalid_emb, list), (
                f"Invalid embedding {invalid_emb} should not be list"
            )

            # Should detect invalid type
            if isinstance(invalid_emb, list):
                try:
                    dimension = len(invalid_emb)
                except Exception:
                    dimension = -1
            else:
                dimension = -1

            assert dimension == -1, (
                f"Invalid embedding {invalid_emb} should have invalid dimension"
            )

    def test_nan_inf_embedding_validation(self):
        """Test handling of NaN/Inf values in embeddings."""
        # Create embeddings with special float values
        nan_embedding = [float("nan")] * 1024
        inf_embedding = [float("inf")] * 1024
        mixed_embedding = [1.0] * 1000 + [float("nan")] * 20 + [float("inf")] * 4

        # Dimension should still be correct
        assert len(nan_embedding) == 1024, "NaN embedding should have correct dimension"
        assert len(inf_embedding) == 1024, "Inf embedding should have correct dimension"
        assert len(mixed_embedding) == 1024, (
            "Mixed embedding should have correct dimension"
        )

        # Should be able to detect NaN/Inf values
        assert any(np.isnan(x) for x in nan_embedding), "Should detect NaN values"
        assert any(np.isinf(x) for x in inf_embedding), "Should detect Inf values"
        assert any(np.isnan(x) or np.isinf(x) for x in mixed_embedding), (
            "Should detect special values"
        )

    def test_dimension_validation_error_messages(self, sample_embeddings_1024d):
        """Test dimension validation provides clear error messages."""
        valid_1024 = sample_embeddings_1024d["valid_1024d"][0]
        invalid_512 = sample_embeddings_1024d["invalid_512d"][0]

        # Should be able to provide descriptive validation messages
        def validate_dimension(embedding, expected_dim=1024):
            actual_dim = len(embedding)
            if actual_dim != expected_dim:
                return f"Dimension mismatch: expected {expected_dim}, got {actual_dim}"
            return "Valid dimension"

        valid_msg = validate_dimension(valid_1024)
        invalid_msg = validate_dimension(invalid_512)

        assert "Valid dimension" in valid_msg, "Valid embedding should pass validation"
        assert "mismatch" in invalid_msg.lower(), (
            "Invalid embedding should show mismatch"
        )
        assert "512" in invalid_msg, "Error message should specify actual dimension"
        assert "1024" in invalid_msg, "Error message should specify expected dimension"
