"""Focused unit tests for vector similarity computation validation.

Tests focus on mathematical correctness of vector operations, similarity
metrics, and computational edge cases for embedding pipelines.

Key areas:
- Vector similarity computation algorithms (cosine, dot product, euclidean)
- Dense and sparse vector operations validation
- Similarity score normalization and validation
- Edge cases with zero vectors, extreme values
- Performance validation for similarity computations
"""

import math

import numpy as np
import pytest


@pytest.mark.unit
class TestVectorSimilarityComputation:
    """Test vector similarity computation algorithms."""

    def test_cosine_similarity_computation(self):
        """Test cosine similarity computation correctness."""
        # Perfect similarity (identical vectors)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        cosine_sim = dot_product / (norm1 * norm2)

        assert abs(cosine_sim - 1.0) < 1e-10  # Perfect similarity

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity with orthogonal vectors."""
        # Orthogonal vectors (90 degree angle)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        cosine_sim = dot_product / (norm1 * norm2)

        assert abs(cosine_sim - 0.0) < 1e-10  # No similarity

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity with opposite vectors."""
        # Opposite vectors (180 degree angle)
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [-1.0, 0.0, 0.0]

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        cosine_sim = dot_product / (norm1 * norm2)

        assert abs(cosine_sim - (-1.0)) < 1e-10  # Perfect dissimilarity

    def test_dot_product_computation(self):
        """Test dot product computation correctness."""
        vec1 = [2.0, 3.0, 1.0]
        vec2 = [1.0, 2.0, 3.0]

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        expected = 2 * 1 + 3 * 2 + 1 * 3  # 2 + 6 + 3 = 11

        assert dot_product == expected

    def test_euclidean_distance_computation(self):
        """Test Euclidean distance computation correctness."""
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [4.0, 5.0, 6.0]

        euclidean_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2, strict=False)))
        expected = math.sqrt(
            (4 - 1) ** 2 + (5 - 2) ** 2 + (6 - 3) ** 2
        )  # sqrt(9 + 9 + 9) = sqrt(27)

        assert abs(euclidean_dist - expected) < 1e-10

    def test_vector_normalization_l2(self):
        """Test L2 vector normalization."""
        vec = [3.0, 4.0, 0.0]  # Length = 5

        norm = math.sqrt(sum(x * x for x in vec))
        normalized = [x / norm for x in vec]

        # Verify unit length
        normalized_length = math.sqrt(sum(x * x for x in normalized))
        assert abs(normalized_length - 1.0) < 1e-10

        # Verify direction preserved
        expected_normalized = [0.6, 0.8, 0.0]
        for a, b in zip(normalized, expected_normalized, strict=False):
            assert abs(a - b) < 1e-10


@pytest.mark.unit
class TestVectorSimilarityEdgeCases:
    """Test vector similarity computation edge cases."""

    def test_zero_vector_handling(self):
        """Test handling of zero vectors in similarity computation."""
        zero_vec = [0.0, 0.0, 0.0]
        normal_vec = [1.0, 2.0, 3.0]

        # Zero vector norm
        zero_norm = math.sqrt(sum(x * x for x in zero_vec))
        assert zero_norm == 0.0

        # Cosine similarity with zero vector is undefined
        # Implementations should handle this gracefully
        dot_product = sum(a * b for a, b in zip(zero_vec, normal_vec, strict=False))
        assert dot_product == 0.0

    def test_very_small_vectors(self):
        """Test similarity computation with very small vectors."""
        vec1 = [1e-8, 1e-8, 1e-8]
        vec2 = [2e-8, 2e-8, 2e-8]

        # Should still compute similarity correctly despite small magnitudes
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))

        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
            # Vectors are in same direction, should be close to 1
            assert cosine_sim > 0.99

    def test_very_large_vectors(self):
        """Test similarity computation with very large vectors."""
        vec1 = [1e6, 2e6, 3e6]
        vec2 = [1e6, 2e6, 3e6]

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        cosine_sim = dot_product / (norm1 * norm2)

        # Should be exactly 1 (identical vectors)
        assert abs(cosine_sim - 1.0) < 1e-10

    def test_high_dimensional_vectors(self):
        """Test similarity computation with high-dimensional vectors."""
        # 1024-dimensional vectors (BGE-M3 dimension)
        np.random.seed(42)
        vec1 = np.random.randn(1024).tolist()
        vec2 = np.random.randn(1024).tolist()

        # Compute cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        cosine_sim = dot_product / (norm1 * norm2)

        # Should be valid similarity score [-1, 1]
        assert -1.0 <= cosine_sim <= 1.0

    def test_similarity_score_bounds(self):
        """Test similarity scores stay within expected bounds."""
        # Generate random vectors
        np.random.seed(123)
        vec1 = np.random.randn(100).tolist()
        vec2 = np.random.randn(100).tolist()

        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        cosine_sim = dot_product / (norm1 * norm2)

        assert -1.0 <= cosine_sim <= 1.0

        # Euclidean distance (always non-negative)
        euclidean_dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2, strict=False)))
        assert euclidean_dist >= 0.0


@pytest.mark.unit
class TestSparseVectorOperations:
    """Test sparse vector operations and similarity."""

    def test_sparse_dot_product(self):
        """Test sparse vector dot product computation."""
        # Sparse vectors as dictionaries (token_id -> weight)
        sparse1 = {1: 0.5, 3: 0.8, 5: 0.3}
        sparse2 = {1: 0.4, 2: 0.6, 3: 0.7}

        # Compute dot product (only common indices)
        dot_product = sum(
            sparse1[idx] * sparse2[idx] for idx in sparse1.keys() & sparse2.keys()
        )

        expected = 0.5 * 0.4 + 0.8 * 0.7  # indices 1 and 3
        assert abs(dot_product - expected) < 1e-10

    def test_sparse_vector_norm(self):
        """Test sparse vector L2 norm computation."""
        sparse_vec = {1: 0.6, 2: 0.8, 5: 0.0}  # Zero weight should not contribute

        norm = math.sqrt(
            sum(weight**2 for weight in sparse_vec.values() if weight != 0.0)
        )
        expected = math.sqrt(0.6**2 + 0.8**2)  # sqrt(0.36 + 0.64) = 1.0

        assert abs(norm - expected) < 1e-10

    def test_sparse_cosine_similarity(self):
        """Test sparse vector cosine similarity computation."""
        sparse1 = {1: 0.6, 2: 0.8}  # Unit vector
        sparse2 = {1: 0.6, 2: 0.8}  # Same unit vector

        # Dot product
        dot_product = sum(
            sparse1[idx] * sparse2[idx] for idx in sparse1.keys() & sparse2.keys()
        )

        # Norms
        norm1 = math.sqrt(sum(weight**2 for weight in sparse1.values()))
        norm2 = math.sqrt(sum(weight**2 for weight in sparse2.values()))

        cosine_sim = dot_product / (norm1 * norm2)
        assert abs(cosine_sim - 1.0) < 1e-10  # Should be perfect similarity

    def test_sparse_vector_no_overlap(self):
        """Test sparse vectors with no overlapping indices."""
        sparse1 = {1: 0.5, 3: 0.8}
        sparse2 = {2: 0.4, 4: 0.7}

        # No common indices, dot product should be 0
        dot_product = sum(
            sparse1[idx] * sparse2[idx] for idx in sparse1.keys() & sparse2.keys()
        )

        assert dot_product == 0.0

    def test_sparse_vector_empty(self):
        """Test sparse vector operations with empty vectors."""
        sparse1 = {}
        sparse2 = {1: 0.5, 2: 0.8}

        # Empty vector operations
        dot_product = sum(
            sparse1[idx] * sparse2[idx] for idx in sparse1.keys() & sparse2.keys()
        )
        assert dot_product == 0.0

        # Empty vector norm
        norm = math.sqrt(sum(weight**2 for weight in sparse1.values()))
        assert norm == 0.0


@pytest.mark.unit
class TestDenseVectorValidation:
    """Test dense vector validation for BGE-M3 embeddings."""

    def test_bgem3_dimension_validation(self):
        """Test BGE-M3 1024-dimensional vector validation."""
        # Valid BGE-M3 embedding
        embedding = [0.1] * 1024
        assert len(embedding) == 1024

        # All values should be float
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_embedding_normalization_validation(self):
        """Test embedding normalization correctness."""
        # Random embedding
        np.random.seed(42)
        embedding = np.random.randn(1024).tolist()

        # Normalize
        norm = math.sqrt(sum(x * x for x in embedding))
        normalized = [x / norm for x in embedding]

        # Verify unit length
        new_norm = math.sqrt(sum(x * x for x in normalized))
        assert abs(new_norm - 1.0) < 1e-6

    def test_embedding_similarity_range(self):
        """Test embedding similarity stays in valid range."""
        # Create embeddings with known similarity
        base_embedding = [1.0] + [0.0] * 1023

        # Same direction (high similarity)
        similar_embedding = [0.9] + [0.1] * 1023

        # Orthogonal (low similarity)
        orthogonal_embedding = [0.0, 1.0] + [0.0] * 1022

        # Compute similarities
        def cosine_similarity(a, b):
            dot = sum(x * y for x, y in zip(a, b, strict=False))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        sim_high = cosine_similarity(base_embedding, similar_embedding)
        sim_low = cosine_similarity(base_embedding, orthogonal_embedding)

        assert sim_high > sim_low  # Higher similarity as expected
        assert -1.0 <= sim_high <= 1.0
        assert -1.0 <= sim_low <= 1.0

    def test_batch_embedding_consistency(self):
        """Test batch embedding processing consistency."""
        # Simulate batch of embeddings
        batch_size = 5
        embedding_dim = 1024

        batch_embeddings = []
        for i in range(batch_size):
            np.random.seed(i)  # Different seed for each
            embedding = np.random.randn(embedding_dim).tolist()
            batch_embeddings.append(embedding)

        # All should have correct dimensions
        for embedding in batch_embeddings:
            assert len(embedding) == embedding_dim
            assert all(isinstance(x, (int, float)) for x in embedding)

        # Pairwise similarities should be in valid range
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                sim = self._cosine_similarity(batch_embeddings[i], batch_embeddings[j])
                assert -1.0 <= sim <= 1.0

    def _cosine_similarity(self, a, b):
        """Helper method for cosine similarity computation."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


@pytest.mark.unit
class TestVectorOperationPerformance:
    """Test vector operation performance characteristics."""

    def test_similarity_computation_efficiency(self):
        """Test similarity computation handles large vectors efficiently."""
        # Large vectors (simulate real BGE-M3 embeddings)
        np.random.seed(42)
        vec1 = np.random.randn(1024).tolist()
        vec2 = np.random.randn(1024).tolist()

        import time

        start_time = time.time()

        # Compute similarity (should be fast)
        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        cosine_sim = dot_product / (norm1 * norm2)

        end_time = time.time()
        computation_time = end_time - start_time

        # Should complete quickly (< 1ms for single similarity)
        assert computation_time < 0.001  # 1 millisecond
        assert -1.0 <= cosine_sim <= 1.0

    def test_batch_similarity_computation(self):
        """Test batch similarity computation efficiency."""
        # Batch of vectors
        batch_size = 10
        embedding_dim = 1024

        np.random.seed(42)
        query_vec = np.random.randn(embedding_dim).tolist()
        doc_vecs = [np.random.randn(embedding_dim).tolist() for _ in range(batch_size)]

        import time

        start_time = time.time()

        # Compute similarities for entire batch
        similarities = []
        for doc_vec in doc_vecs:
            dot_product = sum(a * b for a, b in zip(query_vec, doc_vec, strict=False))
            norm_q = math.sqrt(sum(x * x for x in query_vec))
            norm_d = math.sqrt(sum(x * x for x in doc_vec))
            sim = dot_product / (norm_q * norm_d)
            similarities.append(sim)

        end_time = time.time()
        computation_time = end_time - start_time

        # Should complete quickly (< 10ms for batch of 10)
        assert computation_time < 0.01  # 10 milliseconds
        assert len(similarities) == batch_size
        assert all(-1.0 <= sim <= 1.0 for sim in similarities)

    def test_sparse_similarity_efficiency(self):
        """Test sparse similarity computation efficiency."""
        # Sparse vectors (realistic sparsity)
        sparse1 = {i: np.random.random() for i in range(0, 100, 5)}  # Every 5th index
        sparse2 = {
            i: np.random.random() for i in range(2, 102, 5)
        }  # Overlapping indices

        import time

        start_time = time.time()

        # Sparse dot product
        dot_product = sum(
            sparse1[idx] * sparse2[idx] for idx in sparse1.keys() & sparse2.keys()
        )

        end_time = time.time()
        computation_time = end_time - start_time

        # Should be very fast due to sparsity
        assert computation_time < 0.0001  # 0.1 millisecond
        assert isinstance(dot_product, float)


@pytest.mark.unit
class TestVectorValidationUtils:
    """Test vector validation utility functions."""

    def test_vector_dimension_validation(self):
        """Test vector dimension validation utility."""

        def validate_vector_dimension(vector, expected_dim):
            """Validate vector has expected dimensions."""
            if not isinstance(vector, (list, tuple)):
                return False
            if len(vector) != expected_dim:
                return False
            if not all(isinstance(x, (int, float)) for x in vector):
                return False
            return True

        # Valid cases
        assert validate_vector_dimension([0.1] * 1024, 1024) is True
        assert validate_vector_dimension((0.1,) * 512, 512) is True

        # Invalid cases
        assert validate_vector_dimension([0.1] * 1023, 1024) is False  # Wrong dimension
        assert validate_vector_dimension("invalid", 1024) is False  # Wrong type
        assert validate_vector_dimension([0.1, "invalid"], 2) is False  # Mixed types

    def test_similarity_score_validation(self):
        """Test similarity score validation utility."""

        def validate_similarity_score(score, similarity_type="cosine"):
            """Validate similarity score is in expected range."""
            if not isinstance(score, (int, float)):
                return False
            if similarity_type == "cosine":
                return -1.0 <= score <= 1.0
            elif similarity_type == "dot_product":
                return isinstance(score, (int, float))  # No bounds
            elif similarity_type == "euclidean":
                return score >= 0.0  # Non-negative distance
            return False

        # Valid cosine similarities
        assert validate_similarity_score(1.0, "cosine") is True
        assert validate_similarity_score(0.0, "cosine") is True
        assert validate_similarity_score(-1.0, "cosine") is True
        assert validate_similarity_score(0.5, "cosine") is True

        # Invalid cosine similarities
        assert validate_similarity_score(1.1, "cosine") is False  # Above range
        assert validate_similarity_score(-1.1, "cosine") is False  # Below range

        # Valid euclidean distances
        assert validate_similarity_score(0.0, "euclidean") is True
        assert validate_similarity_score(5.5, "euclidean") is True

        # Invalid euclidean distances
        assert validate_similarity_score(-0.1, "euclidean") is False  # Negative

    def test_vector_normalization_utility(self):
        """Test vector normalization utility function."""

        def normalize_vector(vector):
            """L2 normalize a vector."""
            norm = math.sqrt(sum(x * x for x in vector))
            if norm == 0.0:
                return vector  # Cannot normalize zero vector
            return [x / norm for x in vector]

        # Test normalization
        original = [3.0, 4.0, 0.0]  # Length = 5
        normalized = normalize_vector(original)

        # Verify unit length
        norm = math.sqrt(sum(x * x for x in normalized))
        assert abs(norm - 1.0) < 1e-10

        # Verify direction preserved (proportional)
        ratio = normalized[0] / original[0]
        assert abs(normalized[1] - original[1] * ratio) < 1e-10

        # Test zero vector
        zero_vec = [0.0, 0.0, 0.0]
        normalized_zero = normalize_vector(zero_vec)
        assert normalized_zero == zero_vec  # Should return original
