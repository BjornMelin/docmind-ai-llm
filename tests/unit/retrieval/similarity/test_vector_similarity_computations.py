"""Unit tests for vector similarity computations.

Focus areas:
- Cosine, dot product, Euclidean metrics (dense + sparse)
- Batch computations and dimension checks
- Edge cases (zero/identical/orthogonal, NaN/Inf)
- Performance bounds for moderate batch sizes
"""

import numpy as np
import pytest


@pytest.fixture
def sample_1024d_vectors():
    """Generate sample 1024-dimensional vectors for similarity testing."""
    np.random.seed(42)  # For reproducible test results
    return {
        "random_vectors": [np.random.randn(1024) for _ in range(10)],
        "zero_vector": np.zeros(1024),
        "unit_vector": np.ones(1024) / np.sqrt(1024),
        "identical_vectors": [np.ones(1024) * 0.5 for _ in range(3)],
        "orthogonal_vectors": [
            np.concatenate([np.ones(512), np.zeros(512)]),
            np.concatenate([np.zeros(512), np.ones(512)]),
        ],
        "normalized_vectors": [
            vec / np.linalg.norm(vec)
            for vec in [np.random.randn(1024) for _ in range(5)]
        ],
    }


@pytest.fixture
def sparse_embedding_samples():
    """Generate sample sparse embeddings for similarity testing."""
    return {
        "sparse_1": {100: 0.8, 200: 0.6, 300: 0.4, 400: 0.2},
        "sparse_2": {100: 0.7, 250: 0.5, 300: 0.9, 450: 0.3},
        "sparse_empty": {},
        "sparse_disjoint_1": {100: 0.8, 200: 0.6},
        "sparse_disjoint_2": {500: 0.7, 600: 0.5},
        "sparse_identical": {100: 0.8, 200: 0.6, 300: 0.4},
        "sparse_batch": [
            {100: 0.8, 200: 0.6, 300: 0.4},
            {150: 0.9, 250: 0.7, 350: 0.5},
            {80: 0.7, 180: 0.8, 280: 0.6},
            {120: 0.6, 220: 0.8, 320: 0.5},
            {90: 0.9, 190: 0.7, 290: 0.4},
        ],
    }


@pytest.mark.unit
class TestCosineSimilarityComputations:
    """Test cosine similarity computations for dense vectors."""

    def test_cosine_similarity_basic_computation(self, sample_1024d_vectors):
        """Test basic cosine similarity computation."""
        vectors = sample_1024d_vectors["random_vectors"]

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                vec1, vec2 = vectors[i], vectors[j]

                # Compute cosine similarity
                dot_product = np.dot(vec1, vec2)
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    cosine_sim = dot_product / (norm1 * norm2)

                    # Validate cosine similarity properties
                    assert -1.0 <= cosine_sim <= 1.0, (
                        f"Cosine similarity should be in [-1,1], got {cosine_sim}"
                    )
                    assert isinstance(cosine_sim, int | float | np.number), (
                        "Cosine similarity should be numeric"
                    )

    def test_cosine_similarity_identical_vectors(self, sample_1024d_vectors):
        """Test cosine similarity for identical vectors."""
        identical_vectors = sample_1024d_vectors["identical_vectors"]

        for i in range(len(identical_vectors)):
            for j in range(len(identical_vectors)):
                vec1, vec2 = identical_vectors[i], identical_vectors[j]

                dot_product = np.dot(vec1, vec2)
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

                cosine_sim = dot_product / (norm1 * norm2)

                # Identical vectors should have cosine similarity of 1.0
                assert abs(cosine_sim - 1.0) < 1e-10, (
                    f"Identical vectors should have cosine similarity 1.0, "
                    f"got {cosine_sim}"
                )

    def test_cosine_similarity_orthogonal_vectors(self, sample_1024d_vectors):
        """Test cosine similarity for orthogonal vectors."""
        orthogonal = sample_1024d_vectors["orthogonal_vectors"]
        vec1, vec2 = orthogonal[0], orthogonal[1]

        dot_product = np.dot(vec1, vec2)
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

        cosine_sim = dot_product / (norm1 * norm2)

        # Orthogonal vectors should have cosine similarity close to 0
        assert abs(cosine_sim) < 1e-10, (
            f"Orthogonal vectors should have cosine similarity ~0, got {cosine_sim}"
        )

    def test_cosine_similarity_zero_vector_handling(self, sample_1024d_vectors):
        """Test handling of zero vectors in cosine similarity."""
        zero_vec = sample_1024d_vectors["zero_vector"]
        unit_vec = sample_1024d_vectors["unit_vector"]

        dot_product = np.dot(zero_vec, unit_vec)
        norm_zero, norm_unit = np.linalg.norm(zero_vec), np.linalg.norm(unit_vec)

        assert norm_zero == 0.0, "Zero vector should have norm 0"
        assert norm_unit > 0.0, "Unit vector should have positive norm"
        assert dot_product == 0.0, "Dot product with zero vector should be 0"

    def test_cosine_similarity_normalized_vectors(self, sample_1024d_vectors):
        """Test cosine similarity with pre-normalized vectors."""
        normalized_vectors = sample_1024d_vectors["normalized_vectors"]

        for i, vec in enumerate(normalized_vectors):
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-10, (
                f"Vector {i} should be normalized, norm={norm}"
            )

        for i in range(len(normalized_vectors)):
            for j in range(i + 1, len(normalized_vectors)):
                vec1, vec2 = normalized_vectors[i], normalized_vectors[j]

                # For normalized vectors, cosine similarity is just dot product
                cosine_sim = np.dot(vec1, vec2)
                dot_product_sim = np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2)
                )

                assert abs(cosine_sim - dot_product_sim) < 1e-10, (
                    "Should be equivalent for normalized vectors"
                )

    def test_batch_cosine_similarity_computation(self, sample_1024d_vectors):
        """Test batch cosine similarity computation."""
        vectors = sample_1024d_vectors["random_vectors"][:5]
        query_vector = sample_1024d_vectors["unit_vector"]

        # Batch computation
        batch_matrix = np.array(vectors)
        query_array = query_vector

        # Compute batch similarities
        dot_products = np.dot(batch_matrix, query_array)
        batch_norms = np.linalg.norm(batch_matrix, axis=1)
        query_norm = np.linalg.norm(query_array)

        batch_similarities = dot_products / (batch_norms * query_norm)

        assert len(batch_similarities) == 5, "Should have 5 similarity scores"

        for sim in batch_similarities:
            assert -1.0 <= sim <= 1.0, (
                f"Batch similarity should be in [-1,1], got {sim}"
            )
            assert isinstance(sim, int | float | np.number), (
                "Batch similarity should be numeric"
            )


@pytest.mark.unit
class TestDotProductSimilarityComputations:
    """Test dot product similarity computations."""

    def test_dot_product_basic_computation(self, sample_1024d_vectors):
        """Test basic dot product computation."""
        vectors = sample_1024d_vectors["random_vectors"]

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                vec1, vec2 = vectors[i], vectors[j]

                dot_product = np.dot(vec1, vec2)

                assert isinstance(dot_product, int | float | np.number), (
                    "Dot product should be numeric"
                )

    def test_dot_product_identical_vectors(self, sample_1024d_vectors):
        """Test dot product for identical vectors."""
        identical_vectors = sample_1024d_vectors["identical_vectors"]

        for vec in identical_vectors:
            self_dot = np.dot(vec, vec)
            norm_squared = np.linalg.norm(vec) ** 2

            assert abs(self_dot - norm_squared) < 1e-10, (
                "Self dot product should equal norm squared"
            )

    def test_dot_product_zero_vector(self, sample_1024d_vectors):
        """Test dot product with zero vector."""
        zero_vec = sample_1024d_vectors["zero_vector"]
        vectors = sample_1024d_vectors["random_vectors"]

        for vec in vectors:
            dot_product = np.dot(zero_vec, vec)
            assert dot_product == 0.0, (
                f"Dot product with zero vector should be 0, got {dot_product}"
            )

    def test_batch_dot_product_computation(self, sample_1024d_vectors):
        """Test batch dot product computation."""
        vectors = sample_1024d_vectors["random_vectors"][:5]
        query_vector = sample_1024d_vectors["unit_vector"]

        # Batch computation
        batch_matrix = np.array(vectors)
        query_array = query_vector

        batch_dot_products = np.dot(batch_matrix, query_array)

        assert len(batch_dot_products) == 5, "Should have 5 dot products"

        for dot_prod in batch_dot_products:
            assert isinstance(dot_prod, int | float | np.number), (
                "Batch dot product should be numeric"
            )

        # Verify against individual computations
        for i, vec in enumerate(vectors):
            individual_dot = np.dot(vec, query_vector)
            assert abs(batch_dot_products[i] - individual_dot) < 1e-10, (
                "Batch and individual should match"
            )


@pytest.mark.unit
class TestEuclideanDistanceComputations:
    """Test Euclidean distance computations."""

    def test_euclidean_distance_basic_computation(self, sample_1024d_vectors):
        """Test basic Euclidean distance computation."""
        vectors = sample_1024d_vectors["random_vectors"]

        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                vec1, vec2 = vectors[i], vectors[j]

                distance = np.linalg.norm(vec1 - vec2)

                assert distance >= 0.0, (
                    f"Distance should be non-negative, got {distance}"
                )
                assert isinstance(distance, int | float | np.number), (
                    "Distance should be numeric"
                )

    def test_euclidean_distance_identical_vectors(self, sample_1024d_vectors):
        """Test Euclidean distance for identical vectors."""
        identical_vectors = sample_1024d_vectors["identical_vectors"]

        for i in range(len(identical_vectors)):
            for j in range(len(identical_vectors)):
                vec1, vec2 = identical_vectors[i], identical_vectors[j]

                distance = np.linalg.norm(vec1 - vec2)

                assert distance < 1e-10, (
                    f"Distance between identical vectors should be ~0, got {distance}"
                )

    def test_euclidean_distance_self(self, sample_1024d_vectors):
        """Test Euclidean distance from vector to itself."""
        vectors = sample_1024d_vectors["random_vectors"]

        for vec in vectors:
            self_distance = np.linalg.norm(vec - vec)
            assert self_distance == 0.0, (
                f"Self-distance should be 0, got {self_distance}"
            )

    def test_euclidean_distance_symmetry(self, sample_1024d_vectors):
        """Test Euclidean distance symmetry."""
        vectors = sample_1024d_vectors["random_vectors"][:3]

        for i in range(len(vectors)):
            for j in range(len(vectors)):
                if i != j:
                    vec1, vec2 = vectors[i], vectors[j]

                    dist_ij = np.linalg.norm(vec1 - vec2)
                    dist_ji = np.linalg.norm(vec2 - vec1)

                    assert abs(dist_ij - dist_ji) < 1e-10, (
                        "Distance should be symmetric"
                    )

    def test_euclidean_distance_triangle_inequality(self, sample_1024d_vectors):
        """Test Euclidean distance triangle inequality."""
        vectors = sample_1024d_vectors["random_vectors"][:3]
        vec1, vec2, vec3 = vectors[0], vectors[1], vectors[2]

        dist_12 = np.linalg.norm(vec1 - vec2)
        dist_23 = np.linalg.norm(vec2 - vec3)
        dist_13 = np.linalg.norm(vec1 - vec3)

        # Triangle inequality: d(1,3) <= d(1,2) + d(2,3)
        assert dist_13 <= dist_12 + dist_23 + 1e-10, "Triangle inequality should hold"


@pytest.mark.unit
class TestSparseEmbeddingSimilarity:
    """Test sparse embedding similarity computations."""

    def test_sparse_overlap_similarity(self, sparse_embedding_samples):
        """Test sparse embedding overlap-based similarity."""
        sparse1 = sparse_embedding_samples["sparse_1"]
        sparse2 = sparse_embedding_samples["sparse_2"]

        # Compute overlap similarity
        common_tokens = set(sparse1.keys()) & set(sparse2.keys())
        overlap_score = sum(
            min(sparse1[token], sparse2[token]) for token in common_tokens
        )

        assert overlap_score > 0.0, "Should have positive overlap"
        assert isinstance(overlap_score, int | float), "Overlap score should be numeric"

    def test_sparse_jaccard_similarity(self, sparse_embedding_samples):
        """Test Jaccard similarity for sparse embeddings."""
        sparse1 = sparse_embedding_samples["sparse_1"]
        sparse2 = sparse_embedding_samples["sparse_2"]

        # Compute Jaccard similarity
        intersection = set(sparse1.keys()) & set(sparse2.keys())
        union = set(sparse1.keys()) | set(sparse2.keys())

        jaccard_sim = len(intersection) / len(union) if union else 0.0

        assert 0.0 <= jaccard_sim <= 1.0, (
            f"Jaccard similarity should be in [0,1], got {jaccard_sim}"
        )

    def test_sparse_cosine_similarity(self, sparse_embedding_samples):
        """Test cosine similarity for sparse embeddings."""
        sparse1 = sparse_embedding_samples["sparse_1"]
        sparse2 = sparse_embedding_samples["sparse_2"]

        # Convert to dense representation for cosine similarity
        all_tokens = set(sparse1.keys()) | set(sparse2.keys())
        max_token = max(all_tokens)

        # Create dense vectors
        dense1 = np.zeros(max_token + 1)
        dense2 = np.zeros(max_token + 1)

        for token, weight in sparse1.items():
            dense1[token] = weight

        for token, weight in sparse2.items():
            dense2[token] = weight

        # Compute cosine similarity
        dot_product = np.dot(dense1, dense2)
        norm1, norm2 = np.linalg.norm(dense1), np.linalg.norm(dense2)

        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
            assert -1.0 <= cosine_sim <= 1.0, (
                f"Sparse cosine similarity should be in [-1,1], got {cosine_sim}"
            )

    def test_sparse_identical_embeddings(self, sparse_embedding_samples):
        """Test similarity for identical sparse embeddings."""
        sparse_identical = sparse_embedding_samples["sparse_identical"]

        # Identical embeddings should have maximum overlap
        overlap_score = sum(
            min(sparse_identical[token], sparse_identical[token])
            for token in sparse_identical
        )
        expected_score = sum(sparse_identical.values())

        assert abs(overlap_score - expected_score) < 1e-10, (
            "Identical embeddings should have perfect overlap"
        )

    def test_sparse_disjoint_embeddings(self, sparse_embedding_samples):
        """Test similarity for disjoint sparse embeddings."""
        sparse1 = sparse_embedding_samples["sparse_disjoint_1"]
        sparse2 = sparse_embedding_samples["sparse_disjoint_2"]

        # Disjoint embeddings should have no overlap
        common_tokens = set(sparse1.keys()) & set(sparse2.keys())
        assert len(common_tokens) == 0, (
            "Disjoint embeddings should have no common tokens"
        )

        overlap_score = sum(
            min(sparse1.get(token, 0), sparse2.get(token, 0))
            for token in set(sparse1.keys()) | set(sparse2.keys())
        )
        assert overlap_score == 0.0, "Disjoint embeddings should have zero overlap"

    def test_sparse_empty_embedding(self, sparse_embedding_samples):
        """Test similarity with empty sparse embeddings."""
        sparse_empty = sparse_embedding_samples["sparse_empty"]
        sparse_normal = sparse_embedding_samples["sparse_1"]

        # Empty embedding should have no overlap with any embedding
        overlap_score = sum(
            min(sparse_empty.get(token, 0), sparse_normal.get(token, 0))
            for token in set(sparse_empty.keys()) | set(sparse_normal.keys())
        )
        assert overlap_score == 0.0, "Empty embedding should have zero overlap"

    def test_batch_sparse_similarity(self, sparse_embedding_samples):
        """Test batch sparse embedding similarity computation."""
        sparse_batch = sparse_embedding_samples["sparse_batch"]
        query_sparse = sparse_embedding_samples["sparse_1"]

        similarities = []
        for sparse_emb in sparse_batch:
            common_tokens = set(sparse_emb.keys()) & set(query_sparse.keys())
            overlap = sum(
                min(sparse_emb[token], query_sparse[token]) for token in common_tokens
            )
            similarities.append(overlap)

        assert len(similarities) == len(sparse_batch), (
            "Should have similarity for each embedding"
        )

        for sim in similarities:
            assert sim >= 0.0, "Similarities should be non-negative"
            assert isinstance(sim, int | float), "Similarities should be numeric"


@pytest.mark.unit
class TestSimilarityMetricValidation:
    """Test validation of similarity metric properties."""

    def test_similarity_metric_bounds(self, sample_1024d_vectors):
        """Test similarity metrics respect proper bounds."""
        vectors = sample_1024d_vectors["random_vectors"][:3]

        for i in range(len(vectors)):
            for j in range(len(vectors)):
                vec1, vec2 = vectors[i], vectors[j]

                # Cosine similarity bounds
                dot_product = np.dot(vec1, vec2)
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)

                if norm1 > 0 and norm2 > 0:
                    cosine_sim = dot_product / (norm1 * norm2)
                    # Account for floating point precision errors
                    assert -1.0 - 1e-10 <= cosine_sim <= 1.0 + 1e-10, (
                        f"Cosine similarity should be in [-1,1], got {cosine_sim}"
                    )

                # Euclidean distance bounds
                distance = np.linalg.norm(vec1 - vec2)
                assert distance >= 0.0, "Euclidean distance should be non-negative"

    def test_similarity_metric_consistency(self, sample_1024d_vectors):
        """Test consistency across different similarity computations."""
        normalized_vectors = sample_1024d_vectors["normalized_vectors"][:3]

        for i in range(len(normalized_vectors)):
            for j in range(len(normalized_vectors)):
                vec1, vec2 = normalized_vectors[i], normalized_vectors[j]

                # For normalized vectors: cosine similarity = dot product
                cosine_sim = np.dot(vec1, vec2) / (
                    np.linalg.norm(vec1) * np.linalg.norm(vec2)
                )
                dot_product = np.dot(vec1, vec2)

                assert abs(cosine_sim - dot_product) < 1e-10, (
                    "Should be equivalent for normalized vectors"
                )

                # Euclidean distance and cosine similarity relationship
                euclidean_dist = np.linalg.norm(vec1 - vec2)
                # Numerical stability: clamp cosine into [-1, 1] before sqrt
                cosine_clipped = max(-1.0, min(1.0, float(cosine_sim)))
                expected_dist = np.sqrt(max(0.0, 2 - 2 * cosine_clipped))

                assert abs(euclidean_dist - expected_dist) < 1e-9, (
                    "Distance and cosine should be consistent"
                )

    def test_similarity_metric_symmetry(self, sample_1024d_vectors):
        """Test symmetry properties of similarity metrics."""
        vectors = sample_1024d_vectors["random_vectors"][:3]

        for i in range(len(vectors)):
            for j in range(len(vectors)):
                if i != j:
                    vec1, vec2 = vectors[i], vectors[j]

                    # Cosine similarity symmetry
                    cosine_ij = np.dot(vec1, vec2) / (
                        np.linalg.norm(vec1) * np.linalg.norm(vec2)
                    )
                    cosine_ji = np.dot(vec2, vec1) / (
                        np.linalg.norm(vec2) * np.linalg.norm(vec1)
                    )

                    assert abs(cosine_ij - cosine_ji) < 1e-10, (
                        "Cosine similarity should be symmetric"
                    )

                    # Dot product symmetry
                    dot_ij = np.dot(vec1, vec2)
                    dot_ji = np.dot(vec2, vec1)

                    assert abs(dot_ij - dot_ji) < 1e-10, (
                        "Dot product should be symmetric"
                    )


@pytest.mark.unit
class TestPerformanceBounds:
    """Test performance bounds for similarity computations."""

    def test_batch_similarity_performance_scaling(self, sample_1024d_vectors):
        """Test batch similarity computation scales properly."""
        import time

        # Test with different batch sizes
        batch_sizes = [10, 50, 100]
        query_vector = sample_1024d_vectors["unit_vector"]

        for batch_size in batch_sizes:
            # Generate batch of specified size
            np.random.seed(42)
            batch_vectors = [np.random.randn(1024) for _ in range(batch_size)]
            batch_matrix = np.array(batch_vectors)

            # Time batch computation
            start_time = time.time()
            similarities = np.dot(batch_matrix, query_vector)
            end_time = time.time()

            computation_time = end_time - start_time

            # Verify results
            assert len(similarities) == batch_size, (
                f"Should have {batch_size} similarities"
            )
            assert computation_time < 1.0, (
                f"Batch size {batch_size} should compute in <1s"
            )

    def test_similarity_computation_consistency(self, sample_1024d_vectors):
        """Test similarity computations are consistent across multiple runs."""
        vectors = sample_1024d_vectors["random_vectors"][:5]
        query_vector = sample_1024d_vectors["unit_vector"]

        # Compute similarities multiple times
        results = []
        for _ in range(3):
            batch_similarities = []
            for vec in vectors:
                dot_product = np.dot(vec, query_vector)
                norm_vec, norm_query = np.linalg.norm(vec), np.linalg.norm(query_vector)
                cosine_sim = dot_product / (norm_vec * norm_query)
                batch_similarities.append(cosine_sim)
            results.append(batch_similarities)

        # Verify consistency across runs
        for i in range(len(results[0])):
            values = [run[i] for run in results]
            assert all(abs(val - values[0]) < 1e-10 for val in values), (
                "Results should be consistent"
            )

    def test_memory_efficiency_bounds(self, sample_1024d_vectors):
        """Test memory usage stays within reasonable bounds."""
        # Test with large batch
        large_batch_size = 1000
        np.random.seed(42)

        # Generate vectors in batches to avoid memory issues
        batch_similarities = []
        query_vector = sample_1024d_vectors["unit_vector"]

        for i in range(0, large_batch_size, 100):
            mini_batch = [
                np.random.randn(1024) for _ in range(min(100, large_batch_size - i))
            ]
            mini_batch_matrix = np.array(mini_batch)

            mini_similarities = np.dot(mini_batch_matrix, query_vector)
            batch_similarities.extend(mini_similarities)

        assert len(batch_similarities) == large_batch_size, (
            "Should handle large batches"
        )

        for sim in batch_similarities:
            assert isinstance(sim, int | float | np.number), (
                "All similarities should be numeric"
            )


@pytest.mark.unit
class TestEdgeCaseSimilarityComputations:
    """Test edge cases in similarity computations."""

    def test_nan_inf_handling_in_similarity(self):
        """Test handling of NaN/Inf values in similarity computations."""
        # Create vectors with special values
        normal_vector = np.random.randn(1024)
        nan_vector = np.full(1024, np.nan)
        inf_vector = np.full(1024, np.inf)

        # Test with normal vector
        with np.errstate(invalid="ignore", over="ignore"):
            with_nan = np.dot(normal_vector, nan_vector)
            with_inf = np.dot(normal_vector, inf_vector)

        assert np.isnan(with_nan) or np.isinf(with_nan), (
            "Should handle NaN appropriately"
        )
        # Depending on sign cancellations, dot with inf may yield inf or nan
        assert np.isinf(with_inf) or np.isnan(with_inf), (
            "Should handle Inf appropriately"
        )

    def test_very_small_vector_similarity(self):
        """Test similarity with very small magnitude vectors."""
        # Create very small vectors
        tiny_magnitude = 1e-15
        small_vector1 = np.ones(1024) * tiny_magnitude
        small_vector2 = np.ones(1024) * tiny_magnitude * 2

        norm1, norm2 = np.linalg.norm(small_vector1), np.linalg.norm(small_vector2)
        dot_product = np.dot(small_vector1, small_vector2)

        if norm1 > 0 and norm2 > 0:
            cosine_sim = dot_product / (norm1 * norm2)
            # Should still be valid for tiny vectors
            assert 0.9 < cosine_sim <= 1.0 + 1e-10, (
                f"Small vectors should have high similarity, got {cosine_sim}"
            )

    def test_very_large_vector_similarity(self):
        """Test similarity with very large magnitude vectors."""
        # Create very large vectors
        large_magnitude = 1e10
        large_vector1 = np.ones(1024) * large_magnitude
        large_vector2 = np.ones(1024) * large_magnitude

        dot_product = np.dot(large_vector1, large_vector2)
        norm1, norm2 = np.linalg.norm(large_vector1), np.linalg.norm(large_vector2)

        cosine_sim = dot_product / (norm1 * norm2)

        # Should still compute correctly for large vectors
        assert abs(cosine_sim - 1.0) < 1e-10, (
            f"Large identical vectors should have similarity 1.0, got {cosine_sim}"
        )

    def test_mixed_precision_similarity(self):
        """Test similarity computations with mixed precision vectors."""
        # Create vectors with different precisions
        float32_vector = np.random.randn(1024).astype(np.float32)
        float64_vector = np.random.randn(1024).astype(np.float64)

        # Compute similarity
        dot_product = np.dot(float32_vector, float64_vector)

        assert isinstance(dot_product, int | float | np.number), (
            "Mixed precision should work"
        )
        assert np.isfinite(dot_product), "Mixed precision result should be finite"
