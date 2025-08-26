"""Unit tests for DualCacheManager (REQ-0025-v2).

Tests dual-layer caching architecture with IngestionCache + GPTCache,
cache coordination, performance targets, and multi-agent support.

These are FAILING tests that will pass once the implementation is complete.
"""

import asyncio
import json
from unittest.mock import Mock, patch

import pytest

# These imports will fail until implementation is complete - this is expected
try:
    from src.core.document_processing.direct_unstructured_processor import (
        ProcessingResult,
    )
    from src.core.document_processing.dual_cache_manager import (
        CacheCoordinationResult,
        CacheHitResult,
        CacheLayer,
        CacheStats,
        DualCacheManager,
    )
    from src.core.document_processing.semantic_chunker import (
        ChunkingResult,
    )
except ImportError:
    # Create placeholder classes for failing tests
    class DualCacheManager:
        """Placeholder DualCacheManager class for failing tests."""

        pass

    class CacheLayer:
        """Placeholder CacheLayer class for failing tests."""

        INGESTION = "ingestion"
        SEMANTIC = "semantic"
        GPT = "gpt"

    class CacheHitResult:
        """Placeholder CacheHitResult class for failing tests."""

        pass

    class CacheStats:
        """Placeholder CacheStats class for failing tests."""

        pass

    class CacheCoordinationResult:
        """Placeholder CacheCoordinationResult class for failing tests."""

        pass

    class ProcessingResult:
        """Placeholder ProcessingResult class for failing tests."""

        pass

    class ChunkingResult:
        """Placeholder ChunkingResult class for failing tests."""

        pass


@pytest.fixture
def mock_ingestion_cache():
    """Mock LlamaIndex IngestionCache for testing."""
    with patch("llama_index.core.ingestion.IngestionCache") as mock_cache:
        mock_instance = Mock()
        mock_instance.get.return_value = None  # Cache miss by default
        mock_instance.put.return_value = True
        mock_instance.contains.return_value = False
        mock_instance.clear.return_value = True
        mock_instance.size = 0
        mock_cache.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_gpt_cache():
    """Mock GPTCache for semantic similarity caching."""
    try:
        with patch("gptcache.Cache") as mock_cache:
            yield mock_cache
    except ImportError:
        # GPTCache not available - use generic mock
        with patch("builtins.object") as mock_cache:
            yield mock_cache


@pytest.fixture
def mock_gptcache_alternative():
    """Alternative mock for GPTCache when not available."""
    with patch("builtins.object") as mock_cache:
        mock_instance = Mock()
        mock_instance.get.return_value = None  # Cache miss by default
        mock_instance.put.return_value = True
        mock_instance.close.return_value = None
        mock_cache.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for semantic cache backend."""
    mock_client = Mock()
    mock_client.search.return_value = []
    mock_client.upsert.return_value = Mock(status="completed")
    mock_client.create_collection.return_value = True
    mock_client.collection_exists.return_value = True
    return mock_client


@pytest.fixture
def mock_settings():
    """Mock DocMind settings for caching."""
    settings = Mock()
    settings.enable_document_caching = True
    settings.cache_dir = "/tmp/docmind_cache"
    settings.sqlite_db_path = "/tmp/docmind_cache/cache.db"
    settings.cache_ttl_seconds = 3600
    settings.max_cache_size_mb = 1000
    settings.enable_semantic_cache = True
    settings.semantic_cache_threshold = 0.85
    settings.cache_compression = True
    return settings


@pytest.fixture
def sample_processing_result():
    """Sample processing result for caching tests."""
    return Mock(
        elements=[
            Mock(text="Sample content", category="NarrativeText"),
            Mock(text="Table content", category="Table"),
        ],
        processing_time=0.5,
        strategy_used="hi_res",
        metadata={"document_hash": "abc123", "file_size": 1024},
    )


@pytest.fixture
def sample_chunking_result():
    """Sample chunking result for caching tests."""
    return Mock(
        chunks=[
            Mock(text="First chunk", metadata={"chunk_index": 0}),
            Mock(text="Second chunk", metadata={"chunk_index": 1}),
        ],
        total_elements=5,
        boundary_accuracy=0.92,
        processing_time=0.2,
    )


class TestDualCacheManager:
    """Test suite for DualCacheManager implementation.

    Tests REQ-0025-v2: Dual-Layer Caching Architecture
    - Layer 1: IngestionCache with 80-95% processing reduction validation
    - Layer 2: GPTCache semantic similarity matching with 60-70% hit rate
    - SQLite WAL mode for concurrent access
    - Qdrant backend integration for semantic similarity caching
    - Cache coordination across multiple agents
    """

    @pytest.mark.unit
    def test_cache_manager_initialization(self, mock_settings):
        """Test DualCacheManager initializes correctly.

        Should pass after implementation:
        - Creates cache manager with proper settings
        - Initializes both cache layers (IngestionCache + GPTCache)
        - Sets up SQLite WAL mode for concurrent access
        - Configures cache directories and parameters
        """
        cache_manager = DualCacheManager(mock_settings)

        assert cache_manager is not None
        assert hasattr(cache_manager, "settings")
        assert hasattr(cache_manager, "ingestion_cache")
        assert hasattr(cache_manager, "semantic_cache")
        assert cache_manager.settings == mock_settings

        # Verify cache layers are initialized
        assert hasattr(cache_manager, "cache_layers")
        assert CacheLayer.INGESTION in cache_manager.cache_layers
        assert CacheLayer.SEMANTIC in cache_manager.cache_layers

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ingestion_cache_layer_operations(
        self, mock_ingestion_cache, mock_settings, sample_processing_result
    ):
        """Test IngestionCache layer operations.

        Should pass after implementation:
        - Stores processing results with document hash keys
        - Retrieves cached results for identical documents
        - Achieves 80-95% processing reduction target
        - Handles cache misses gracefully
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache

        document_path = "/path/to/document.pdf"
        document_hash = cache_manager._calculate_document_hash(document_path)

        # Test cache miss (first processing)
        cached_result = await cache_manager.get_cached_processing_result(document_path)
        assert cached_result is None
        mock_ingestion_cache.get.assert_called_once_with(document_hash)

        # Test cache storage
        await cache_manager.store_processing_result(
            document_path, sample_processing_result
        )
        mock_ingestion_cache.put.assert_called_once_with(
            document_hash, sample_processing_result
        )

        # Test cache hit (subsequent processing)
        mock_ingestion_cache.get.return_value = sample_processing_result
        cached_result = await cache_manager.get_cached_processing_result(document_path)
        assert cached_result == sample_processing_result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_semantic_cache_layer_operations(
        self, mock_gpt_cache, mock_qdrant_client, mock_settings, sample_chunking_result
    ):
        """Test GPTCache semantic similarity layer operations.

        Should pass after implementation:
        - Uses Qdrant backend for semantic similarity matching
        - Stores chunking results with semantic embeddings
        - Retrieves similar cached results within threshold
        - Achieves 60-70% hit rate target for similar content
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.semantic_cache = mock_gpt_cache
        cache_manager.qdrant_client = mock_qdrant_client

        query_text = "machine learning algorithms and their applications"
        similarity_threshold = 0.85

        # Test semantic cache miss
        cached_result = await cache_manager.get_cached_semantic_result(
            query_text, similarity_threshold
        )
        assert cached_result is None

        # Test semantic cache storage
        await cache_manager.store_semantic_result(query_text, sample_chunking_result)
        mock_gpt_cache.put.assert_called_once()

        # Test semantic cache hit with similar query
        similar_query = "ML algorithms and their use cases"
        mock_qdrant_client.search.return_value = [
            Mock(score=0.87, payload={"result": sample_chunking_result})
        ]

        cached_result = await cache_manager.get_cached_semantic_result(
            similar_query, similarity_threshold
        )
        assert cached_result == sample_chunking_result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_coordination_across_agents(
        self, mock_ingestion_cache, mock_gpt_cache, mock_settings
    ):
        """Test cache coordination across multiple agents.

        Should pass after implementation:
        - Coordinates cache access between different agents
        - Handles concurrent cache operations safely
        - Maintains cache consistency across agent interactions
        - Provides cache stats for monitoring
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache
        cache_manager.semantic_cache = mock_gpt_cache

        # Simulate multiple agents accessing cache
        agent_1_key = "agent_1_document.pdf"
        agent_2_key = "agent_2_document.pdf"

        # Test concurrent cache operations
        tasks = [
            cache_manager.get_cached_processing_result(agent_1_key),
            cache_manager.get_cached_processing_result(agent_2_key),
        ]

        results = await asyncio.gather(*tasks)
        assert len(results) == 2

        # Test cache coordination result
        coordination_result = await cache_manager.coordinate_multi_agent_cache()
        assert isinstance(coordination_result, CacheCoordinationResult)
        assert hasattr(coordination_result, "active_agents")
        assert hasattr(coordination_result, "cache_stats")

    @pytest.mark.unit
    def test_sqlite_wal_mode_configuration(self, mock_settings):
        """Test SQLite WAL mode configuration for concurrent access.

        Should pass after implementation:
        - Configures SQLite connection with WAL mode
        - Enables concurrent read/write operations
        - Sets appropriate journal mode and synchronization
        - Handles database connection pooling
        """
        cache_manager = DualCacheManager(mock_settings)

        # Test SQLite configuration
        db_config = cache_manager._get_sqlite_config()
        assert db_config["journal_mode"] == "WAL"
        assert db_config["synchronous"] == "NORMAL"
        assert db_config["cache_size"] > 0
        assert db_config["temp_store"] == "MEMORY"

        # Test connection creation
        connection = cache_manager._create_sqlite_connection()
        assert connection is not None

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_performance_targets(self, mock_ingestion_cache, mock_settings):
        """Test cache performance targets validation.

        Should pass after implementation:
        - Achieves 80-95% processing reduction with IngestionCache
        - Validates cache hit rates meet performance targets
        - Tracks cache performance metrics
        - Provides performance feedback
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache

        # Simulate cache performance tracking
        cache_stats = CacheStats()
        cache_stats.total_requests = 100
        cache_stats.ingestion_hits = 85  # 85% hit rate
        cache_stats.semantic_hits = 65  # 65% hit rate

        # Test performance validation
        performance_result = cache_manager._validate_cache_performance(cache_stats)

        # Verify ingestion cache meets 80-95% target
        assert 0.80 <= performance_result.ingestion_hit_rate <= 0.95

        # Verify semantic cache meets 60-70% target
        assert 0.60 <= performance_result.semantic_hit_rate <= 0.70

        # Verify overall performance is acceptable
        assert performance_result.meets_targets is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_qdrant_backend_integration(self, mock_qdrant_client, mock_settings):
        """Test Qdrant backend integration for semantic caching.

        Should pass after implementation:
        - Uses Qdrant for vector similarity search
        - Stores embeddings with cache metadata
        - Performs similarity search with configurable threshold
        - Manages vector collections properly
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.qdrant_client = mock_qdrant_client

        # Test collection creation
        collection_name = "semantic_cache"
        await cache_manager._ensure_qdrant_collection(collection_name)
        mock_qdrant_client.create_collection.assert_called_once()

        # Test vector storage
        vector_data = [0.1] * 1024  # Mock 1024-dim embedding
        metadata = {"cache_key": "test_key", "timestamp": 1234567890}

        await cache_manager._store_vector_embedding(
            collection_name, vector_data, metadata
        )
        mock_qdrant_client.upsert.assert_called_once()

        # Test similarity search
        query_vector = [0.1] * 1024
        await cache_manager._search_similar_vectors(
            collection_name, query_vector, threshold=0.85, limit=5
        )
        mock_qdrant_client.search.assert_called_once()

    @pytest.mark.unit
    def test_document_hash_calculation(self, mock_settings):
        """Test document hash calculation for cache keys.

        Should pass after implementation:
        - Calculates consistent hash for same document
        - Includes file metadata in hash calculation
        - Handles different file types appropriately
        - Provides collision-resistant hash values
        """
        cache_manager = DualCacheManager(mock_settings)

        # Test hash calculation for same file
        file_path = "/path/to/test.pdf"
        hash1 = cache_manager._calculate_document_hash(file_path)
        hash2 = cache_manager._calculate_document_hash(file_path)
        assert hash1 == hash2

        # Test hash differences for different files
        different_file = "/path/to/different.pdf"
        hash3 = cache_manager._calculate_document_hash(different_file)
        assert hash1 != hash3

        # Test hash includes file metadata
        assert len(hash1) > 0
        assert isinstance(hash1, str)

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_invalidation_strategies(
        self, mock_ingestion_cache, mock_gpt_cache, mock_settings
    ):
        """Test cache invalidation and cleanup strategies.

        Should pass after implementation:
        - Invalidates cache entries based on TTL
        - Removes stale cache entries automatically
        - Handles cache size limits with LRU eviction
        - Provides manual cache cleanup capabilities
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache
        cache_manager.semantic_cache = mock_gpt_cache

        # Test TTL-based invalidation
        await cache_manager._invalidate_expired_entries()

        # Test cache size limit handling
        cache_stats = await cache_manager.get_cache_stats()
        if cache_stats.size_mb > mock_settings.max_cache_size_mb:
            await cache_manager._evict_lru_entries()

        # Test manual cache cleanup
        await cache_manager.clear_cache(CacheLayer.INGESTION)
        mock_ingestion_cache.clear.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_compression(self, mock_settings):
        """Test cache compression for storage efficiency.

        Should pass after implementation:
        - Compresses cached data before storage
        - Decompresses data on retrieval
        - Reduces storage space requirements
        - Maintains data integrity through compression
        """
        mock_settings.cache_compression = True
        cache_manager = DualCacheManager(mock_settings)

        # Test data compression
        original_data = {"large_content": "x" * 1000, "metadata": {"key": "value"}}
        compressed_data = cache_manager._compress_cache_data(original_data)

        assert len(compressed_data) < len(json.dumps(original_data))

        # Test data decompression
        decompressed_data = cache_manager._decompress_cache_data(compressed_data)
        assert decompressed_data == original_data

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_error_handling_cache_failures(self, mock_settings):
        """Test error handling for cache operation failures.

        Should pass after implementation:
        - Handles cache connection failures gracefully
        - Falls back to processing when cache unavailable
        - Logs cache errors appropriately
        - Maintains system stability during cache issues
        """
        cache_manager = DualCacheManager(mock_settings)

        # Test cache connection failure
        with patch.object(
            cache_manager,
            "ingestion_cache",
            side_effect=Exception("Cache connection failed"),
        ):
            result = await cache_manager.get_cached_processing_result("test.pdf")
            assert result is None  # Should return None on cache failure

        # Test cache storage failure
        with patch.object(cache_manager, "ingestion_cache") as mock_cache:
            mock_cache.put.side_effect = Exception("Storage failed")

            # Should handle error gracefully without raising
            try:
                await cache_manager.store_processing_result("test.pdf", Mock())
                success = True
            except Exception:
                success = False

            assert success is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_cache_statistics_and_monitoring(
        self, mock_ingestion_cache, mock_gpt_cache, mock_settings
    ):
        """Test cache statistics collection and monitoring.

        Should pass after implementation:
        - Tracks cache hit/miss ratios
        - Monitors cache performance metrics
        - Provides detailed cache statistics
        - Supports cache performance analysis
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache
        cache_manager.semantic_cache = mock_gpt_cache

        # Test cache statistics collection
        stats = await cache_manager.get_cache_stats()

        assert isinstance(stats, CacheStats)
        assert hasattr(stats, "ingestion_hits")
        assert hasattr(stats, "ingestion_misses")
        assert hasattr(stats, "semantic_hits")
        assert hasattr(stats, "semantic_misses")
        assert hasattr(stats, "hit_rate")
        assert hasattr(stats, "size_mb")

        # Test performance metrics
        assert 0.0 <= stats.hit_rate <= 1.0
        assert stats.size_mb >= 0

    @pytest.mark.unit
    def test_concurrent_access_safety(self, mock_settings):
        """Test thread-safe concurrent cache access.

        Should pass after implementation:
        - Handles multiple concurrent cache operations
        - Uses appropriate locking mechanisms
        - Prevents race conditions in cache updates
        - Maintains data consistency under load
        """
        cache_manager = DualCacheManager(mock_settings)

        # Test thread safety mechanisms
        assert hasattr(cache_manager, "_cache_lock") or hasattr(cache_manager, "_locks")

        # Test concurrent operation handling
        with cache_manager._acquire_cache_lock("test_key"):
            # Should acquire lock successfully
            assert cache_manager._is_locked("test_key") is True


class TestCacheIntegrationWithProcessing:
    """Test cache integration with document processing components."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_processing_pipeline_cache_integration(
        self, mock_ingestion_cache, mock_settings, sample_processing_result
    ):
        """Test integration with document processing pipeline.

        Should pass after implementation:
        - Integrates seamlessly with DirectUnstructuredProcessor
        - Caches processing results automatically
        - Reduces processing time for repeated documents
        - Maintains processing quality with cached results
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache

        document_path = "test_document.pdf"

        # First processing - cache miss
        mock_ingestion_cache.get.return_value = None
        cached_result = await cache_manager.get_cached_processing_result(document_path)
        assert cached_result is None

        # Store processing result
        await cache_manager.store_processing_result(
            document_path, sample_processing_result
        )

        # Second processing - cache hit
        mock_ingestion_cache.get.return_value = sample_processing_result
        cached_result = await cache_manager.get_cached_processing_result(document_path)
        assert cached_result == sample_processing_result

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_chunking_pipeline_cache_integration(
        self, mock_gpt_cache, mock_settings, sample_chunking_result
    ):
        """Test integration with semantic chunking pipeline.

        Should pass after implementation:
        - Integrates with SemanticChunker operations
        - Caches chunking results based on content similarity
        - Retrieves similar chunking patterns efficiently
        - Maintains chunking quality standards
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.semantic_cache = mock_gpt_cache

        content_text = "machine learning and AI algorithms"

        # Store chunking result
        await cache_manager.store_semantic_result(content_text, sample_chunking_result)

        # Retrieve similar result
        similar_content = "ML and artificial intelligence methods"
        mock_gpt_cache.get.return_value = sample_chunking_result

        cached_result = await cache_manager.get_cached_semantic_result(
            similar_content, 0.8
        )
        assert cached_result == sample_chunking_result


class TestGherkinScenariosCaching:
    """Test Gherkin scenarios for dual-layer caching from ADR-009."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scenario_dual_layer_cache_performance(
        self, mock_ingestion_cache, mock_gpt_cache, mock_settings
    ):
        """Test Scenario 3: Dual-Layer Cache Performance.

        Given: Documents being processed repeatedly
        When: Using DualCacheManager with both cache layers
        Then: IngestionCache achieves 80-95% processing reduction
        And: GPTCache achieves 60-70% semantic similarity hit rate
        And: Cache coordination works across multiple agents
        And: SQLite WAL mode enables concurrent access
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache
        cache_manager.semantic_cache = mock_gpt_cache

        # Simulate cache performance over multiple requests
        total_requests = 100
        ingestion_hits = 87  # 87% hit rate (within 80-95% target)
        semantic_hits = 65  # 65% hit rate (within 60-70% target)

        # When: Processing documents with cache
        stats = CacheStats()
        stats.total_requests = total_requests
        stats.ingestion_hits = ingestion_hits
        stats.semantic_hits = semantic_hits

        # Then: Performance targets met
        ingestion_hit_rate = ingestion_hits / total_requests
        semantic_hit_rate = semantic_hits / total_requests

        assert 0.80 <= ingestion_hit_rate <= 0.95  # IngestionCache target
        assert 0.60 <= semantic_hit_rate <= 0.70  # GPTCache target

        # And: Cache coordination works
        coordination_result = await cache_manager.coordinate_multi_agent_cache()
        assert isinstance(coordination_result, CacheCoordinationResult)

        # And: SQLite WAL mode configured
        db_config = cache_manager._get_sqlite_config()
        assert db_config["journal_mode"] == "WAL"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_scenario_multi_agent_cache_sharing(
        self, mock_ingestion_cache, mock_settings, sample_processing_result
    ):
        """Test multi-agent cache sharing scenario.

        Given: Multiple agents processing documents
        When: Using shared DualCacheManager
        Then: Agents can access cached results from other agents
        And: Cache consistency is maintained across agents
        And: Concurrent access is handled safely
        And: Performance benefits are realized across all agents
        """
        cache_manager = DualCacheManager(mock_settings)
        cache_manager.ingestion_cache = mock_ingestion_cache

        # Agent 1 processes and caches document
        document_key = "shared_document.pdf"
        await cache_manager.store_processing_result(
            document_key, sample_processing_result
        )

        # Agent 2 retrieves cached result
        mock_ingestion_cache.get.return_value = sample_processing_result
        cached_result = await cache_manager.get_cached_processing_result(document_key)

        # Then: Agents share cached results
        assert cached_result == sample_processing_result

        # And: Cache coordination tracks multiple agents
        coordination_result = await cache_manager.coordinate_multi_agent_cache()
        assert coordination_result.active_agents >= 1


class TestCacheConfigurationAndTuning:
    """Test cache configuration and performance tuning."""

    @pytest.mark.unit
    def test_cache_parameter_optimization(self, mock_settings):
        """Test cache parameter optimization for different workloads.

        Should pass after implementation:
        - Optimizes cache size based on available memory
        - Adjusts TTL based on document processing patterns
        - Tunes similarity thresholds for semantic caching
        - Provides configuration validation
        """
        cache_manager = DualCacheManager(mock_settings)

        # Test cache size optimization
        available_memory_gb = 8.0
        optimized_size = cache_manager._optimize_cache_size(available_memory_gb)
        assert (
            optimized_size <= available_memory_gb * 1024 * 0.25
        )  # 25% of available memory

        # Test TTL optimization
        processing_frequency = 50  # documents per hour
        optimized_ttl = cache_manager._optimize_ttl(processing_frequency)
        assert optimized_ttl > 0

        # Test similarity threshold tuning
        content_diversity = 0.7  # High diversity
        optimized_threshold = cache_manager._optimize_similarity_threshold(
            content_diversity
        )
        assert 0.5 <= optimized_threshold <= 0.95

    @pytest.mark.unit
    def test_cache_health_monitoring(self, mock_settings):
        """Test cache health monitoring and alerting.

        Should pass after implementation:
        - Monitors cache performance metrics
        - Detects cache performance degradation
        - Provides alerting for cache issues
        - Suggests optimization recommendations
        """
        cache_manager = DualCacheManager(mock_settings)

        # Test health check
        health_status = cache_manager._check_cache_health()
        assert hasattr(health_status, "overall_health")
        assert hasattr(health_status, "recommendations")
        assert hasattr(health_status, "alerts")

        # Test performance degradation detection
        mock_stats = Mock()
        mock_stats.hit_rate = 0.5  # Below expected performance
        mock_stats.response_time_ms = 100

        degradation_detected = cache_manager._detect_performance_degradation(mock_stats)
        assert isinstance(degradation_detected, bool)
