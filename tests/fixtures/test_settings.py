"""Clean test-specific settings using BaseSettings subclass pattern.

This module provides test-optimized settings following the pytest + pydantic-settings
pattern aligned with the recovered configuration architecture. Properly inherits from
the unified DocMindSettings with nested configuration models.

Architecture:
- MockDocMindSettings: Fast unit tests with CPU-only, optimized defaults
- IntegrationTestSettings: Moderate performance for integration tests
- SystemTestSettings: Production settings for full system tests

Key Features:
- Test-production separation via optimized nested config defaults
- ADR compliance testing support
- Clean temporary directory management
- No backward compatibility code bloat
"""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from src.config.settings import (
    SETTINGS_MODEL_CONFIG,
    AgentConfig,
    CacheConfig,
    DocMindSettings,
    EmbeddingConfig,
    ProcessingConfig,
    RetrievalConfig,
    VLLMConfig,
)


class MockVLLMConfig(VLLMConfig):
    """Test-optimized vLLM configuration for fast, minimal resource usage."""

    # Small context for speed
    context_window: int = Field(default=8192, ge=8192, le=200000)
    max_tokens: int = Field(default=256, ge=100, le=8192)

    # Conservative GPU settings
    gpu_memory_utilization: float = Field(default=0.5, ge=0.5, le=0.95)
    max_num_seqs: int = Field(default=4, ge=1, le=64)
    max_num_batched_tokens: int = Field(default=2048, ge=1024, le=16384)


class MockProcessingConfig(ProcessingConfig):
    """Test-optimized processing configuration for small, fast documents."""

    # Small chunks for speed
    chunk_size: int = Field(default=100, ge=100, le=10000)
    new_after_n_chars: int = Field(default=100, ge=100, le=8000)
    combine_text_under_n_chars: int = Field(default=50, ge=50, le=2000)
    max_document_size_mb: int = Field(default=1, ge=1, le=500)


class MockAgentConfig(AgentConfig):
    """Test-optimized agent configuration for fast decisions."""

    # Fast timeouts for testing
    decision_timeout: int = Field(default=100, ge=10, le=1000)  # 100ms for tests
    max_retries: int = Field(default=1, ge=0, le=10)  # Fewer retries
    max_concurrent_agents: int = Field(default=2, ge=1, le=10)  # Fewer agents

    # Small context management for tests (use minimum allowed values for speed)
    context_trim_threshold: int = Field(
        default=65536, ge=65536, le=131072
    )  # Use minimum allowed
    context_buffer_size: int = Field(default=2048, ge=2048, le=16384)
    chat_memory_limit_tokens: int = Field(
        default=32768, ge=32768, le=98304
    )  # Use minimum allowed
    use_tool_registry: bool = Field(default=True)
    use_shared_llm_client: bool = Field(default=True)
    enable_deadline_propagation: bool = Field(default=False)
    enable_router_injection: bool = Field(default=False)


class MockEmbeddingConfig(EmbeddingConfig):
    """Test-optimized embedding configuration for minimal resource usage."""

    # Small dimensions and batches for speed
    dimension: int = Field(default=384, ge=256, le=4096)  # Smaller than production
    max_length: int = Field(default=512, ge=512, le=16384)  # Smaller context
    batch_size_gpu: int = Field(default=2, ge=1, le=128)  # Small batches
    batch_size_cpu: int = Field(default=1, ge=1, le=32)  # Single item batches


class MockRetrievalConfig(RetrievalConfig):
    """Test-optimized retrieval configuration for fast searches."""

    top_k: int = Field(default=5, ge=1, le=50)  # Fewer results
    reranking_top_k: int = Field(default=3, ge=1, le=20)  # Fewer reranked results


class MockCacheConfig(CacheConfig):
    """Test-optimized cache configuration - caching disabled for test isolation."""

    enable_document_caching: bool = Field(default=False)  # Disabled for test isolation
    ttl_seconds: int = Field(
        default=300, ge=300, le=86400
    )  # Short TTL for tests (minimum allowed)
    max_size_mb: int = Field(
        default=100, ge=100, le=10000
    )  # Small cache size (minimum allowed)
    enable_semantic_cache: bool = Field(default=False)  # Disabled for simplicity


class MockDocMindSettings(DocMindSettings):
    """Test-specific configuration with overrides for fast, deterministic testing.

    Inherits from production settings but provides test-optimized defaults through
    nested configuration overrides. Designed for unit tests with:
    - CPU-only operation (no GPU dependencies)
    - Minimal resource usage (small contexts, batches)
    - Fast timeouts (100ms agent decisions)
    - Disabled caching (test isolation)
    - Small document sizes (fast processing)
    """

    model_config = SettingsConfigDict(
        env_file=None,  # Don't load .env in tests
        env_prefix="DOCMIND_TEST_",  # Use different prefix for isolation
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_default=True,
        extra="forbid",
    )

    # Basic app settings optimized for testing
    debug: bool = Field(default=True)
    log_level: str = Field(default="DEBUG")

    # Temporary directories for test isolation
    data_dir: Path = Field(default=Path("./test_data"))
    cache_dir: Path = Field(default=Path("./test_cache"))
    sqlite_db_path: Path = Field(default=Path("./test_data/test.db"))
    log_file: Path = Field(default=Path("./test_logs/test.log"))

    # Disable expensive operations for unit tests
    enable_gpu_acceleration: bool = Field(default=False)  # CPU-only for tests
    enable_dspy_optimization: bool = Field(default=False)  # Disabled for speed
    enable_performance_logging: bool = Field(default=False)  # Reduced logging
    enable_graphrag: bool = Field(default=False)  # Disabled for simplicity

    # Override nested configurations with test-optimized versions
    vllm: MockVLLMConfig = Field(default_factory=MockVLLMConfig)
    processing: MockProcessingConfig = Field(default_factory=MockProcessingConfig)
    agents: MockAgentConfig = Field(default_factory=MockAgentConfig)
    embedding: MockEmbeddingConfig = Field(default_factory=MockEmbeddingConfig)
    retrieval: MockRetrievalConfig = Field(default_factory=MockRetrievalConfig)
    cache: MockCacheConfig = Field(default_factory=MockCacheConfig)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization with test-specific optimizations."""
        # Create test directories (separate from production)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if self.sqlite_db_path.parent != self.data_dir:
            self.sqlite_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Call parent's post_init
        super().model_post_init(__context)


class IntegrationVLLMConfig(VLLMConfig):
    """Integration test vLLM config with moderate resource usage."""

    context_window: int = Field(default=32768, ge=8192, le=200000)  # Moderate context
    max_tokens: int = Field(default=1024, ge=100, le=8192)
    gpu_memory_utilization: float = Field(default=0.75, ge=0.5, le=0.95)
    max_num_seqs: int = Field(default=8, ge=1, le=64)


class IntegrationAgentConfig(AgentConfig):
    """Integration test agent config with realistic timeouts."""

    decision_timeout: int = Field(
        default=200, ge=10, le=1000
    )  # ADR-011 compliant 200ms
    max_retries: int = Field(default=2, ge=0, le=10)  # Standard retries
    max_concurrent_agents: int = Field(default=3, ge=1, le=10)  # Standard concurrency


class IntegrationProcessingConfig(ProcessingConfig):
    """Integration processing config with larger, safer chunk sizes.

    Prevents metadata-length > chunk-size errors in pipeline during tests.
    """

    chunk_size: int = Field(default=2048, ge=256, le=10000)
    new_after_n_chars: int = Field(default=1200, ge=200, le=8000)
    combine_text_under_n_chars: int = Field(default=200, ge=50, le=2000)


class IntegrationCacheConfig(CacheConfig):
    """Integration test cache config with caching enabled."""

    enable_document_caching: bool = Field(default=True)  # Enabled for integration tests
    ttl_seconds: int = Field(default=1800, ge=300, le=86400)  # 30 min TTL


class IntegrationTestSettings(MockDocMindSettings):
    """Integration test settings with moderate performance requirements.

    Balances test speed with realistic configuration for component integration.
    Enables more features than unit tests but lighter than full system tests.
    """

    model_config = SettingsConfigDict(
        env_file=None,
        env_prefix="DOCMIND_INTEGRATION_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_default=True,
        extra="forbid",
    )

    # Enable realistic features for integration testing
    enable_gpu_acceleration: bool = Field(default=True)
    enable_performance_logging: bool = Field(default=True)
    enable_graphrag: bool = Field(default=False)  # Keep disabled for speed

    # Override with integration-specific nested configs
    vllm: IntegrationVLLMConfig = Field(default_factory=IntegrationVLLMConfig)
    agents: IntegrationAgentConfig = Field(default_factory=IntegrationAgentConfig)
    processing: IntegrationProcessingConfig = Field(
        default_factory=IntegrationProcessingConfig
    )
    cache: IntegrationCacheConfig = Field(default_factory=IntegrationCacheConfig)

    # Convenience alias for tests expecting a flat attribute
    @property
    def embedding_dimension(self) -> int:  # pragma: no cover - simple proxy
        """Expose embedding dimension at top-level for fixture compatibility."""
        return int(self.embedding.dimension)


class SystemTestSettings(DocMindSettings):
    """System test settings - uses production defaults.

    Full production configuration for comprehensive end-to-end testing.
    No overrides - validates the actual production configuration.
    Uses production nested configurations with all features enabled.
    """

    model_config = SettingsConfigDict(
        **SETTINGS_MODEL_CONFIG,
        validate_default=True,
    )

    # Uses all production defaults from DocMindSettings and nested configurations
    # This ensures system tests validate the actual production setup


def create_test_settings(**overrides) -> MockDocMindSettings:
    """Factory function for creating test settings with specific overrides.

    Args:
        **overrides: Field overrides for the test settings

    Returns:
        MockDocMindSettings instance with applied overrides

    Example:
        settings = create_test_settings(
            enable_gpu_acceleration=True,
            debug=True
        )
    """
    return MockDocMindSettings(**overrides)


def create_integration_settings(**overrides) -> IntegrationTestSettings:
    """Factory function for creating integration test settings.

    Args:
        **overrides: Field overrides for the integration settings

    Returns:
        IntegrationTestSettings instance with applied overrides
    """
    return IntegrationTestSettings(**overrides)


def create_system_settings(**overrides) -> SystemTestSettings:
    """Factory function for creating system test settings.

    Args:
        **overrides: Field overrides for the system settings

    Returns:
        SystemTestSettings instance with applied overrides
    """
    return SystemTestSettings(**overrides)
