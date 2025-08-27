"""Validation tests for the clean test infrastructure implementation.

This test validates that the new BaseSettings subclass pattern works correctly
and provides proper isolation from production configuration.

Tests the Phase 3B implementation requirements:
- ✅ Zero custom backward compatibility code
- ✅ Standard library patterns only (pytest + pydantic-settings)
- ✅ Test isolation from production configuration
- ✅ Clean fixture architecture
- ✅ No production code contamination in tests
"""

from pathlib import Path

import pytest

from src.config.settings import DocMindSettings
from tests.fixtures.test_settings import (
    IntegrationMockSettings,
    MockDocMindSettings,
    SystemMockSettings,
    create_test_settings,
)


def test_test_settings_isolation():
    """Test settings should be isolated from production settings."""
    test_settings = MockDocMindSettings()
    prod_settings = DocMindSettings()

    # Test defaults should differ from production for key testing optimizations
    # Test settings have flat fields while production uses nested configs
    assert test_settings.debug != prod_settings.debug
    assert test_settings.log_level != prod_settings.log_level

    # Test settings should have test-optimized values
    assert test_settings.enable_gpu_acceleration is False  # Test-specific field
    assert test_settings.debug is True  # Enabled for testing
    assert test_settings.log_level == "DEBUG"
    assert test_settings.enable_dspy_optimization is False  # Disabled for speed

    # Production uses nested configs while test settings extend with flat fields
    assert hasattr(prod_settings, "vllm")  # Production has nested configs
    assert hasattr(prod_settings, "agents")
    assert hasattr(prod_settings, "embedding")


def test_integration_settings_moderate_performance():
    """Integration settings should balance speed with realistic configuration."""
    integration_settings = IntegrationMockSettings()
    test_settings = MockDocMindSettings()

    # Should be more realistic than unit test settings
    assert integration_settings.enable_gpu_acceleration is True
    assert integration_settings.context_window_size > test_settings.context_window_size
    assert (
        integration_settings.agent_decision_timeout
        > test_settings.agent_decision_timeout
    )
    assert integration_settings.enable_document_caching is True

    # Integration settings should have moderate values for performance
    assert integration_settings.context_window_size == 4096  # Moderate size
    assert integration_settings.enable_performance_logging is True


def test_system_settings_production_defaults():
    """System settings should use production defaults."""
    system_settings = SystemMockSettings()
    prod_settings = DocMindSettings()

    # Should match production defaults for core settings
    assert system_settings.debug == prod_settings.debug
    assert system_settings.log_level == prod_settings.log_level
    assert system_settings.app_name == prod_settings.app_name

    # Should have nested configuration models
    assert hasattr(system_settings, "vllm")
    assert hasattr(system_settings, "agents")
    assert hasattr(system_settings, "embedding")

    # Should use .env loading for system tests
    assert system_settings.model_config["env_file"] == ".env"


def test_settings_with_fixtures(test_settings):
    """Test that fixtures work correctly."""
    # Should be MockDocMindSettings instance
    assert isinstance(test_settings, MockDocMindSettings)

    # Should have temporary directories set
    assert test_settings.data_dir != Path("./data")  # Not default production path
    assert "test_settings" in str(test_settings.data_dir)  # Has test prefix

    # Should have test-optimized defaults
    assert test_settings.enable_gpu_acceleration is False
    assert test_settings.debug is True


def test_settings_with_overrides_fixture(settings_with_overrides):
    """Test the factory fixture for settings overrides."""
    # Test model_copy pattern works
    settings = settings_with_overrides(
        enable_gpu_acceleration=True, context_window_size=2048, chunk_size=1024
    )

    assert settings.enable_gpu_acceleration is True
    assert settings.context_window_size == 2048
    assert settings.chunk_size == 1024

    # Other defaults should be preserved
    assert settings.debug is True  # Test default preserved
    assert settings.log_level == "DEBUG"


def test_settings_with_temp_dirs_fixture(settings_with_temp_dirs):
    """Test the temporary directory fixture."""
    assert isinstance(settings_with_temp_dirs, MockDocMindSettings)

    # Should have test paths
    data_path = Path(settings_with_temp_dirs.data_dir)
    cache_path = Path(settings_with_temp_dirs.cache_dir)

    assert "tmp" in str(data_path).lower()  # Uses temp directory
    assert data_path != cache_path  # Different directories

    # Should have test defaults
    assert settings_with_temp_dirs.enable_gpu_acceleration is False


def test_factory_functions():
    """Test the factory functions work correctly."""
    # Test settings factory
    test_settings = create_test_settings(
        context_window_size=512, enable_performance_logging=True
    )
    assert isinstance(test_settings, MockDocMindSettings)
    assert test_settings.context_window_size == 512
    assert test_settings.enable_performance_logging is True

    # Should preserve test defaults
    assert test_settings.enable_gpu_acceleration is False


def test_environment_variable_isolation():
    """Test that test settings use different environment prefix."""
    test_settings = MockDocMindSettings()
    prod_settings = DocMindSettings()

    # Different environment prefixes for isolation
    assert test_settings.model_config["env_prefix"] == "DOCMIND_TEST_"
    assert prod_settings.model_config["env_prefix"] == "DOCMIND_"

    # Test settings shouldn't load .env file
    assert test_settings.model_config["env_file"] is None
    assert prod_settings.model_config["env_file"] == ".env"


def test_adr_compliance_separation():
    """Test that test settings properly separate from production ADR requirements."""
    test_settings = MockDocMindSettings()
    prod_settings = DocMindSettings()

    # Test can use faster timeouts while production follows ADR-024
    # Note: This test validates that test settings can deviate for performance
    # while production maintains ADR compliance
    assert test_settings.agent_decision_timeout == 100  # Test optimized
    # Production timeout will be fixed to 200ms in Phase 3 (production cleanup)

    # Test settings can use smaller dimensions for speed
    assert test_settings.embedding_dimension == 384  # Smaller for tests
    # Production will use BGE-M3 1024 dimensions per ADR-002


def test_inheritance_pattern():
    """Test that inheritance pattern works correctly."""
    # MockDocMindSettings should inherit from DocMindSettings
    assert issubclass(MockDocMindSettings, DocMindSettings)
    assert issubclass(IntegrationMockSettings, MockDocMindSettings)
    assert issubclass(SystemMockSettings, DocMindSettings)

    # But each should have its own configuration
    test_settings = MockDocMindSettings()
    integration_settings = IntegrationMockSettings()

    # Should have different environment prefixes
    assert test_settings.model_config["env_prefix"] == "DOCMIND_TEST_"
    assert integration_settings.model_config["env_prefix"] == "DOCMIND_INTEGRATION_"


def test_no_production_contamination():
    """Test that no production logic is contaminated with test-specific code."""
    # This test validates that production settings remain clean
    # and test settings exist in separate classes

    import inspect

    from src.config.settings import DocMindSettings

    # Production settings should not have test-specific logic
    prod_source = inspect.getsource(DocMindSettings)

    # Should not contain test-specific patterns (ignoring documentation)
    assert "TEST_" not in prod_source  # No test environment prefixes
    assert "test compatibility" not in prod_source.lower()  # No backward compat code
    assert "_sync_nested_models" not in prod_source  # No complex sync logic

    # Test settings should be completely separate
    test_source = inspect.getsource(MockDocMindSettings)
    assert "test" in test_source.lower()  # Should have test-specific code


@pytest.mark.unit
def test_performance_optimized_defaults():
    """Test that test settings have performance-optimized defaults."""
    test_settings = MockDocMindSettings()

    # Memory limits should be smaller for tests
    assert test_settings.max_memory_gb <= 1.0
    assert test_settings.max_vram_gb <= 2.0
    assert test_settings.max_document_size_mb <= 10

    # Batch sizes should be smaller
    assert test_settings.default_batch_size <= 5
    assert test_settings.top_k <= 3
    assert test_settings.reranking_top_k <= 2

    # Expensive operations disabled
    assert test_settings.enable_dspy_optimization is False
    assert test_settings.enable_performance_logging is False
    assert test_settings.enable_document_caching is False


@pytest.mark.integration
def test_fixture_integration_with_pytest_markers(integration_settings):
    """Test that integration fixtures work with pytest markers."""
    # This test should only run with integration marker
    assert isinstance(integration_settings, IntegrationMockSettings)

    # Should have integration-appropriate settings
    assert integration_settings.enable_gpu_acceleration is True
    assert integration_settings.enable_document_caching is True
    assert integration_settings.context_window_size == 4096


@pytest.mark.system
def test_system_fixture_production_compliance(system_settings):
    """Test that system fixtures use production configuration."""
    # This test should only run with system marker
    assert isinstance(system_settings, SystemMockSettings)

    # Should use production defaults and nested configuration
    assert hasattr(system_settings, "vllm")  # Nested config structure
    assert hasattr(system_settings, "agents")
    assert system_settings.debug == False  # Production default
    assert system_settings.log_level == "INFO"  # Production default
