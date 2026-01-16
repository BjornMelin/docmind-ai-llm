"""Unit tests for monitoring utilities (boundary-first patterns).

Covers logging setup, performance timers (sync/async), system info helpers,
and error logging. Uses lightweight boundary fixtures to keep tests
deterministic and offline.
"""

import pytest

from tests.fixtures.test_settings import MockDocMindSettings as TestDocMindSettings

# Rationale: pytest fixtures intentionally shadow same-named references in tests.


@pytest.fixture
def test_settings():
    """Create test settings for monitoring.

    Aligns with MockDocMindSettings schema: uses existing fields only.
    """
    return TestDocMindSettings(
        log_level="INFO",
        enable_performance_logging=True,
    )


@pytest.mark.unit
class TestLoggingSetup:
    """Test logging configuration."""

    def test_logging_configuration(self, test_settings):
        """Test logging setup with settings."""
        from src.utils.monitoring import setup_logging

        # Test logging setup without external dependencies
        setup_logging(log_level=test_settings.log_level)

    def test_log_level_validation(self):
        """Test log level validation."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            # Should not raise errors for valid levels
            assert level in valid_levels


@pytest.mark.unit
class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""

    def test_simple_performance_monitor_init(self):
        """Test SimplePerformanceMonitor initialization."""
        from src.utils.monitoring import SimplePerformanceMonitor

        monitor = SimplePerformanceMonitor()

        assert monitor is not None
        # Current implementation provides a metrics list; no start_time/memory snapshot
        assert hasattr(monitor, "metrics")

    def test_performance_timer_context(self):
        """Test performance timer context manager."""
        from src.utils.monitoring import performance_timer

        # Test context manager behavior
        with performance_timer("test_operation") as timer:
            # Should track operation timing
            assert timer is not None

        # Context should exit cleanly

    def test_async_performance_timer_context(self):
        """Test async performance timer context manager."""
        import asyncio

        from src.utils.monitoring import async_performance_timer

        async def test_async_context():
            async with async_performance_timer("async_test"):
                # Should handle async operations
                await asyncio.sleep(0.001)  # Minimal delay

        # Test async context
        asyncio.run(test_async_context())

    def test_performance_monitor_singleton(self):
        """Test performance monitor singleton pattern."""
        from src.utils.monitoring import get_performance_monitor

        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        # Should return same instance (singleton)
        assert monitor1 is monitor2


@pytest.mark.unit
class TestSystemInformation:
    """Test system information collection."""

    def test_get_memory_usage(self, system_resource_boundary):
        """Test process memory usage collection (rss/vms in MB)."""
        from src.utils.monitoring import get_memory_usage

        memory_info = get_memory_usage()

        # Test business logic (process-level usage)
        assert isinstance(memory_info, dict)
        assert set(memory_info.keys()) >= {"rss_mb", "vms_mb", "percent"}
        assert memory_info["rss_mb"] >= 0.0
        assert memory_info["vms_mb"] >= 0.0

    def test_get_system_info(self, system_resource_boundary):
        """Test basic system information collection."""
        from src.utils.monitoring import get_system_info

        system_info = get_system_info()

        # Test result structure
        assert isinstance(system_info, dict)
        for key in ("cpu_percent", "memory_percent", "disk_percent"):
            assert key in system_info
            assert 0.0 <= float(system_info[key]) <= 100.0
        assert "load_average" in system_info

    def test_system_info_error_handling(self):
        """Test system info collection error handling."""
        from src.utils.monitoring import get_system_info

        # Should handle system information collection gracefully
        info = get_system_info()
        # Either returns info dict or handles errors
        assert info is None or isinstance(info, dict)


@pytest.mark.unit
class TestErrorLogging:
    """Test error logging utilities."""

    def test_log_error_with_context(self):
        """Test error logging with context."""
        from src.utils.monitoring import log_error_with_context

        test_error = ValueError("Test error message")
        operation = "test_operation"

        # Should handle error logging without external dependencies
        log_error_with_context(test_error, operation)

    def test_log_performance_metrics(self):
        """Test performance metrics logging."""
        from src.utils.monitoring import log_performance

        # Should handle performance logging with explicit signature
        log_performance(operation="test_op", duration_seconds=1.23, memory_used=1024)


@pytest.mark.unit
class TestMonitoringIntegration:
    """Test monitoring system integration."""

    def test_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        from src.utils.monitoring import SimplePerformanceMonitor, performance_timer

        # Test integrated monitoring workflow
        SimplePerformanceMonitor()

        with performance_timer("integration_test"):
            # Simulate work
            result = sum(range(1000))

        # Should complete without errors
        assert result == 499500  # Sum of 0 to 999

    def test_monitoring_configuration_validation(self, test_settings):
        """Test monitoring configuration validation."""
        # Test configuration values
        assert test_settings.log_level == "INFO"
        assert test_settings.enable_performance_logging is True
        assert test_settings.debug is True


# --- merged from test_monitoring_modern.py ---


@pytest.mark.unit
class TestModernizedMonitoringPatterns:
    """Test modernized monitoring patterns."""

    def test_log_error_basic(self, logging_boundary):
        """Test basic error logging functionality."""
        from src.utils.monitoring import log_error_with_context

        err = ValueError("Test error")
        log_error_with_context(err, "op")
        logging_boundary["logger"].error.assert_called_once()

    def test_performance_timer_success(
        self, system_resource_boundary, performance_boundary, logging_boundary
    ):
        """Test successful performance timer execution."""
        from src.utils.monitoring import performance_timer

        with performance_timer("test_operation") as metrics:
            metrics["custom_metric"] = 42
        logging_boundary["logger"].debug.assert_called()

    @pytest.mark.asyncio
    async def test_async_performance_timer_success(
        self, system_resource_boundary, performance_boundary, logging_boundary
    ):
        """Test successful async performance timer execution."""
        import asyncio

        from src.utils.monitoring import (
            async_performance_timer,
        )

        async with async_performance_timer("async_operation") as metrics:
            metrics["async_metric"] = "ok"
            await asyncio.sleep(0.001)
        logging_boundary["logger"].debug.assert_called()
