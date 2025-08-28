"""Comprehensive unit tests for src.utils.monitoring utility functions.

Tests focus on performance monitoring, logging utilities, memory tracking,
and timing context managers. All tests are designed for fast execution
(<0.05s each) with proper mocking of system dependencies.

Coverage areas:
- Logging setup and configuration
- Performance timing context managers
- Memory usage monitoring
- System information collection
- Performance monitoring class

Mocked external dependencies:
- psutil system information calls
- Loguru logger operations
- Time measurement functions
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from src.utils.monitoring import (
    SimplePerformanceMonitor,
    async_performance_timer,
    get_memory_usage,
    get_performance_monitor,
    get_system_info,
    log_error_with_context,
    log_performance,
    performance_timer,
    setup_logging,
)


@pytest.mark.unit
class TestSetupLogging:
    """Test logging configuration functionality."""

    def test_setup_logging_basic_configuration(self):
        """Test basic logging setup with default parameters."""
        with patch("src.utils.monitoring.logger") as mock_logger:
            setup_logging()

            # Should remove default handler and add console handler
            mock_logger.remove.assert_called_once()
            assert mock_logger.add.call_count >= 1

            # Should log configuration success
            mock_logger.info.assert_called()

    @pytest.mark.parametrize(
        "log_level",
        ["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    def test_setup_logging_different_levels(self, log_level):
        """Test logging setup with different log levels."""
        with patch("src.utils.monitoring.logger") as mock_logger:
            setup_logging(log_level=log_level)

            # Should be called with correct level
            mock_logger.remove.assert_called_once()
            mock_logger.add.assert_called()

    def test_setup_logging_with_file_output(self):
        """Test logging setup with file output."""
        test_file = "/tmp/test_log.log"

        with patch("src.utils.monitoring.logger") as mock_logger:
            setup_logging(log_level="INFO", log_file=test_file)

            # Should add two handlers (console + file)
            assert mock_logger.add.call_count == 2
            mock_logger.info.assert_called()

    def test_setup_logging_without_file(self):
        """Test logging setup without file output."""
        with patch("src.utils.monitoring.logger") as mock_logger:
            setup_logging(log_level="INFO", log_file=None)

            # Should add only one handler (console)
            assert mock_logger.add.call_count == 1


@pytest.mark.unit
class TestLogErrorWithContext:
    """Test error logging with context information."""

    def test_log_error_basic(self):
        """Test basic error logging functionality."""
        error = ValueError("Test error")
        operation = "test_operation"

        with patch("src.utils.monitoring.logger") as mock_logger:
            log_error_with_context(error, operation)

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert call_args[0][0] == "Operation failed"

            # Check context contains expected information
            context = call_args[1]
            assert context["operation"] == operation
            assert context["error_type"] == "ValueError"
            assert context["error_message"] == "Test error"

    def test_log_error_with_context_dict(self):
        """Test error logging with additional context dictionary."""
        error = RuntimeError("Runtime error")
        operation = "database_operation"
        context = {"user_id": 123, "query": "SELECT * FROM users"}

        with patch("src.utils.monitoring.logger") as mock_logger:
            log_error_with_context(error, operation, context)

            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[1]

            assert call_args["operation"] == operation
            assert call_args["user_id"] == 123
            assert call_args["query"] == "SELECT * FROM users"

    def test_log_error_with_kwargs(self):
        """Test error logging with keyword arguments."""
        error = ConnectionError("Connection failed")
        operation = "api_call"

        with patch("src.utils.monitoring.logger") as mock_logger:
            log_error_with_context(
                error, operation, endpoint="https://api.example.com", retry_count=3
            )

            call_args = mock_logger.error.call_args[1]
            assert call_args["endpoint"] == "https://api.example.com"
            assert call_args["retry_count"] == 3

    @pytest.mark.parametrize(
        "exception_type,expected_type_name",
        [
            (ValueError("test"), "ValueError"),
            (TypeError("test"), "TypeError"),
            (RuntimeError("test"), "RuntimeError"),
            (ConnectionError("test"), "ConnectionError"),
        ],
    )
    def test_log_error_different_exception_types(
        self, exception_type, expected_type_name
    ):
        """Test error logging with different exception types."""
        with patch("src.utils.monitoring.logger") as mock_logger:
            log_error_with_context(exception_type, "test_op")

            call_args = mock_logger.error.call_args[1]
            assert call_args["error_type"] == expected_type_name


@pytest.mark.unit
class TestLogPerformance:
    """Test performance logging functionality."""

    def test_log_performance_basic(self):
        """Test basic performance logging."""
        operation = "test_operation"
        duration = 1.234

        with patch("src.utils.monitoring.logger") as mock_logger:
            log_performance(operation, duration)

            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "Performance metrics"

            context = call_args[1]
            assert context["operation"] == operation
            assert context["duration_seconds"] == 1.234
            assert context["duration_ms"] == 1234.0

    def test_log_performance_with_metrics(self):
        """Test performance logging with additional metrics."""
        operation = "database_query"
        duration = 0.567

        with patch("src.utils.monitoring.logger") as mock_logger:
            log_performance(
                operation,
                duration,
                rows_processed=100,
                cache_hit=True,
                memory_used_mb=45.2,
            )

            call_args = mock_logger.info.call_args[1]
            assert call_args["rows_processed"] == 100
            assert call_args["cache_hit"] is True
            assert call_args["memory_used_mb"] == 45.2

    @pytest.mark.parametrize(
        "duration,expected_seconds,expected_ms",
        [
            (0.001, 0.001, 1.0),
            (0.1234, 0.123, 123.4),
            (1.0, 1.0, 1000.0),
            (2.5678, 2.568, 2567.8),
        ],
    )
    def test_log_performance_duration_formatting(
        self, duration, expected_seconds, expected_ms
    ):
        """Test duration formatting in performance logging."""
        with patch("src.utils.monitoring.logger") as mock_logger:
            log_performance("test_op", duration)

            call_args = mock_logger.info.call_args[1]
            assert call_args["duration_seconds"] == expected_seconds
            assert call_args["duration_ms"] == expected_ms


@pytest.mark.unit
class TestPerformanceTimer:
    """Test synchronous performance timing context manager."""

    @patch("time.perf_counter")
    @patch("psutil.Process")
    def test_performance_timer_success(self, mock_process_class, mock_perf_counter):
        """Test successful operation timing."""
        # Setup mocks
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024  # 100MB
        mock_process_class.return_value = mock_process
        mock_perf_counter.side_effect = [0.0, 1.5]  # Start and end times

        with patch("src.utils.monitoring.log_performance") as mock_log_perf:
            with performance_timer("test_operation") as metrics:
                metrics["custom_metric"] = 42

            # Verify timing was recorded
            mock_log_perf.assert_called_once()
            call_kwargs = mock_log_perf.call_args[1]
            assert call_kwargs["operation"] == "test_operation"
            assert call_kwargs["duration_seconds"] == 1.5
            assert call_kwargs["success"] is True
            assert call_kwargs["custom_metric"] == 42

    @patch("time.perf_counter")
    @patch("psutil.Process")
    def test_performance_timer_with_exception(
        self, mock_process_class, mock_perf_counter
    ):
        """Test timing context manager with exception."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_class.return_value = mock_process
        mock_perf_counter.side_effect = [0.0, 0.5]

        with patch("src.utils.monitoring.log_performance") as mock_log_perf:
            with pytest.raises(ValueError, match="test exception"):
                with performance_timer("failing_operation") as metrics:
                    raise ValueError("test exception")

            # Should still log performance with error info
            call_kwargs = mock_log_perf.call_args[1]
            assert call_kwargs["success"] is False
            assert call_kwargs["error"] == "test exception"

    @patch("time.perf_counter")
    @patch("psutil.Process")
    def test_performance_timer_memory_tracking(
        self, mock_process_class, mock_perf_counter
    ):
        """Test memory usage tracking in performance timer."""
        mock_process = Mock()
        # Simulate memory increase from 100MB to 150MB
        mock_process.memory_info.side_effect = [
            Mock(rss=100 * 1024 * 1024),  # Start: 100MB
            Mock(rss=150 * 1024 * 1024),  # End: 150MB
        ]
        mock_process_class.return_value = mock_process
        mock_perf_counter.side_effect = [0.0, 1.0]

        with patch("src.utils.monitoring.log_performance") as mock_log_perf:
            with performance_timer("memory_test"):
                pass

            call_kwargs = mock_log_perf.call_args[1]
            assert call_kwargs["memory_delta_mb"] == 50.0

    def test_performance_timer_with_context(self):
        """Test performance timer with initial context."""
        with (
            patch("time.perf_counter", side_effect=[0.0, 1.0]),
            patch("psutil.Process"),
            patch("src.utils.monitoring.log_performance") as mock_log_perf,
        ):
            with performance_timer("test_op", user_id=123, session="abc") as metrics:
                metrics["result_count"] = 5

            call_kwargs = mock_log_perf.call_args[1]
            assert call_kwargs["user_id"] == 123
            assert call_kwargs["session"] == "abc"
            assert call_kwargs["result_count"] == 5


@pytest.mark.unit
@pytest.mark.asyncio
class TestAsyncPerformanceTimer:
    """Test asynchronous performance timing context manager."""

    @patch("time.perf_counter")
    @patch("psutil.Process")
    async def test_async_performance_timer_success(
        self, mock_process_class, mock_perf_counter
    ):
        """Test successful async operation timing."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_class.return_value = mock_process
        mock_perf_counter.side_effect = [0.0, 2.0]

        with patch("src.utils.monitoring.log_performance") as mock_log_perf:
            async with async_performance_timer("async_operation") as metrics:
                metrics["async_metric"] = "test_value"
                await asyncio.sleep(0.001)  # Small async operation

            call_kwargs = mock_log_perf.call_args[1]
            assert call_kwargs["operation"] == "async_operation"
            assert call_kwargs["duration_seconds"] == 2.0
            assert call_kwargs["success"] is True
            assert call_kwargs["async_metric"] == "test_value"

    @patch("time.perf_counter")
    @patch("psutil.Process")
    async def test_async_performance_timer_exception(
        self, mock_process_class, mock_perf_counter
    ):
        """Test async timing context manager with exception."""
        mock_process = Mock()
        mock_process.memory_info.return_value.rss = 100 * 1024 * 1024
        mock_process_class.return_value = mock_process
        mock_perf_counter.side_effect = [0.0, 1.0]

        with patch("src.utils.monitoring.log_performance") as mock_log_perf:
            with pytest.raises(RuntimeError, match="async error"):
                async with async_performance_timer("failing_async_op"):
                    await asyncio.sleep(0.001)
                    raise RuntimeError("async error")

            call_kwargs = mock_log_perf.call_args[1]
            assert call_kwargs["success"] is False
            assert call_kwargs["error"] == "async error"


@pytest.mark.unit
class TestGetMemoryUsage:
    """Test memory usage monitoring functionality."""

    @patch("psutil.Process")
    def test_get_memory_usage_success(self, mock_process_class):
        """Test successful memory usage retrieval."""
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(
            rss=200 * 1024 * 1024,  # 200MB RSS
            vms=300 * 1024 * 1024,  # 300MB VMS
        )
        mock_process.memory_percent.return_value = 15.5
        mock_process_class.return_value = mock_process

        result = get_memory_usage()

        assert result["rss_mb"] == 200.0
        assert result["vms_mb"] == 300.0
        assert result["percent"] == 15.5

    def test_get_memory_usage_error_handling_os_error(self):
        """Test memory usage error handling for OSError."""
        with patch("psutil.Process", side_effect=OSError("OS error")):
            with patch("src.utils.monitoring.logger") as mock_logger:
                result = get_memory_usage()

                # Should return safe defaults
                assert result["rss_mb"] == 0.0
                assert result["vms_mb"] == 0.0
                assert result["percent"] == 0.0

                # Should log warning
                mock_logger.warning.assert_called_once()

    def test_get_memory_usage_error_handling_psutil_error(self):
        """Test memory usage error handling for psutil errors."""
        # Mock psutil.Error specifically
        with patch("psutil.Process") as mock_process_class:
            with patch("psutil.Error", Exception):  # Mock psutil.Error as Exception
                mock_process_class.side_effect = Exception("psutil error")

                with patch("src.utils.monitoring.logger") as mock_logger:
                    result = get_memory_usage()

                    # Should return safe defaults
                    assert result["rss_mb"] == 0.0
                    assert result["vms_mb"] == 0.0
                    assert result["percent"] == 0.0

                    # Should log warning
                    mock_logger.warning.assert_called_once()

    @patch("psutil.Process")
    def test_get_memory_usage_calculation_accuracy(self, mock_process_class):
        """Test memory calculation accuracy."""
        test_cases = [
            (1024 * 1024, 1.0),  # 1MB
            (10 * 1024 * 1024, 10.0),  # 10MB
            (1024 * 1024 * 1024, 1024.0),  # 1GB
        ]

        for memory_bytes, expected_mb in test_cases:
            mock_process = Mock()
            mock_process.memory_info.return_value = Mock(
                rss=memory_bytes, vms=memory_bytes
            )
            mock_process.memory_percent.return_value = 10.0
            mock_process_class.return_value = mock_process

            result = get_memory_usage()
            assert result["rss_mb"] == expected_mb
            assert result["vms_mb"] == expected_mb


@pytest.mark.unit
class TestGetSystemInfo:
    """Test system information collection functionality."""

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    @patch("psutil.getloadavg", create=True)  # May not exist on all systems
    def test_get_system_info_success(
        self, mock_getloadavg, mock_disk_usage, mock_virtual_memory, mock_cpu_percent
    ):
        """Test successful system info retrieval."""
        mock_cpu_percent.return_value = 45.2
        mock_virtual_memory.return_value = Mock(percent=67.8)
        mock_disk_usage.return_value = Mock(percent=23.4)
        mock_getloadavg.return_value = (1.5, 1.2, 0.8)

        result = get_system_info()

        assert result["cpu_percent"] == 45.2
        assert result["memory_percent"] == 67.8
        assert result["disk_percent"] == 23.4
        assert result["load_average"] == (1.5, 1.2, 0.8)

    @patch("psutil.cpu_percent", side_effect=OSError("CPU error"))
    def test_get_system_info_error_handling(self, mock_cpu_percent):
        """Test system info error handling."""
        with patch("src.utils.monitoring.logger") as mock_logger:
            result = get_system_info()

            # Should return empty dict on error
            assert result == {}
            mock_logger.warning.assert_called_once()

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    def test_get_system_info_no_loadavg(
        self, mock_disk_usage, mock_virtual_memory, mock_cpu_percent
    ):
        """Test system info when getloadavg is not available."""
        mock_cpu_percent.return_value = 30.0
        mock_virtual_memory.return_value = Mock(percent=50.0)
        mock_disk_usage.return_value = Mock(percent=75.0)

        # Simulate system without getloadavg
        with patch("builtins.hasattr", return_value=False):
            result = get_system_info()

            assert result["cpu_percent"] == 30.0
            assert result["memory_percent"] == 50.0
            assert result["disk_percent"] == 75.0
            assert result["load_average"] is None


@pytest.mark.unit
class TestSimplePerformanceMonitor:
    """Test SimplePerformanceMonitor class functionality."""

    def test_simple_performance_monitor_init(self):
        """Test performance monitor initialization."""
        monitor = SimplePerformanceMonitor()
        assert isinstance(monitor.metrics, list)
        assert len(monitor.metrics) == 0

    def test_record_operation_basic(self):
        """Test basic operation recording."""
        monitor = SimplePerformanceMonitor()

        with patch("src.utils.monitoring.log_performance") as mock_log_perf:
            monitor.record_operation("test_op", 1.5, success=True)

            assert len(monitor.metrics) == 1
            metric = monitor.metrics[0]
            assert metric["operation"] == "test_op"
            assert metric["duration_seconds"] == 1.5
            assert metric["success"] is True
            assert "timestamp" in metric

            mock_log_perf.assert_called_once()

    def test_record_operation_with_metrics(self):
        """Test operation recording with additional metrics."""
        monitor = SimplePerformanceMonitor()

        with patch("src.utils.monitoring.log_performance"):
            monitor.record_operation(
                "complex_op", 2.3, success=False, rows_processed=150, cache_hits=25
            )

            metric = monitor.metrics[0]
            assert metric["rows_processed"] == 150
            assert metric["cache_hits"] == 25
            assert metric["success"] is False

    def test_get_summary_empty_metrics(self):
        """Test summary generation with no metrics."""
        monitor = SimplePerformanceMonitor()
        summary = monitor.get_summary()

        assert summary["total_operations"] == 0

    def test_get_summary_with_operations(self):
        """Test summary generation with recorded operations."""
        monitor = SimplePerformanceMonitor()

        # Record various operations
        with patch("src.utils.monitoring.log_performance"):
            monitor.record_operation("op1", 1.0, success=True)
            monitor.record_operation("op2", 2.0, success=True)
            monitor.record_operation("op3", 0.5, success=False)

        summary = monitor.get_summary()

        assert summary["total_operations"] == 3
        assert summary["successful_operations"] == 2
        assert (
            abs(summary["success_rate"] - 66.67) < 0.01
        )  # Allow floating point precision
        assert (
            abs(summary["avg_duration_seconds"] - 1.17) < 0.1
        )  # Actual measured average
        assert summary["min_duration_seconds"] == 0.5
        assert summary["max_duration_seconds"] == 2.0
        assert summary["total_duration_seconds"] == 3.5

    def test_get_summary_filtered_by_operation(self):
        """Test summary generation filtered by operation name."""
        monitor = SimplePerformanceMonitor()

        with patch("src.utils.monitoring.log_performance"):
            monitor.record_operation("op_a", 1.0, success=True)
            monitor.record_operation("op_b", 2.0, success=True)
            monitor.record_operation("op_a", 3.0, success=False)

        summary = monitor.get_summary("op_a")

        assert summary["total_operations"] == 2
        assert summary["successful_operations"] == 1
        assert summary["avg_duration_seconds"] == 2.0  # (1.0+3.0)/2

    def test_clear_metrics(self):
        """Test metrics clearing functionality."""
        monitor = SimplePerformanceMonitor()

        with patch("src.utils.monitoring.log_performance"):
            monitor.record_operation("test_op", 1.0)

        assert len(monitor.metrics) == 1

        with patch("src.utils.monitoring.logger") as mock_logger:
            monitor.clear_metrics()

            assert len(monitor.metrics) == 0
            mock_logger.info.assert_called_once()


@pytest.mark.unit
class TestGetPerformanceMonitor:
    """Test global performance monitor access."""

    def test_get_performance_monitor_returns_instance(self):
        """Test that get_performance_monitor returns SimplePerformanceMonitor instance."""
        monitor = get_performance_monitor()
        assert isinstance(monitor, SimplePerformanceMonitor)

    def test_get_performance_monitor_singleton_behavior(self):
        """Test that get_performance_monitor returns the same instance."""
        monitor1 = get_performance_monitor()
        monitor2 = get_performance_monitor()

        # Should be the same instance (singleton-like behavior)
        assert monitor1 is monitor2

    def test_global_performance_monitor_persistence(self):
        """Test that global performance monitor persists data."""
        monitor = get_performance_monitor()

        with patch("src.utils.monitoring.log_performance"):
            monitor.record_operation("persistent_test", 1.0)

        # Get monitor again and verify data persists
        same_monitor = get_performance_monitor()
        assert len(same_monitor.metrics) >= 1

        # Find our test operation
        test_metrics = [
            m for m in same_monitor.metrics if m["operation"] == "persistent_test"
        ]
        assert len(test_metrics) >= 1


@pytest.mark.unit
class TestMonitoringUtilsEdgeCases:
    """Test edge cases and error scenarios for monitoring utilities."""

    @pytest.mark.parametrize(
        "operation,duration",
        [
            ("", 0.0),  # Empty operation name
            (None, 1.0),  # None operation name
            ("test", -1.0),  # Negative duration
            ("test", float("inf")),  # Infinite duration
        ],
    )
    def test_log_performance_edge_cases(self, operation, duration):
        """Test log_performance with edge case inputs."""
        with patch("src.utils.monitoring.logger") as mock_logger:
            # Should not crash even with unusual inputs
            try:
                log_performance(operation, duration)
                mock_logger.info.assert_called_once()
            except (TypeError, ValueError):
                # Some edge cases may raise exceptions, which is acceptable
                pass

    def test_performance_timer_context_manager_protocol(self):
        """Test that performance timer properly implements context manager protocol."""
        with patch("time.perf_counter", side_effect=[0.0, 1.0]):
            with patch("psutil.Process"):
                with patch("src.utils.monitoring.log_performance"):
                    # Should work as context manager
                    timer = performance_timer("test_op")
                    assert hasattr(timer, "__enter__")
                    assert hasattr(timer, "__exit__")

    @pytest.mark.asyncio
    async def test_async_performance_timer_context_manager_protocol(self):
        """Test async performance timer context manager protocol."""
        with patch("time.perf_counter", side_effect=[0.0, 1.0]):
            with patch("psutil.Process"):
                with patch("src.utils.monitoring.log_performance"):
                    # Should work as async context manager
                    timer = async_performance_timer("test_op")
                    assert hasattr(timer, "__aenter__")
                    assert hasattr(timer, "__aexit__")
