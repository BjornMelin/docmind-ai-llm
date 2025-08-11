"""Unified performance monitoring utilities for DocMind AI.

This module provides comprehensive performance monitoring capabilities that combine
and improve upon the various monitoring implementations throughout the codebase.
Consolidates performance tracking to follow DRY principles and provide consistent
metrics across all application components.

Key features:
- Unified PerformanceMonitor supporting both general and document-specific metrics
- Structured metrics using dataclasses for type safety
- Async context manager support for seamless integration
- Memory usage tracking with automatic cleanup
- Throughput calculations and success rate monitoring
- Comprehensive reporting and summary generation

Example:
    Basic async operation monitoring::

        from utils.monitoring import get_performance_monitor
        import asyncio

        monitor = get_performance_monitor()

        # Monitor async operations
        async with monitor.measure("database_query") as metrics:
            result = await database.query("SELECT * FROM users")
            metrics.record_result(len(result))

        # Monitor document processing
        async with monitor.measure_document_processing(
            "pdf_parsing", file_path
        ) as metrics:
            documents = await parse_pdf(file_path)
            metrics.update_document_metrics(
                document_count=len(documents),
                table_count=count_tables(documents)
            )
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
from loguru import logger

from .logging_utils import log_performance


@dataclass
class BaseMetrics:
    """Base metrics tracked for all operations with comprehensive psutil data."""

    operation: str
    start_time: float = field(default_factory=time.perf_counter)
    duration_seconds: float = 0.0
    memory_delta_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    success: bool = False
    error_message: str | None = None
    custom_data: dict[str, Any] = field(default_factory=dict)

    # Enhanced psutil-based metrics
    cpu_percent: float = 0.0
    cpu_times_delta: dict[str, float] = field(default_factory=dict)
    memory_percent: float = 0.0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_count: int = 0
    io_write_count: int = 0
    ctx_switches_voluntary: int = 0
    ctx_switches_involuntary: int = 0
    num_threads: int = 0
    num_fds: int = 0  # File descriptors (Linux/macOS)
    cpu_affinity: list[int] = field(default_factory=list)


@dataclass
class DocumentProcessingMetrics(BaseMetrics):
    """Extended metrics for document processing operations with file I/O tracking."""

    file_path: str = ""
    file_size_mb: float = 0.0
    document_count: int = 0
    table_count: int = 0
    image_count: int = 0
    throughput_mb_per_sec: float = 0.0

    # Enhanced file operation metrics
    file_read_time_ms: float = 0.0
    disk_io_efficiency: float = 0.0  # MB/s actual vs theoretical
    memory_mapped_size_mb: float = 0.0
    page_faults: int = 0

    def calculate_throughput(self) -> None:
        """Calculate processing throughput in MB/s."""
        if self.duration_seconds > 0:
            self.throughput_mb_per_sec = self.file_size_mb / self.duration_seconds

    def calculate_io_efficiency(self) -> None:
        """Calculate disk I/O efficiency based on actual vs expected performance."""
        if self.file_read_time_ms > 0 and self.file_size_mb > 0:
            actual_throughput = self.file_size_mb / (self.file_read_time_ms / 1000)
            # Assume reasonable SSD performance baseline of 500 MB/s
            expected_throughput = 500.0
            self.disk_io_efficiency = (actual_throughput / expected_throughput) * 100


class PerformanceMonitor:
    """Unified performance monitoring with comprehensive psutil-based system metrics.

    Combines the best features from all existing PerformanceMonitor implementations
    with enhanced psutil capabilities:
    - Context manager pattern for automatic timing and resource tracking
    - Comprehensive CPU, memory, I/O, and system metrics
    - Process.oneshot() optimization for efficient multi-attribute access
    - Document-specific metrics support with file I/O analysis
    - System-wide resource monitoring and analysis
    - Structured logging with loguru for metrics output

    This class replaces all other PerformanceMonitor implementations in the codebase
    to eliminate duplication and provide consistent, library-first monitoring.
    """

    def __init__(self):
        """Initialize the unified performance monitor with psutil process handle."""
        self.metrics: dict[str, list[BaseMetrics]] = {}
        self.operation_counts: dict[str, int] = {}
        self.error_counts: dict[str, int] = {}
        self._process = psutil.Process()
        self._system_metrics_cache = {}
        self._cache_timeout = 1.0  # Cache system metrics for 1 second

    @asynccontextmanager
    async def measure(self, operation_name: str, **custom_data):
        """Measure async operation performance with comprehensive psutil metrics.

        Args:
            operation_name: Name of the operation being measured
            **custom_data: Additional data to include in metrics

        Yields:
            BaseMetrics instance for recording additional data during operation

        Note:
            Uses psutil.Process.oneshot() for efficient multi-attribute access
            and comprehensive resource monitoring including CPU, memory, I/O, and
            context switches.
        """
        metric = BaseMetrics(operation=operation_name, custom_data=custom_data)

        # Capture initial system and process state using oneshot for efficiency
        with self._process.oneshot():
            start_memory_info = self._process.memory_info()
            start_cpu_times = self._process.cpu_times()
            try:
                start_io_counters = self._process.io_counters()
            except (psutil.AccessDenied, AttributeError):
                start_io_counters = None
            try:
                start_ctx_switches = self._process.num_ctx_switches()
            except (psutil.AccessDenied, AttributeError):
                start_ctx_switches = None

            metric.num_threads = self._process.num_threads()
            try:
                metric.num_fds = self._process.num_fds()
            except (psutil.AccessDenied, AttributeError):
                metric.num_fds = 0
            try:
                metric.cpu_affinity = self._process.cpu_affinity()
            except (psutil.AccessDenied, AttributeError):
                metric.cpu_affinity = []

        # Track operation count
        self.operation_counts[operation_name] = (
            self.operation_counts.get(operation_name, 0) + 1
        )

        try:
            yield metric
            metric.success = True
        except Exception as e:
            # Track error
            self.error_counts[operation_name] = (
                self.error_counts.get(operation_name, 0) + 1
            )
            metric.error_message = str(e)
            logger.error(f"Operation {operation_name} failed: {e}")
            raise
        finally:
            # Calculate final metrics with comprehensive psutil data
            metric.duration_seconds = time.perf_counter() - metric.start_time

            # Capture final state using oneshot for efficiency
            with self._process.oneshot():
                end_memory_info = self._process.memory_info()
                end_cpu_times = self._process.cpu_times()
                try:
                    end_io_counters = self._process.io_counters()
                except (psutil.AccessDenied, AttributeError):
                    end_io_counters = None
                try:
                    end_ctx_switches = self._process.num_ctx_switches()
                except (psutil.AccessDenied, AttributeError):
                    end_ctx_switches = None

                # Calculate deltas
                metric.memory_delta_mb = (
                    (end_memory_info.rss - start_memory_info.rss) / 1024 / 1024
                )
                metric.memory_percent = self._process.memory_percent()

                # CPU usage over the operation duration
                if metric.duration_seconds > 0:
                    metric.cpu_percent = self._process.cpu_percent()

                # CPU times delta
                metric.cpu_times_delta = {
                    "user": end_cpu_times.user - start_cpu_times.user,
                    "system": end_cpu_times.system - start_cpu_times.system,
                }

                # I/O counters delta
                if start_io_counters and end_io_counters:
                    metric.io_read_bytes = (
                        end_io_counters.read_bytes - start_io_counters.read_bytes
                    )
                    metric.io_write_bytes = (
                        end_io_counters.write_bytes - start_io_counters.write_bytes
                    )
                    metric.io_read_count = (
                        end_io_counters.read_count - start_io_counters.read_count
                    )
                    metric.io_write_count = (
                        end_io_counters.write_count - start_io_counters.write_count
                    )

                # Context switches delta
                if start_ctx_switches and end_ctx_switches:
                    metric.ctx_switches_voluntary = (
                        end_ctx_switches.voluntary - start_ctx_switches.voluntary
                    )
                    metric.ctx_switches_involuntary = (
                        end_ctx_switches.involuntary - start_ctx_switches.involuntary
                    )

            # Store metric
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(metric)

            # Enhanced structured logging with comprehensive psutil metrics
            log_performance(
                operation_name,
                metric.duration_seconds,
                success=metric.success,
                memory_delta_mb=metric.memory_delta_mb,
                memory_percent=metric.memory_percent,
                cpu_percent=metric.cpu_percent,
                cpu_user_time=metric.cpu_times_delta.get("user", 0),
                cpu_system_time=metric.cpu_times_delta.get("system", 0),
                io_read_mb=metric.io_read_bytes / 1024 / 1024
                if metric.io_read_bytes
                else 0,
                io_write_mb=metric.io_write_bytes / 1024 / 1024
                if metric.io_write_bytes
                else 0,
                io_operations=metric.io_read_count + metric.io_write_count,
                ctx_switches=metric.ctx_switches_voluntary
                + metric.ctx_switches_involuntary,
                num_threads=metric.num_threads,
                **custom_data,
            )

    @asynccontextmanager
    async def measure_document_processing(self, operation_name: str, file_path: str):
        """Measure document processing operations with comprehensive file I/O metrics.

        Args:
            operation_name: Name of the document processing operation
            file_path: Path to the file being processed

        Yields:
            DocumentProcessingMetrics instance for recording processing results

        Note:
            Enhanced with psutil-based file I/O monitoring, disk efficiency analysis,
            and memory-mapped file size tracking for comprehensive performance insights.
        """
        # Calculate file size and check if file exists
        file_size_mb = 0.0
        file_stat = None
        try:
            file_stat = Path(file_path).stat()
            file_size_mb = file_stat.st_size / 1024 / 1024
        except (OSError, FileNotFoundError):
            logger.warning(f"Could not determine file size for {file_path}")

        metric = DocumentProcessingMetrics(
            operation=operation_name, file_path=file_path, file_size_mb=file_size_mb
        )

        # Capture initial state with comprehensive metrics
        file_read_start = time.perf_counter()
        with self._process.oneshot():
            start_memory_info = self._process.memory_info()
            start_cpu_times = self._process.cpu_times()
            try:
                start_io_counters = self._process.io_counters()
            except (psutil.AccessDenied, AttributeError):
                start_io_counters = None
            try:
                start_ctx_switches = self._process.num_ctx_switches()
            except (psutil.AccessDenied, AttributeError):
                start_ctx_switches = None

            # Check for memory-mapped files
            try:
                memory_maps = self._process.memory_maps(grouped=True)
                for mmap in memory_maps:
                    if file_path in mmap.path:
                        metric.memory_mapped_size_mb = mmap.size / 1024 / 1024
                        break
            except (psutil.AccessDenied, AttributeError):
                pass

        # Track operation count
        self.operation_counts[operation_name] = (
            self.operation_counts.get(operation_name, 0) + 1
        )

        try:
            yield metric
            metric.success = True
        except Exception as e:
            # Track error
            self.error_counts[operation_name] = (
                self.error_counts.get(operation_name, 0) + 1
            )
            metric.error_message = str(e)
            logger.error(f"Document processing {operation_name} failed: {e}")
            raise
        finally:
            # Calculate comprehensive final metrics
            metric.duration_seconds = time.perf_counter() - metric.start_time
            metric.file_read_time_ms = (time.perf_counter() - file_read_start) * 1000

            # Capture final state with comprehensive psutil data
            with self._process.oneshot():
                end_memory_info = self._process.memory_info()
                end_cpu_times = self._process.cpu_times()
                try:
                    end_io_counters = self._process.io_counters()
                except (psutil.AccessDenied, AttributeError):
                    end_io_counters = None
                try:
                    end_ctx_switches = self._process.num_ctx_switches()
                except (psutil.AccessDenied, AttributeError):
                    end_ctx_switches = None

                # Calculate comprehensive deltas
                metric.memory_delta_mb = (
                    (end_memory_info.rss - start_memory_info.rss) / 1024 / 1024
                )
                metric.memory_percent = self._process.memory_percent()

                # CPU metrics
                if metric.duration_seconds > 0:
                    metric.cpu_percent = self._process.cpu_percent()

                metric.cpu_times_delta = {
                    "user": end_cpu_times.user - start_cpu_times.user,
                    "system": end_cpu_times.system - start_cpu_times.system,
                }

                # I/O metrics for file operations
                if start_io_counters and end_io_counters:
                    metric.io_read_bytes = (
                        end_io_counters.read_bytes - start_io_counters.read_bytes
                    )
                    metric.io_write_bytes = (
                        end_io_counters.write_bytes - start_io_counters.write_bytes
                    )
                    metric.io_read_count = (
                        end_io_counters.read_count - start_io_counters.read_count
                    )
                    metric.io_write_count = (
                        end_io_counters.write_count - start_io_counters.write_count
                    )

                # Context switches and page faults
                if start_ctx_switches and end_ctx_switches:
                    metric.ctx_switches_voluntary = (
                        end_ctx_switches.voluntary - start_ctx_switches.voluntary
                    )
                    metric.ctx_switches_involuntary = (
                        end_ctx_switches.involuntary - start_ctx_switches.involuntary
                    )

                # Thread and file descriptor counts
                metric.num_threads = self._process.num_threads()
                try:
                    metric.num_fds = self._process.num_fds()
                except (psutil.AccessDenied, AttributeError):
                    metric.num_fds = 0

            # Calculate derived metrics
            metric.calculate_throughput()
            metric.calculate_io_efficiency()

            # Store metric
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            self.metrics[operation_name].append(metric)

            # Enhanced structured logging for document processing
            status = "✅" if metric.success else "❌"
            logger.info(
                f"{status} {operation_name}: {Path(file_path).name}",
                extra={
                    "performance": {
                        "operation": operation_name,
                        "file_size_mb": metric.file_size_mb,
                        "duration_seconds": metric.duration_seconds,
                        "throughput_mb_per_sec": metric.throughput_mb_per_sec,
                        "file_read_time_ms": metric.file_read_time_ms,
                        "disk_io_efficiency_percent": metric.disk_io_efficiency,
                        "memory_delta_mb": metric.memory_delta_mb,
                        "memory_mapped_mb": metric.memory_mapped_size_mb,
                        "cpu_percent": metric.cpu_percent,
                        "io_read_mb": metric.io_read_bytes / 1024 / 1024
                        if metric.io_read_bytes
                        else 0,
                        "io_write_mb": metric.io_write_bytes / 1024 / 1024
                        if metric.io_write_bytes
                        else 0,
                        "io_operations": metric.io_read_count + metric.io_write_count,
                        "ctx_switches": metric.ctx_switches_voluntary
                        + metric.ctx_switches_involuntary,
                        "document_count": metric.document_count,
                        "table_count": metric.table_count,
                        "image_count": metric.image_count,
                        "success": metric.success,
                        "error": metric.error_message,
                    }
                },
            )

    async def measure_async_operation(self, name: str, operation: callable) -> Any:
        """Measure and record async operation performance (compatibility method).

        Args:
            name: Name of the operation for metrics tracking
            operation: Async callable to measure

        Returns:
            Result of the operation

        Note:
            This method maintains compatibility with existing code while using
            the enhanced psutil-based monitoring system.
        """
        async with self.measure(name):
            return await operation()

    @asynccontextmanager
    async def system_resource_monitor(self, operation_name: str):
        """Context manager for comprehensive system resource monitoring.

        Args:
            operation_name: Name of the operation for tracking

        Yields:
            Dictionary with initial system metrics for comparison

        Note:
            Monitors system-wide CPU, memory, disk I/O, and network I/O
            before and after operations. Useful for understanding system
            impact of resource-intensive operations.
        """
        initial_metrics = self.get_system_metrics()
        start_time = time.perf_counter()

        logger.info(
            f"Starting system monitoring for {operation_name}",
            extra={
                "system_cpu_percent": initial_metrics.get("cpu", {}).get("percent", 0),
                "system_memory_percent": initial_metrics.get("memory", {})
                .get("virtual", {})
                .get("percent", 0),
                "system_load_avg": initial_metrics.get("load_avg"),
            },
        )

        try:
            yield initial_metrics
        finally:
            duration = time.perf_counter() - start_time
            final_metrics = self.get_system_metrics()

            # Calculate system resource deltas
            cpu_delta = final_metrics.get("cpu", {}).get(
                "percent", 0
            ) - initial_metrics.get("cpu", {}).get("percent", 0)

            memory_delta = final_metrics.get("memory", {}).get("virtual", {}).get(
                "percent", 0
            ) - initial_metrics.get("memory", {}).get("virtual", {}).get("percent", 0)

            # Disk I/O delta if available
            disk_read_delta = 0
            disk_write_delta = 0
            if initial_metrics.get("disk_io") and final_metrics.get("disk_io"):
                disk_read_delta = (
                    (
                        final_metrics["disk_io"]["read_bytes"]
                        - initial_metrics["disk_io"]["read_bytes"]
                    )
                    / 1024
                    / 1024
                )  # Convert to MB
                disk_write_delta = (
                    (
                        final_metrics["disk_io"]["write_bytes"]
                        - initial_metrics["disk_io"]["write_bytes"]
                    )
                    / 1024
                    / 1024
                )  # Convert to MB

            # Network I/O delta if available
            net_recv_delta = 0
            net_sent_delta = 0
            if initial_metrics.get("network_io") and final_metrics.get("network_io"):
                net_recv_delta = (
                    (
                        final_metrics["network_io"]["bytes_recv"]
                        - initial_metrics["network_io"]["bytes_recv"]
                    )
                    / 1024
                    / 1024
                )  # Convert to MB
                net_sent_delta = (
                    (
                        final_metrics["network_io"]["bytes_sent"]
                        - initial_metrics["network_io"]["bytes_sent"]
                    )
                    / 1024
                    / 1024
                )  # Convert to MB

            logger.info(
                f"System monitoring completed for {operation_name}",
                extra={
                    "duration_seconds": duration,
                    "system_cpu_delta_percent": cpu_delta,
                    "system_memory_delta_percent": memory_delta,
                    "disk_read_mb": disk_read_delta,
                    "disk_write_mb": disk_write_delta,
                    "network_recv_mb": net_recv_delta,
                    "network_sent_mb": net_sent_delta,
                },
            )

    def get_system_metrics(self) -> dict[str, Any]:
        """Get comprehensive system-wide metrics using psutil.

        Returns:
            Dictionary containing system CPU, memory, disk, and network metrics

        Note:
            Caches results for 1 second to avoid excessive system calls
        """
        current_time = time.time()
        if (
            current_time - self._system_metrics_cache.get("timestamp", 0)
        ) < self._cache_timeout:
            return self._system_metrics_cache.get("data", {})

        try:
            # System CPU metrics
            cpu_times = psutil.cpu_times()
            cpu_stats = psutil.cpu_stats()

            # System memory metrics
            virtual_mem = psutil.virtual_memory()
            swap_mem = psutil.swap_memory()

            # System disk I/O metrics
            try:
                disk_io = psutil.disk_io_counters()
            except RuntimeError:  # No disks on system
                disk_io = None

            # System network I/O metrics
            try:
                net_io = psutil.net_io_counters()
            except RuntimeError:  # No network interfaces
                net_io = None

            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
            except (AttributeError, OSError):  # Not available on Windows
                load_avg = None

            system_metrics = {
                "cpu": {
                    "percent": psutil.cpu_percent(interval=None),
                    "count_logical": psutil.cpu_count(logical=True),
                    "count_physical": psutil.cpu_count(logical=False),
                    "times": cpu_times._asdict(),
                    "stats": cpu_stats._asdict(),
                    "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                },
                "memory": {
                    "virtual": virtual_mem._asdict(),
                    "swap": swap_mem._asdict(),
                },
                "disk_io": disk_io._asdict() if disk_io else None,
                "network_io": net_io._asdict() if net_io else None,
                "load_avg": load_avg,
                "boot_time": psutil.boot_time(),
            }

            # Cache the results
            self._system_metrics_cache = {
                "timestamp": current_time,
                "data": system_metrics,
            }

            return system_metrics

        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
            return {}

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get comprehensive performance metrics summary with enhanced psutil data.

        Returns:
            Dictionary with aggregated performance statistics including
            operation counts, success rates, timing statistics, resource usage,
            and comprehensive system metrics
        """
        if not self.metrics:
            return {"total_operations": 0}

        total_operations = sum(self.operation_counts.values())
        total_errors = sum(self.error_counts.values())

        # Calculate aggregate statistics for all metrics
        all_durations = []
        all_memory_deltas = []
        all_cpu_percents = []
        all_io_read_bytes = []
        all_io_write_bytes = []
        all_ctx_switches = []

        for metric_list in self.metrics.values():
            for metric in metric_list:
                all_durations.append(metric.duration_seconds)
                all_memory_deltas.append(metric.memory_delta_mb)
                if metric.cpu_percent > 0:
                    all_cpu_percents.append(metric.cpu_percent)
                if metric.io_read_bytes > 0:
                    all_io_read_bytes.append(metric.io_read_bytes)
                if metric.io_write_bytes > 0:
                    all_io_write_bytes.append(metric.io_write_bytes)
                total_ctx_switches = (
                    metric.ctx_switches_voluntary + metric.ctx_switches_involuntary
                )
                if total_ctx_switches > 0:
                    all_ctx_switches.append(total_ctx_switches)

        # Get current system metrics for context
        current_system_metrics = self.get_system_metrics()

        return {
            "summary": {
                "total_operations": total_operations,
                "total_errors": total_errors,
                "success_rate": (total_operations - total_errors) / total_operations
                if total_operations > 0
                else 0,
                "total_time_seconds": sum(all_durations),
                "avg_time_seconds": sum(all_durations) / len(all_durations)
                if all_durations
                else 0,
                "min_time_seconds": min(all_durations) if all_durations else 0,
                "max_time_seconds": max(all_durations) if all_durations else 0,
                # Enhanced resource usage statistics
                "total_memory_delta_mb": sum(all_memory_deltas),
                "avg_memory_delta_mb": sum(all_memory_deltas) / len(all_memory_deltas)
                if all_memory_deltas
                else 0,
                "avg_cpu_percent": sum(all_cpu_percents) / len(all_cpu_percents)
                if all_cpu_percents
                else 0,
                "total_io_read_mb": sum(all_io_read_bytes) / 1024 / 1024
                if all_io_read_bytes
                else 0,
                "total_io_write_mb": sum(all_io_write_bytes) / 1024 / 1024
                if all_io_write_bytes
                else 0,
                "avg_ctx_switches": sum(all_ctx_switches) / len(all_ctx_switches)
                if all_ctx_switches
                else 0,
            },
            "operation_counts": self.operation_counts,
            "error_counts": self.error_counts,
            "system_state": {
                "cpu_percent": current_system_metrics.get("cpu", {}).get("percent", 0),
                "memory_percent": current_system_metrics.get("memory", {})
                .get("virtual", {})
                .get("percent", 0),
                "load_avg": current_system_metrics.get("load_avg"),
                "cpu_count": current_system_metrics.get("cpu", {}).get(
                    "count_logical", 0
                ),
                "memory_total_gb": current_system_metrics.get("memory", {})
                .get("virtual", {})
                .get("total", 0)
                / 1024
                / 1024
                / 1024,
            },
            "detailed_metrics": self.metrics,
        }

    def get_report(self) -> dict[str, Any]:
        """Get comprehensive performance report (compatibility method).

        Returns:
            Detailed performance report with all tracked metrics
        """
        return self.get_metrics_summary()

    def log_performance_summary(self) -> None:
        """Log a comprehensive performance summary with enhanced psutil metrics.

        Outputs human-readable summary of all tracked performance metrics
        including operation counts, error rates, timing statistics, and
        comprehensive resource usage analysis.
        """
        report = self.get_metrics_summary()
        summary = report["summary"]
        system_state = report["system_state"]

        logger.info(
            f"Performance Summary - Operations: {summary['total_operations']}, "
            f"Errors: {summary['total_errors']}, "
            f"Success Rate: {summary['success_rate']:.2%}, "
            f"Total Time: {summary['total_time_seconds']:.2f}s, "
            f"Avg Time: {summary['avg_time_seconds']:.2f}s"
        )

        logger.info(
            f"Resource Usage - Memory Delta: {summary['total_memory_delta_mb']:.1f}MB, "
            f"Avg CPU: {summary['avg_cpu_percent']:.1f}%, "
            f"Total I/O: {summary['total_io_read_mb']:.1f}MB read / "
            f"{summary['total_io_write_mb']:.1f}MB write, "
            f"Avg Context Switches: {summary['avg_ctx_switches']:.0f}"
        )

        logger.info(
            f"System State - CPU: {system_state['cpu_percent']:.1f}%, "
            f"Memory: {system_state['memory_percent']:.1f}%, "
            f"Load Avg: {system_state['load_avg']}, "
            f"CPUs: {system_state['cpu_count']}, "
            f"RAM: {system_state['memory_total_gb']:.1f}GB"
        )

        # Log per-operation statistics with enhanced metrics
        for operation, count in self.operation_counts.items():
            error_count = self.error_counts.get(operation, 0)
            success_rate = (count - error_count) / count if count > 0 else 0

            if operation in self.metrics:
                metrics_list = self.metrics[operation]
                durations = [m.duration_seconds for m in metrics_list]
                memory_deltas = [m.memory_delta_mb for m in metrics_list]
                cpu_percents = [
                    m.cpu_percent for m in metrics_list if m.cpu_percent > 0
                ]
                io_operations = [
                    m.io_read_count + m.io_write_count
                    for m in metrics_list
                    if (m.io_read_count + m.io_write_count) > 0
                ]

                avg_duration = sum(durations) / len(durations) if durations else 0
                avg_memory_delta = (
                    sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
                )
                avg_cpu = sum(cpu_percents) / len(cpu_percents) if cpu_percents else 0
                avg_io_ops = (
                    sum(io_operations) / len(io_operations) if io_operations else 0
                )

                logger.info(
                    f"  {operation}: {count} ops, {success_rate:.1%} success, "
                    f"{avg_duration:.3f}s avg, {avg_memory_delta:+.1f}MB mem, "
                    f"{avg_cpu:.0f}% CPU, {avg_io_ops:.0f} I/O ops"
                )

    def clear_metrics(self) -> None:
        """Clear all stored metrics and counters.

        Useful for resetting monitoring state between test runs or
        application phases.
        """
        self.metrics.clear()
        self.operation_counts.clear()
        self.error_counts.clear()


# Global monitor instance
_global_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor instance with enhanced psutil.

    Enhanced with comprehensive psutil capabilities for system monitoring.

    Returns:
        Global PerformanceMonitor instance for application-wide metrics tracking
        with comprehensive system resource monitoring

    Note:
        Uses singleton pattern to ensure consistent metrics across the application.
        All components should use this function to access the performance monitor.
        The monitor includes comprehensive psutil-based tracking for CPU, memory,
        I/O, and system resources.
    """
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


@asynccontextmanager
async def comprehensive_resource_monitor(operation_name: str):
    """Comprehensive resource monitoring context manager.

    Args:
        operation_name: Name of the operation being monitored

    Yields:
        Tuple of (BaseMetrics, dict) - process metrics and system metrics

    Note:
        Combines process-level monitoring with system-wide resource tracking.
        Provides the most complete view of resource usage during operations.
        Uses psutil.Process.oneshot() optimization for efficient data collection.
    """
    monitor = get_performance_monitor()

    # Use both process-level and system-level monitoring
    async with (
        monitor.measure(operation_name) as process_metrics,
        monitor.system_resource_monitor(operation_name) as system_metrics,
    ):
        yield process_metrics, system_metrics


def reset_performance_monitor() -> None:
    """Reset the global performance monitor with enhanced psutil capabilities.

    Creates a new global monitor instance, clearing all existing metrics
    and resetting system metrics cache. Useful for testing and application restarts.
    """
    global _global_monitor
    _global_monitor = PerformanceMonitor()
