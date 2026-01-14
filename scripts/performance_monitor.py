#!/usr/bin/env python3
"""Performance Regression Monitor for DocMind AI.

This script provides comprehensive performance monitoring and regression detection:
- Test execution time monitoring
- Performance baseline tracking
- Regression alerts and reporting
- CI/CD integration support
- Performance trend analysis

Usage:
    python scripts/performance_monitor.py --run-tests --check-regressions
    python scripts/performance_monitor.py --baseline --threshold 20
    python scripts/performance_monitor.py --report --days 30

Exit codes:
    0: No performance regressions detected
    1: Performance regressions detected
    2: Error running performance monitoring
"""

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore
except (ImportError, OSError):  # pragma: no cover - optional dependency
    psutil = None  # type: ignore
try:
    import resource  # type: ignore
except (ImportError, OSError):  # pragma: no cover - optional dependency
    resource = None  # type: ignore
try:
    import torch  # type: ignore
except (ImportError, OSError):  # pragma: no cover - optional dependency
    torch = None  # type: ignore

# Import existing regression tracker
sys.path.append(str(Path(__file__).parent.parent))
try:
    from tests.performance.performance_regression_tracker import RegressionTracker
except ImportError:
    # Fallback for when run outside of project
    RegressionTracker = None

logger = logging.getLogger(__name__)

# Performance monitoring configuration
PERFORMANCE_CONFIG = {
    "test_suite_timeout": 600,  # 10 minutes max for test suite
    "test_collection_timeout": 15,  # 15 seconds max for test collection
    "regression_threshold": 20,  # 20% increase triggers alert
    "memory_threshold_mb": 500,  # 500MB increase triggers alert
    "baseline_storage": Path("tests/performance/baselines"),
    "reports_storage": Path("tests/performance/reports"),
}

# Critical performance metrics to monitor
CRITICAL_METRICS = [
    "test_suite_duration",
    "test_collection_time",
    "average_test_duration",
    "memory_usage_peak",
    "gpu_vram_peak_mb",
    "embedding_latency",
    "retrieval_latency",
    "llm_inference_time",
]


class PerformanceMonitor:
    """Performance monitoring and regression detection."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize performance monitor.

        Args:
            config: Custom configuration overrides
        """
        self.config = {**PERFORMANCE_CONFIG, **(config or {})}
        self.regression_tracker = RegressionTracker() if RegressionTracker else None
        self.results: dict[str, Any] = {}
        self.failures: list[str] = []
        self.warnings: list[str] = []

        # Ensure directories exist
        self.config["baseline_storage"].mkdir(parents=True, exist_ok=True)
        self.config["reports_storage"].mkdir(parents=True, exist_ok=True)

    def run_test_suite_performance(
        self, test_args: list[str] | None = None
    ) -> dict[str, Any]:
        """Run test suite and measure performance metrics.

        Args:
            test_args: Additional arguments for pytest

        Returns:
            Dictionary with performance measurements
        """
        test_args = test_args or []

        try:
            logger.info("Starting test suite performance measurement...")
            start_time = time.time()

            # Prepare memory tracking (CPU + GPU)
            self._maybe_reset_gpu_peak()

            # Base pytest command with performance tracking
            cmd = self._build_pytest_cmd(test_args)

            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config["test_suite_timeout"],
            )

            end_time = time.time()
            duration = end_time - start_time

            # Parse pytest output for timing information
            performance_data = self._parse_pytest_output(result.stdout, result.stderr)
            # CPU/GPU peaks
            cpu_peak_mb = self._read_cpu_peak_mb()
            gpu_peak_mb = self._read_gpu_peak_mb()

            performance_data.update(
                {
                    "total_duration": duration,
                    "exit_code": result.returncode,
                    "timestamp": datetime.now().isoformat(),
                    "command": " ".join(cmd),
                }
            )
            if cpu_peak_mb is not None:
                performance_data["memory_usage_peak"] = cpu_peak_mb
            if gpu_peak_mb is not None:
                performance_data["gpu_vram_peak_mb"] = gpu_peak_mb

            logger.info("Test suite completed in %.2fs", duration)
            return performance_data

        except subprocess.TimeoutExpired:
            self.failures.append(
                f"Test suite exceeded timeout of {self.config['test_suite_timeout']}s"
            )
            return {
                "status": "timeout",
                "duration": self.config["test_suite_timeout"],
                "timestamp": datetime.now().isoformat(),
            }
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            self.failures.append(f"Failed to run test suite: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    # --- Internal helpers (extracted for clarity) ---
    def _build_pytest_cmd(self, test_args: list[str]) -> list[str]:
        """Build the subprocess command used to invoke pytest.

        Args:
            test_args: Extra CLI args appended to the pytest invocation.

        Returns:
            The argv list suitable for subprocess.run().
        """
        return [
            "uv",
            "run",
            "python",
            "-m",
            "pytest",
            "--durations=10",
            "--tb=short",
            "-q",
            *test_args,
        ]

    def _maybe_reset_gpu_peak(self) -> None:
        """Reset CUDA peak memory stats if a CUDA-capable torch is available."""
        if (
            torch is not None
            and hasattr(torch.cuda, "is_available")
            and torch.cuda.is_available()
        ):
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats()

    def _read_cpu_peak_mb(self) -> float | None:
        """Read a best-effort estimate of peak process memory usage (MB).

        Prefers resource.getrusage() when available; falls back to psutil RSS.
        Note: ru_maxrss units differ across platforms (bytes on macOS, KiB on Linux).

        Returns:
            Peak memory usage in MB, or None if unavailable.
        """
        with contextlib.suppress(Exception):
            if resource is not None:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                peak = getattr(ru, "ru_maxrss", 0)
                return (
                    float(peak) / (1024.0 * 1024.0)
                    if sys.platform == "darwin"
                    else float(peak) / 1024.0
                )
        if psutil is not None:
            with contextlib.suppress(Exception):
                proc = psutil.Process(os.getpid())
                return float(proc.memory_info().rss) / (1024.0 * 1024.0)
        return None

    def _read_gpu_peak_mb(self) -> float | None:
        """Read CUDA peak allocated memory for device 0 (MB) when available.

        Returns:
            Peak allocated VRAM in MB, or None if torch/CUDA is unavailable or the
            query fails.
        """
        try:
            if (
                torch is not None
                and hasattr(torch.cuda, "is_available")
                and torch.cuda.is_available()
            ):
                peak_bytes = torch.cuda.max_memory_allocated(0)
                return float(peak_bytes) / (1024.0 * 1024.0)
        except (RuntimeError, AttributeError, ValueError):
            return None
        return None

    def measure_test_collection_time(self) -> dict[str, Any]:
        """Measure test collection performance.

        Returns:
            Dictionary with collection timing data
        """
        try:
            logger.info("Measuring test collection time...")
            start_time = time.time()

            result = subprocess.run(
                ["uv", "run", "python", "-m", "pytest", "--collect-only", "-q"],
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config["test_collection_timeout"],
            )

            end_time = time.time()
            collection_time = end_time - start_time

            # Count collected tests
            test_count = 0
            for line in result.stdout.split("\n"):
                if "test session starts" in line.lower():
                    continue
                if " collected" in line:
                    with contextlib.suppress(ValueError, IndexError):
                        test_count = int(line.split()[0])

            collection_data = {
                "collection_time": collection_time,
                "test_count": test_count,
                "tests_per_second": test_count / collection_time
                if collection_time > 0
                else 0,
                "exit_code": result.returncode,
                "timestamp": datetime.now().isoformat(),
            }

            logger.info("Collected %d tests in %.2fs", test_count, collection_time)
            return collection_data

        except subprocess.TimeoutExpired:
            self.warnings.append(
                f"Test collection exceeded timeout of "
                f"{self.config['test_collection_timeout']}s"
            )
            return {
                "status": "timeout",
                "collection_time": self.config["test_collection_timeout"],
                "timestamp": datetime.now().isoformat(),
            }
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            self.warnings.append(f"Failed to measure collection time: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def _parse_pytest_output(self, stdout: str, stderr: str) -> dict[str, Any]:
        """Parse pytest output for performance metrics.

        Args:
            stdout: Standard output from pytest
            stderr: Standard error from pytest

        Returns:
            Dictionary with parsed performance data
        """
        data = {
            "slowest_tests": [],
            "test_count": 0,
            "passed_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "warnings_count": 0,
        }

        lines = (stdout + stderr).split("\n")

        # Parse slowest tests section
        in_slowest_section = False
        for line in lines:
            if "slowest durations" in line.lower():
                in_slowest_section = True
                continue
            if in_slowest_section and line.strip() and not line.startswith("="):
                try:
                    # Parse format: "0.12s call     tests/test_file.py::test_function"
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[0].endswith("s"):
                        duration = float(parts[0][:-1])  # Remove 's' suffix
                        test_name = " ".join(parts[2:])
                        data["slowest_tests"].append(
                            {
                                "test": test_name,
                                "duration": duration,
                            }
                        )
                except (ValueError, IndexError):
                    continue
            if "=" in line and in_slowest_section:
                break

        # Parse test results summary
        for line in lines:
            if " passed" in line or " failed" in line or " skipped" in line:
                try:
                    # Parse format like "12 passed, 3 failed, 1 skipped in 45.67s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed" and i > 0:
                            data["passed_count"] = int(parts[i - 1])
                        elif part == "failed" and i > 0:
                            data["failed_count"] = int(parts[i - 1])
                        elif part == "skipped" and i > 0:
                            data["skipped_count"] = int(parts[i - 1])
                        elif part == "warnings" and i > 0:
                            data["warnings_count"] = int(parts[i - 1])
                except (ValueError, IndexError):
                    continue

        data["test_count"] = (
            data["passed_count"] + data["failed_count"] + data["skipped_count"]
        )

        # Calculate average test duration
        if data["slowest_tests"]:
            total_duration = sum(test["duration"] for test in data["slowest_tests"])
            data["average_duration"] = total_duration / len(data["slowest_tests"])

        return data

    def record_performance_baseline(self, performance_data: dict[str, Any]) -> None:
        """Record performance measurements as baseline.

        Args:
            performance_data: Performance measurements to record
        """
        if not self.regression_tracker:
            self.warnings.append(
                "Regression tracker not available, skipping baseline recording"
            )
            return

        try:
            # Record key metrics
            if "total_duration" in performance_data:
                self.regression_tracker.record_performance(
                    "test_suite_duration",
                    performance_data["total_duration"],
                    "seconds",
                    "latency",
                    {"test_count": performance_data.get("test_count", 0)},
                )

            if "collection_time" in performance_data:
                self.regression_tracker.record_performance(
                    "test_collection_time",
                    performance_data["collection_time"],
                    "seconds",
                    "latency",
                )

            if "average_duration" in performance_data:
                self.regression_tracker.record_performance(
                    "average_test_duration",
                    performance_data["average_duration"],
                    "seconds",
                    "latency",
                )

            if "memory_usage_peak" in performance_data:
                self.regression_tracker.record_performance(
                    "memory_usage_peak",
                    performance_data["memory_usage_peak"],
                    "MB",
                    "memory",
                )

            if performance_data.get("gpu_vram_peak_mb"):
                self.regression_tracker.record_performance(
                    "gpu_vram_peak_mb",
                    performance_data["gpu_vram_peak_mb"],
                    "MB",
                    "memory",
                )

            logger.info("Performance baseline recorded successfully")

        except (RuntimeError, ValueError) as e:
            self.warnings.append(f"Failed to record baseline: {e}")

    def check_performance_regressions(
        self, current_data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Check for performance regressions against baseline.

        Args:
            current_data: Current performance measurements (optional)

        Returns:
            Dictionary with regression analysis results
        """
        if not self.regression_tracker:
            return {"status": "error", "message": "Regression tracker not available"}

        try:
            results = {}
            regressions_found = False

            # Check critical metrics
            for metric in CRITICAL_METRICS:
                # Ensure we pass a dict to the regression tracker; fall back sensibly
                if current_data is not None:
                    metric_data = current_data
                elif isinstance(self.results, dict):
                    metric_data = self.results
                elif isinstance(self.results, list) and self.results:
                    # Assume results is a list of dicts; take the latest entry
                    metric_data = self.results[-1]
                else:
                    metric_data = {}

                regression_check = self.regression_tracker.check_regression(
                    metric,
                    current_data=metric_data,
                    threshold_pct=float(self.config.get("regression_threshold", 20)),
                )

                if regression_check.get("regression_detected"):
                    regressions_found = True
                    factor = regression_check.get("regression_factor", 1.0)
                    current_val = regression_check.get("current_value", 0)

                    self.failures.append(
                        f"Performance regression in {metric}: "
                        f"current={current_val:.2f}, factor={factor:.2f}x"
                    )

                results[metric] = regression_check

            # Generate comprehensive regression report
            regression_report = getattr(
                self.regression_tracker,
                "generate_regression_report",
                lambda: {"status": "not_supported"},
            )()

            return {
                "status": "regressions_found"
                if regressions_found
                else "no_regressions",
                "regressions_detected": regressions_found,
                "metric_results": results,
                "regression_report": regression_report,
                "timestamp": datetime.now().isoformat(),
            }

        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            self.failures.append(f"Failed to check regressions: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def generate_performance_report(self, days_back: int = 30) -> str:
        """Generate comprehensive performance report.

        Args:
            days_back: Number of days to include in report

        Returns:
            Formatted performance report string
        """
        report_lines = [
            "=" * 70,
            "DOCMIND AI PERFORMANCE MONITORING REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: {days_back} days",
            "",
        ]

        # Test suite performance summary
        if self.results:
            report_lines.extend(
                [
                    "TEST SUITE PERFORMANCE:",
                    "  Total Duration:      "
                    f"{self.results.get('total_duration', 0):.2f}s",
                    "  Collection Time:     "
                    f"{self.results.get('collection_time', 0):.2f}s",
                    f"  Tests Collected:     {self.results.get('test_count', 0)}",
                    f"  Tests Passed:        {self.results.get('passed_count', 0)}",
                    f"  Tests Failed:        {self.results.get('failed_count', 0)}",
                    f"  Tests Skipped:       {self.results.get('skipped_count', 0)}",
                    "",
                ]
            )

            # Slowest tests
            slowest_tests = self.results.get("slowest_tests", [])
            if slowest_tests:
                report_lines.extend(
                    [
                        "SLOWEST TESTS (Top 5):",
                        *(
                            (
                                "  "
                                + test["test"][:60]
                                + ("..." if len(test["test"]) > 60 else "")
                                + f": {test['duration']:.2f}s"
                            )
                            for test in slowest_tests[:5]
                        ),
                        "",
                    ]
                )

        # Performance trends (if regression tracker available)
        if self.regression_tracker:
            try:
                for metric in CRITICAL_METRICS[:5]:  # Top 5 critical metrics
                    trend = self.regression_tracker.get_trend_analysis(
                        metric, days_back
                    )
                    if "error" not in trend:
                        direction_marker = {
                            "improving": "UP",
                            "stable": "STABLE",
                            "degrading": "DOWN",
                        }.get(trend["trend_direction"], "UNKNOWN")

                        report_lines.append(
                            f"  {metric}: {direction_marker} "
                            f"{trend['trend_direction']} "
                            f"({trend['data_points']} samples)"
                        )

                report_lines.append("")

            except (KeyError, ValueError, TypeError) as e:
                report_lines.extend(
                    [
                        f"  Error analyzing trends: {e}",
                        "",
                    ]
                )

        # Failures and warnings
        if self.failures:
            report_lines.extend(
                [
                    "PERFORMANCE ISSUES:",
                    *[f"  - {failure}" for failure in self.failures],
                    "",
                ]
            )

        if self.warnings:
            report_lines.extend(
                [
                    "PERFORMANCE WARNINGS:",
                    *[f"  - {warning}" for warning in self.warnings],
                    "",
                ]
            )

        # Recommendations
        recommendations = []

        if (
            self.results.get("collection_time", 0)
            > self.config["test_collection_timeout"] / 2
        ):
            recommendations.append("Consider optimizing test discovery/collection")

        if self.results.get("total_duration", 0) > 300:  # 5 minutes
            recommendations.append(
                "Test suite duration is high, consider parallelization"
            )

        if self.results.get("failed_count", 0) > 0:
            recommendations.append("Address test failures to improve suite reliability")

        if recommendations:
            report_lines.extend(
                [
                    "RECOMMENDATIONS:",
                    *[f"  - {rec}" for rec in recommendations],
                    "",
                ]
            )

        report_lines.append("=" * 70)
        return "\n".join(report_lines)

    def save_performance_data(
        self, data: dict[str, Any], filename: str | None = None
    ) -> None:
        """Save performance data to file.

        Args:
            data: Performance data to save
            filename: Custom filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_data_{timestamp}.json"

        filepath = self.config["reports_storage"] / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info("Performance data saved to %s", filepath)

        except (OSError, TypeError) as e:
            self.warnings.append(f"Failed to save performance data: {e}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the performance monitor.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Monitor performance and detect regressions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run test suite and measure performance",
    )
    parser.add_argument(
        "--collection-only",
        action="store_true",
        help="Measure test collection time only",
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Record current performance as baseline"
    )
    parser.add_argument(
        "--check-regressions",
        action="store_true",
        help="Check for performance regressions",
    )
    parser.add_argument(
        "--report", action="store_true", help="Generate performance report"
    )
    parser.add_argument(
        "--threshold", type=float, default=20.0, help="Regression threshold percentage"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Number of days for trend analysis"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Test suite timeout in seconds"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save performance data to file"
    )
    parser.add_argument(
        "--test-args", nargs="*", help="Additional arguments for pytest"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for CLI usage.

    Args:
        verbose: When True, set DEBUG level; otherwise INFO.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def _init_monitor(args: argparse.Namespace) -> PerformanceMonitor:
    """Initialize a PerformanceMonitor from parsed CLI args.

    Args:
        args: Parsed command-line arguments.

    Returns:
        An initialized PerformanceMonitor instance.
    """
    config = {
        "test_suite_timeout": args.timeout,
        "regression_threshold": args.threshold,
    }
    return PerformanceMonitor(config)


def _handle_collection(monitor: PerformanceMonitor) -> None:
    """Measure and report pytest collection performance.

    Args:
        monitor: Performance monitor instance to update and use for reporting.
    """
    collection_data = monitor.measure_test_collection_time()
    monitor.results.update(collection_data)

    # Handle error or timeout status early
    status = collection_data.get("status")
    if status == "error":
        error_msg = collection_data.get("error", "Unknown error")
        print(f"FAIL: Test collection failed: {error_msg}")
        return
    if status == "timeout":
        collection_time = collection_data.get("collection_time", 0)
        print(f"FAIL: Test collection timed out: {collection_time:.2f}s")
        return

    collection_time = collection_data.get("collection_time", 0)
    if collection_time >= monitor.config["test_collection_timeout"]:
        print(f"WARN: Test collection time exceeds target: {collection_time:.2f}s")
        return
    print(
        f"OK: Test collection: {collection_time:.2f}s "
        f"({collection_data.get('test_count', 0)} tests)"
    )


def _handle_test_run(monitor: PerformanceMonitor, test_args: list[str] | None) -> int:
    """Run the test suite and update monitor results.

    Args:
        monitor: Performance monitor instance to update.
        test_args: Optional pytest args to pass through.

    Returns:
        Process-style exit code: 0 for success, 2 for execution error/timeout.
    """
    performance_data = monitor.run_test_suite_performance(test_args)
    monitor.results.update(performance_data)
    if performance_data.get("status") in ["timeout", "error"]:
        print(f"FAIL: Test suite execution failed: {performance_data}")
        return 2
    duration = performance_data.get("total_duration", 0)
    test_count = performance_data.get("test_count", 0)
    print(f"OK: Test suite: {duration:.2f}s ({test_count} tests)")
    return 0


def _handle_baseline(monitor: PerformanceMonitor) -> None:
    """Record current monitor results as a regression baseline when available."""
    if monitor.results:
        monitor.record_performance_baseline(monitor.results)
        print("OK: Performance baseline recorded")


def _handle_regressions(monitor: PerformanceMonitor) -> int:
    """Check current results against baselines and print a summary.

    Args:
        monitor: Performance monitor instance containing current results.

    Returns:
        Exit code: 0 if no regressions, 1 if regressions detected.
    """
    regression_results = monitor.check_performance_regressions(monitor.results)
    if regression_results.get("regressions_detected"):
        print("FAIL: Performance regressions detected!")
        return 1
    print("OK: No performance regressions detected")
    return 0


def _handle_report(monitor: PerformanceMonitor, days: int) -> None:
    """Generate and print a human-readable performance report.

    Args:
        monitor: Performance monitor instance with accumulated results.
        days: Number of days to include for trend analysis.
    """
    report = monitor.generate_performance_report(days)
    print(report)


def _handle_save(monitor: PerformanceMonitor) -> None:
    """Persist current results to a JSON report file when available."""
    if monitor.results:
        monitor.save_performance_data(monitor.results)


def _print_warnings_and_failures(monitor: PerformanceMonitor) -> int:
    """Print accumulated warnings/failures and compute an exit code.

    Args:
        monitor: Performance monitor instance with warnings/failures populated.

    Returns:
        1 when failures exist; otherwise 0.
    """
    exit_code = 0
    if monitor.warnings:
        print("\nWARNINGS:")
        for warning in monitor.warnings:
            print(f"  - {warning}")
    if monitor.failures:
        print("\nFAILURES:")
        for failure in monitor.failures:
            print(f"  - {failure}")
        exit_code = 1
    return exit_code


def main() -> int:
    """Main entry point for performance monitoring."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    monitor = _init_monitor(args)

    exit_code = 0
    try:
        if args.collection_only or args.run_tests:
            _handle_collection(monitor)
        if args.run_tests:
            exit_code = max(exit_code, _handle_test_run(monitor, args.test_args))
        if args.baseline:
            _handle_baseline(monitor)
        if args.check_regressions:
            exit_code = max(exit_code, _handle_regressions(monitor))
        if args.report:
            _handle_report(monitor, args.days)
        if args.save:
            _handle_save(monitor)
        exit_code = max(exit_code, _print_warnings_and_failures(monitor))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Unexpected error during performance monitoring")
        print(f"ERROR: Unexpected error: {e}")
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
