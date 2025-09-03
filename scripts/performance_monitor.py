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
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
    "embedding_latency",
    "retrieval_latency",
    "llm_inference_time",
]


class PerformanceMonitor:
    """Comprehensive performance monitoring and regression detection."""

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

            # Base pytest command with performance tracking
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "pytest",
                "--durations=10",  # Show 10 slowest tests
                "--tb=short",
                "-q",  # Quiet mode
                *test_args,
            ]

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
            performance_data.update(
                {
                    "total_duration": duration,
                    "exit_code": result.returncode,
                    "timestamp": datetime.now().isoformat(),
                    "command": " ".join(cmd),
                }
            )

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
        except Exception as e:
            self.failures.append(f"Failed to run test suite: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

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
        except Exception as e:
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
                regression_check = self.regression_tracker.check_regression(metric)

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
            regression_report = self.regression_tracker.generate_regression_report()

            return {
                "status": "regressions_found"
                if regressions_found
                else "no_regressions",
                "regressions_detected": regressions_found,
                "metric_results": results,
                "regression_report": regression_report,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
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
                    f"  Total Duration:      "
                    f"{self.results.get('total_duration', 0):.2f}s",
                    f"  Collection Time:     "
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
                        direction_emoji = {
                            "improving": "📈",
                            "stable": "➡️",
                            "degrading": "📉",
                        }.get(trend["trend_direction"], "❓")

                        report_lines.append(
                            f"  {metric}: {direction_emoji} {trend['trend_direction']} "
                            f"({trend['data_points']} samples)"
                        )

                report_lines.append("")

            except Exception as e:
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
                    *[f"  ❌ {failure}" for failure in self.failures],
                    "",
                ]
            )

        if self.warnings:
            report_lines.extend(
                [
                    "PERFORMANCE WARNINGS:",
                    *[f"  ⚠️  {warning}" for warning in self.warnings],
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
                    *[f"  💡 {rec}" for rec in recommendations],
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


def main() -> int:
    """Main entry point for performance monitoring."""
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

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize monitor
    config = {
        "test_suite_timeout": args.timeout,
        "regression_threshold": args.threshold,
    }
    monitor = PerformanceMonitor(config)

    exit_code = 0

    try:
        # Run test collection measurement
        if args.collection_only or args.run_tests:
            collection_data = monitor.measure_test_collection_time()
            monitor.results.update(collection_data)

            if (
                collection_data.get("collection_time", 0)
                > monitor.config["test_collection_timeout"]
            ):
                print(
                    f"⚠️  Test collection time exceeds target: "
                    f"{collection_data['collection_time']:.2f}s"
                )
            else:
                print(
                    f"✅ Test collection: {collection_data['collection_time']:.2f}s "
                    f"({collection_data.get('test_count', 0)} tests)"
                )

        # Run full test suite
        if args.run_tests:
            performance_data = monitor.run_test_suite_performance(args.test_args)
            monitor.results.update(performance_data)

            if performance_data.get("status") in ["timeout", "error"]:
                print(f"❌ Test suite execution failed: {performance_data}")
                exit_code = 2
            else:
                duration = performance_data.get("total_duration", 0)
                test_count = performance_data.get("test_count", 0)
                print(f"✅ Test suite: {duration:.2f}s ({test_count} tests)")

        # Record baseline
        if args.baseline and monitor.results:
            monitor.record_performance_baseline(monitor.results)
            print("📊 Performance baseline recorded")

        # Check regressions
        if args.check_regressions:
            regression_results = monitor.check_performance_regressions(monitor.results)

            if regression_results.get("regressions_detected"):
                print("❌ Performance regressions detected!")
                exit_code = 1
            else:
                print("✅ No performance regressions detected")

        # Generate and display report
        if args.report:
            report = monitor.generate_performance_report(args.days)
            print(report)

        # Save performance data
        if args.save and monitor.results:
            monitor.save_performance_data(monitor.results)

        # Print warnings and failures
        if monitor.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in monitor.warnings:
                print(f"  • {warning}")

        if monitor.failures:
            print("\n❌ FAILURES:")
            for failure in monitor.failures:
                print(f"  • {failure}")
            exit_code = 1

    except Exception as e:
        logger.exception("Unexpected error during performance monitoring")
        print(f"❌ Unexpected error: {e}")
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
