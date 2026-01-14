#!/usr/bin/env python3
"""Test Suite Health Monitor for DocMind AI.

This module provides comprehensive test suite health monitoring:
- Flaky test detection and tracking
- Test anti-pattern identification
- Test execution stability analysis
- Test duration monitoring
- Health reports and recommendations

Usage:
    python scripts/test_health.py --analyze --runs 10
    python scripts/test_health.py --report --days 7
    python scripts/test_health.py --check-patterns

Exit codes:
    0: Test suite health is good
    1: Health issues detected
    2: Error during health monitoring
"""

import argparse
import json
import logging
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Test health configuration
HEALTH_CONFIG = {
    "flaky_threshold": 0.8,  # Tests must pass 80% of runs to not be flaky
    "min_runs_for_flaky": 5,  # Minimum runs to determine flakiness
    "slow_test_threshold": 30.0,  # Tests taking >30s are considered slow
    "very_slow_threshold": 60.0,  # Tests taking >60s are very slow
    "collection_timeout": 30,  # Test collection timeout
    "single_test_timeout": 300,  # Individual test timeout
    "health_storage": Path("tests/health"),
    "reports_storage": Path("tests/health/reports"),
}

# Test anti-patterns to detect
ANTI_PATTERNS = {
    "sleep_usage": {
        "pattern": r"time\.sleep\(|sleep\(",
        "message": "Use of sleep() in tests - consider mocking or fixtures",
        "severity": "medium",
    },
    "hardcoded_paths": {
        "pattern": r'["\'][/\\].*[/\\]["\']|["\']C:\\.*["\']',
        "message": "Hardcoded file paths - use Path() or fixtures",
        "severity": "medium",
    },
    "print_statements": {
        "pattern": r"print\(.*\)",
        "message": "Print statements in tests - use logging or caplog",
        "severity": "low",
    },
    "bare_except": {
        "pattern": r"except\s*:",
        "message": "Bare except clause - catch specific exceptions",
        "severity": "high",
    },
    "todo_fixme": {
        "pattern": r"#\s*(TODO|FIXME|XXX)",
        "message": "TODO/FIXME comments in test code",
        "severity": "low",
    },
    "long_test_names": {
        "pattern": r"def test_[a-zA-Z_]{80,}",
        "message": "Very long test name - consider shorter, descriptive names",
        "severity": "low",
    },
    "missing_docstrings": {
        "pattern": r'def test_[^:]+:\s*\n\s*[^"""\']{3}',
        "message": "Test missing docstring",
        "severity": "low",
    },
}


class TestHealthMonitor:
    """Comprehensive test suite health monitoring and analysis."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize test health monitor.

        Args:
            config: Custom configuration overrides
        """
        self.config = {**HEALTH_CONFIG, **(config or {})}
        self.health_data: dict[str, Any] = {}
        self.failures: list[str] = []
        self.warnings: list[str] = []

        # Ensure directories exist
        self.config["health_storage"].mkdir(parents=True, exist_ok=True)
        self.config["reports_storage"].mkdir(parents=True, exist_ok=True)

    def run_flakiness_analysis(
        self, runs: int = 10, test_pattern: str | None = None
    ) -> dict[str, Any]:
        """Run multiple test executions to detect flaky tests.

        Args:
            runs: Number of test runs to execute
            test_pattern: Optional pattern to filter specific tests

        Returns:
            Dictionary with flakiness analysis results
        """
        logger.info("Running flakiness analysis with %d iterations...", runs)

        test_results: dict[str, list[bool]] = defaultdict(list)
        execution_times: dict[str, list[float]] = defaultdict(list)

        for run in range(runs):
            logger.info("Flakiness run %d/%d", run + 1, runs)

            try:
                cmd = [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "pytest",
                    "--tb=no",
                    "-v",
                    "--durations=0",
                ]

                if test_pattern:
                    cmd.extend(["-k", test_pattern])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=self.config["single_test_timeout"],
                )

                # Parse individual test results
                self._parse_test_results(result.stdout, test_results, execution_times)

            except subprocess.TimeoutExpired:
                self.warnings.append(f"Test run {run + 1} exceeded timeout")
            except (OSError, ValueError) as e:
                self.warnings.append(f"Error in test run {run + 1}: {e}")

        return self._analyze_flakiness(test_results, execution_times, runs)

    def _parse_test_results(
        self, output: str, results: dict[str, list[bool]], times: dict[str, list[float]]
    ) -> None:
        """Parse pytest output for individual test results.

        Args:
            output: Pytest output
            results: Dictionary to store test results
            times: Dictionary to store execution times
        """
        lines = output.split("\n")

        for line in lines:
            # Parse test results: "tests/test_file.py::test_name PASSED [100%]"
            if "::" in line and any(
                status in line for status in ["PASSED", "FAILED", "ERROR"]
            ):
                try:
                    parts = line.split()
                    test_name = parts[0]
                    status = None

                    for part in parts:
                        if part in ["PASSED", "FAILED", "ERROR", "SKIPPED"]:
                            status = part
                            break

                    if status and test_name:
                        results[test_name].append(status == "PASSED")

                except (ValueError, IndexError):
                    continue

            # Parse durations: "0.12s call     tests/test_file.py::test_function"
            elif "s call" in line and "::" in line:
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3 and parts[0].endswith("s"):
                        duration = float(parts[0][:-1])
                        test_name = parts[2]
                        times[test_name].append(duration)
                except (ValueError, IndexError):
                    continue

    def _analyze_flakiness(
        self,
        results: dict[str, list[bool]],
        times: dict[str, list[float]],
        total_runs: int,
    ) -> dict[str, Any]:
        """Analyze test results for flakiness.

        Args:
            results: Test pass/fail results
            times: Test execution times
            total_runs: Total number of runs

        Returns:
            Flakiness analysis results
        """
        flaky_tests = []
        stable_tests = []
        slow_tests = []

        for test_name, test_results in results.items():
            if len(test_results) < self.config["min_runs_for_flaky"]:
                continue

            pass_count = sum(test_results)
            pass_rate = pass_count / len(test_results)

            test_times = times.get(test_name, [])
            avg_time = sum(test_times) / len(test_times) if test_times else 0.0
            max_time = max(test_times) if test_times else 0.0

            test_analysis = {
                "test_name": test_name,
                "pass_rate": pass_rate,
                "pass_count": pass_count,
                "total_runs": len(test_results),
                "avg_duration": avg_time,
                "max_duration": max_time,
                "duration_variance": self._calculate_variance(test_times)
                if len(test_times) > 1
                else 0.0,
            }

            # Categorize tests
            if pass_rate < self.config["flaky_threshold"]:
                flaky_tests.append(test_analysis)
            else:
                stable_tests.append(test_analysis)

            if avg_time > self.config["slow_test_threshold"]:
                slow_tests.append(test_analysis)

        # Sort by various criteria
        flaky_tests.sort(key=lambda x: x["pass_rate"])
        slow_tests.sort(key=lambda x: x["avg_duration"], reverse=True)

        return {
            "total_tests_analyzed": len(results),
            "total_runs": total_runs,
            "flaky_tests": flaky_tests,
            "stable_tests": len(stable_tests),
            "slow_tests": slow_tests,
            "flakiness_threshold": self.config["flaky_threshold"],
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def _calculate_variance(self, values: list[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / (len(values) - 1)

    def analyze_test_patterns(
        self, test_dirs: list[str] | None = None
    ) -> dict[str, Any]:
        """Analyze test code for anti-patterns.

        Args:
            test_dirs: Directories to analyze (defaults to tests/)

        Returns:
            Pattern analysis results
        """
        test_dirs = test_dirs or ["tests/"]

        logger.info("Analyzing test patterns...")

        pattern_violations = defaultdict(list)
        files_analyzed = 0
        total_violations = 0

        for test_dir in test_dirs:
            test_path = Path(test_dir)
            if not test_path.exists():
                self.warnings.append(f"Test directory not found: {test_dir}")
                continue

            for test_file in test_path.rglob("test_*.py"):
                files_analyzed += 1
                violations = self._analyze_file_patterns(test_file)

                for violation in violations:
                    pattern_violations[violation["pattern"]].append(violation)
                    total_violations += 1

        # Summarize violations by pattern
        pattern_summary = {}
        for pattern_name, violations in pattern_violations.items():
            pattern_config = ANTI_PATTERNS.get(pattern_name, {})
            pattern_summary[pattern_name] = {
                "count": len(violations),
                "severity": pattern_config.get("severity", "unknown"),
                "message": pattern_config.get("message", "No description"),
                "files_affected": len({v["file"] for v in violations}),
                "violations": violations,
            }

        return {
            "files_analyzed": files_analyzed,
            "total_violations": total_violations,
            "patterns_found": len(pattern_violations),
            "pattern_summary": pattern_summary,
            "analysis_timestamp": datetime.now().isoformat(),
        }

    def _analyze_file_patterns(self, file_path: Path) -> list[dict[str, Any]]:
        """Analyze a single file for anti-patterns.

        Args:
            file_path: Path to the test file

        Returns:
            List of pattern violations found
        """
        violations = []

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                lines = content.split("\n")

            for pattern_name, pattern_config in ANTI_PATTERNS.items():
                pattern = pattern_config["pattern"]

                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        violations.append(
                            {
                                "pattern": pattern_name,
                                "file": str(file_path),
                                "line": line_num,
                                "content": line.strip(),
                                "severity": pattern_config["severity"],
                                "message": pattern_config["message"],
                            }
                        )

        except (OSError, UnicodeDecodeError) as e:
            self.warnings.append(f"Error analyzing {file_path}: {e}")

        return violations

    def check_test_stability(self, days_back: int = 7) -> dict[str, Any]:
        """Check test execution stability over time.

        Args:
            days_back: Number of days to analyze

        Returns:
            Test stability analysis
        """
        logger.info("Checking test stability for last %d days...", days_back)

        # This would typically load from historical test data
        # For now, we'll run a stability check based on current execution

        try:
            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "pytest",
                "--tb=short",
                "-v",
                "--durations=10",
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.config["single_test_timeout"],
            )

            stability_data = {
                "exit_code": result.returncode,
                "execution_successful": result.returncode == 0,
                "timestamp": datetime.now().isoformat(),
            }

            # Parse output for test counts and timing
            output_lines = result.stdout.split("\n")
            for line in output_lines:
                if " passed" in line or " failed" in line:
                    # Extract test counts from summary line
                    try:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "passed" and i > 0:
                                stability_data["tests_passed"] = int(parts[i - 1])
                            elif part == "failed" and i > 0:
                                stability_data["tests_failed"] = int(parts[i - 1])
                            elif part == "skipped" and i > 0:
                                stability_data["tests_skipped"] = int(parts[i - 1])
                    except (ValueError, IndexError):
                        continue

            # Calculate stability score
            total_tests = stability_data.get("tests_passed", 0) + stability_data.get(
                "tests_failed", 0
            )
            if total_tests > 0:
                stability_data["pass_rate"] = (
                    stability_data.get("tests_passed", 0) / total_tests
                )
            else:
                stability_data["pass_rate"] = 0.0

            return stability_data

        except subprocess.TimeoutExpired:
            self.failures.append("Test stability check timed out")
            return {"status": "timeout", "timestamp": datetime.now().isoformat()}
        except (OSError, ValueError) as e:
            self.failures.append(f"Error checking test stability: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    def generate_health_report(self) -> str:
        """Generate comprehensive test health report.

        Returns:
            Formatted health report string
        """
        report_lines = [
            "=" * 70,
            "DOCMIND AI TEST SUITE HEALTH REPORT",
            "=" * 70,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        # Flakiness analysis
        if "flakiness" in self.health_data:
            flaky_data = self.health_data["flakiness"]
            flaky_count = len(flaky_data.get("flaky_tests", []))
            total_tested = flaky_data.get("total_tests_analyzed", 0)

            report_lines.extend(
                [
                    "FLAKINESS ANALYSIS:",
                    f"  Tests Analyzed:      {total_tested}",
                    f"  Flaky Tests:         {flaky_count}",
                    "  Stability Rate:      "
                    f"{(1 - flaky_count / total_tested) * 100:.1f}%"
                    if total_tested > 0
                    else "  Stability Rate:      N/A",
                    f"  Analysis Runs:       {flaky_data.get('total_runs', 0)}",
                    "",
                ]
            )

            # Top flaky tests
            flaky_tests = flaky_data.get("flaky_tests", [])
            if flaky_tests:
                report_lines.extend(
                    [
                        "MOST FLAKY TESTS:",
                        *(
                            f"  {test['test_name']}: {test['pass_rate']:.1%} pass rate "
                            f"({test['pass_count']}/{test['total_runs']} runs)"
                            for test in flaky_tests[:5]
                        ),
                        "",
                    ]
                )

        # Pattern analysis
        if "patterns" in self.health_data:
            pattern_data = self.health_data["patterns"]
            total_violations = pattern_data.get("total_violations", 0)
            files_analyzed = pattern_data.get("files_analyzed", 0)

            report_lines.extend(
                [
                    "CODE PATTERN ANALYSIS:",
                    f"  Files Analyzed:      {files_analyzed}",
                    f"  Total Violations:    {total_violations}",
                    f"  Patterns Found:      {pattern_data.get('patterns_found', 0)}",
                    "",
                ]
            )

            # Pattern violations by severity
            pattern_summary = pattern_data.get("pattern_summary", {})
            severity_counts = {"high": 0, "medium": 0, "low": 0}

            for pattern_name, pattern_info in pattern_summary.items():
                severity = pattern_info.get("severity", "unknown")
                count = pattern_info.get("count", 0)
                if severity in severity_counts:
                    severity_counts[severity] += count

                report_lines.append(
                    f"  {pattern_name}: {count} violations "
                    f"({pattern_info.get('files_affected', 0)} files)"
                )

            report_lines.extend(
                [
                    "",
                    "VIOLATIONS BY SEVERITY:",
                    f"  High:    {severity_counts['high']}",
                    f"  Medium:  {severity_counts['medium']}",
                    f"  Low:     {severity_counts['low']}",
                    "",
                ]
            )

        # Test stability
        if "stability" in self.health_data:
            stability_data = self.health_data["stability"]

            exec_status = (
                "SUCCESS" if stability_data.get("execution_successful") else "FAILED"
            )
            report_lines.extend(
                [
                    "TEST SUITE STABILITY:",
                    f"  Execution Status:    {exec_status}",
                    f"  Pass Rate:           {stability_data.get('pass_rate', 0):.1%}",
                    f"  Tests Passed:        {stability_data.get('tests_passed', 0)}",
                    f"  Tests Failed:        {stability_data.get('tests_failed', 0)}",
                    f"  Tests Skipped:       {stability_data.get('tests_skipped', 0)}",
                    "",
                ]
            )

        # Health issues
        if self.failures:
            report_lines.extend(
                [
                    "HEALTH ISSUES:",
                    *[f"  - {failure}" for failure in self.failures],
                    "",
                ]
            )

        if self.warnings:
            report_lines.extend(
                [
                    "HEALTH WARNINGS:",
                    *[f"  - {warning}" for warning in self.warnings],
                    "",
                ]
            )

        # Recommendations
        recommendations = self._generate_recommendations()
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

    def _generate_recommendations(self) -> list[str]:
        """Generate health recommendations based on analysis.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Flakiness recommendations
        if "flakiness" in self.health_data:
            flaky_tests = self.health_data["flakiness"].get("flaky_tests", [])
            if flaky_tests:
                recommendations.append(
                    f"Address {len(flaky_tests)} flaky tests to improve "
                    "suite reliability"
                )

                # Recommend looking at specific flaky tests
                if any(test["pass_rate"] < 0.5 for test in flaky_tests):
                    recommendations.append(
                        "Priority: Fix tests with <50% pass rate first"
                    )

        # Pattern recommendations
        if "patterns" in self.health_data:
            pattern_summary = self.health_data["patterns"].get("pattern_summary", {})

            high_severity = sum(
                info["count"]
                for info in pattern_summary.values()
                if info.get("severity") == "high"
            )
            if high_severity > 0:
                recommendations.append(
                    f"Fix {high_severity} high-severity code patterns"
                )

            if "sleep_usage" in pattern_summary:
                recommendations.append(
                    "Replace time.sleep() with proper mocking/fixtures"
                )

            if "hardcoded_paths" in pattern_summary:
                recommendations.append(
                    "Use pathlib.Path() instead of hardcoded file paths"
                )

        # Stability recommendations
        if "stability" in self.health_data:
            stability = self.health_data["stability"]
            if not stability.get("execution_successful"):
                recommendations.append("Fix test failures to improve stability")

            pass_rate = stability.get("pass_rate", 1.0)
            if pass_rate < 0.95:
                recommendations.append(
                    "Improve test pass rate to >95% for better stability"
                )

        return recommendations

    def save_health_data(
        self, data: dict[str, Any], filename: str | None = None
    ) -> None:
        """Save health analysis data to file.

        Args:
            data: Health data to save
            filename: Custom filename (optional)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_analysis_{timestamp}.json"

        filepath = self.config["reports_storage"] / filename

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info("Health data saved to %s", filepath)
        except (OSError, TypeError) as e:
            self.warnings.append(f"Failed to save health data: {e}")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for test health checks."""
    parser = argparse.ArgumentParser(
        description="Monitor test suite health and detect issues",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Run comprehensive health analysis"
    )
    parser.add_argument(
        "--flakiness", action="store_true", help="Run flakiness analysis only"
    )
    parser.add_argument(
        "--patterns", action="store_true", help="Run pattern analysis only"
    )
    parser.add_argument(
        "--stability", action="store_true", help="Check test stability only"
    )
    parser.add_argument(
        "--runs", type=int, default=5, help="Number of runs for flakiness analysis"
    )
    parser.add_argument(
        "--test-pattern", help="Pattern to filter tests for flakiness analysis"
    )
    parser.add_argument(
        "--test-dirs",
        nargs="+",
        default=["tests/"],
        help="Directories to analyze for patterns",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Days to look back for stability analysis"
    )
    parser.add_argument("--report", action="store_true", help="Generate health report")
    parser.add_argument("--save", action="store_true", help="Save health data to file")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure logging verbosity for the CLI."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )


def _run_flakiness(monitor: TestHealthMonitor, args: argparse.Namespace) -> int:
    """Run flakiness analysis and return exit code."""
    print("Running flakiness analysis...")
    flakiness_data = monitor.run_flakiness_analysis(args.runs, args.test_pattern)
    monitor.health_data["flakiness"] = flakiness_data
    flaky_count = len(flakiness_data.get("flaky_tests", []))
    if flaky_count > 0:
        print(f"WARN: Found {flaky_count} flaky tests")
        return 1
    print("OK: No flaky tests detected")
    return 0


def _run_patterns(monitor: TestHealthMonitor, args: argparse.Namespace) -> int:
    """Run test pattern analysis and return exit code."""
    print("Running pattern analysis...")
    pattern_data = monitor.analyze_test_patterns(args.test_dirs)
    monitor.health_data["patterns"] = pattern_data
    high_violations = sum(
        info["count"]
        for info in pattern_data.get("pattern_summary", {}).values()
        if info.get("severity") == "high"
    )
    if high_violations > 0:
        print(f"WARN: Found {high_violations} high-severity pattern violations")
        return 1
    total_violations = pattern_data.get("total_violations", 0)
    print(f"OK: Pattern analysis complete ({total_violations} total violations)")
    return 0


def _run_stability(monitor: TestHealthMonitor, args: argparse.Namespace) -> int:
    """Run stability checks and return exit code."""
    print("Checking test stability...")
    stability_data = monitor.check_test_stability(args.days)
    monitor.health_data["stability"] = stability_data
    if not stability_data.get("execution_successful"):
        print("FAIL: Test suite execution failed")
        return 1
    pass_rate = stability_data.get("pass_rate", 0)
    print(f"OK: Test stability check complete ({pass_rate:.1%} pass rate)")
    return 0


def _print_warnings_and_failures(monitor: TestHealthMonitor) -> int:
    """Print warnings/failures and return an exit code."""
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
    """Main entry point for test health monitoring."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)

    monitor = TestHealthMonitor()
    exit_code = 0

    try:
        if args.analyze or args.flakiness:
            exit_code = max(exit_code, _run_flakiness(monitor, args))
        if args.analyze or args.patterns:
            exit_code = max(exit_code, _run_patterns(monitor, args))
        if args.analyze or args.stability:
            exit_code = max(exit_code, _run_stability(monitor, args))
        if args.report or args.analyze:
            report = monitor.generate_health_report()
            print("\n" + report)
        if args.save and monitor.health_data:
            monitor.save_health_data(monitor.health_data)
        exit_code = max(exit_code, _print_warnings_and_failures(monitor))
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Unexpected error during health monitoring")
        print(f"ERROR: Unexpected error: {e}")
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
