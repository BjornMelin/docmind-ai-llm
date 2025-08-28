#!/usr/bin/env python
"""CI/CD Performance Monitor - Based on Phase 5A Exceptional Results.

This script monitors test performance, detects regressions, and enforces
quality gates based on the outstanding Phase 5A achievements:
- 95.4% test success rate (229 passed, 11 failed)
- 29.71% coverage (trending upward from 26.09%)
- <0.1s average unit test performance (excellent)
- 80%+ production readiness achieved

Performance monitoring includes:
- Test execution time tracking and regression detection
- Success rate monitoring with quality gate enforcement
- Coverage trend analysis and regression prevention
- CI/CD pipeline performance optimization
- Automated alerts for performance degradation

Usage:
    python scripts/performance_monitor_ci.py --analyze-latest      # Analyze latest test results
    python scripts/performance_monitor_ci.py --regression-check    # Check for performance regressions
    python scripts/performance_monitor_ci.py --quality-gates       # Enforce quality gates
    python scripts/performance_monitor_ci.py --trend-analysis      # Generate trend analysis report
    python scripts/performance_monitor_ci.py --dashboard          # Generate performance dashboard
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path


class PerformanceMetrics:
    """Container for performance metrics and quality gate status."""

    def __init__(self):
        """Initialize performance metrics."""
        self.timestamp = datetime.utcnow()
        self.success_rate = 0.0
        self.avg_test_time = 0.0
        self.coverage_percent = 0.0
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.regression_detected = False
        self.quality_gates_passed = False


class Phase5ABaseline:
    """Phase 5A exceptional results baseline for comparison."""

    SUCCESS_RATE = 95.4
    COVERAGE_PERCENT = 29.71
    UNIT_TEST_AVG_TIME = 0.1
    PRODUCTION_READINESS = 80.0

    # Quality gate thresholds
    MIN_SUCCESS_RATE = 95.0
    MIN_COVERAGE = 29.0
    MAX_UNIT_TEST_TIME = 0.2
    MAX_REGRESSION_THRESHOLD = 1.5


class PerformanceMonitor:
    """CI/CD Performance Monitor with quality gate enforcement."""

    def __init__(self, project_root: Path):
        """Initialize the performance monitor.

        Args:
            project_root (Path): Absolute path to the project's root directory.
        """
        self.project_root = project_root
        self.db_path = project_root / "performance_metrics.db"
        self.baseline = Phase5ABaseline()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize the performance metrics database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    success_rate REAL NOT NULL,
                    avg_test_time REAL NOT NULL,
                    coverage_percent REAL NOT NULL,
                    total_tests INTEGER NOT NULL,
                    passed_tests INTEGER NOT NULL,
                    failed_tests INTEGER NOT NULL,
                    regression_detected BOOLEAN NOT NULL,
                    quality_gates_passed BOOLEAN NOT NULL,
                    ci_mode TEXT,
                    git_commit TEXT,
                    build_number TEXT
                )
            """)
            conn.commit()

    def analyze_latest_results(self) -> PerformanceMetrics:
        """Analyze the latest test results and extract performance metrics."""
        print("📊 Analyzing latest test results vs Phase 5A baseline")
        print(
            f"🎯 Phase 5A Baseline: {self.baseline.SUCCESS_RATE}% success, {self.baseline.COVERAGE_PERCENT}% coverage, <{self.baseline.UNIT_TEST_AVG_TIME}s unit tests"
        )

        metrics = PerformanceMetrics()

        # Analyze coverage.json if available
        coverage_file = self.project_root / "coverage.json"
        if coverage_file.exists():
            try:
                with coverage_file.open() as f:
                    coverage_data = json.load(f)

                metrics.coverage_percent = coverage_data["totals"]["percent_covered"]
                print(
                    f"📈 Coverage: {metrics.coverage_percent:.1f}% (baseline: {self.baseline.COVERAGE_PERCENT}%)"
                )

            except Exception as e:
                print(f"⚠️  Coverage analysis failed: {e}")

        # Analyze pytest results if available (from pytest-json-report)
        pytest_results = self.project_root / "pytest_results.json"
        if pytest_results.exists():
            try:
                with pytest_results.open() as f:
                    pytest_data = json.load(f)

                summary = pytest_data.get("summary", {})
                metrics.total_tests = summary.get("total", 0)
                metrics.passed_tests = summary.get("passed", 0)
                metrics.failed_tests = summary.get("failed", 0)

                if metrics.total_tests > 0:
                    metrics.success_rate = (
                        metrics.passed_tests / metrics.total_tests
                    ) * 100

                # Calculate average test time
                duration = pytest_data.get("duration", 0)
                if metrics.total_tests > 0 and duration > 0:
                    metrics.avg_test_time = duration / metrics.total_tests

                print(
                    f"📊 Success Rate: {metrics.success_rate:.1f}% (baseline: {self.baseline.SUCCESS_RATE}%)"
                )
                print(
                    f"⚡ Avg Test Time: {metrics.avg_test_time:.3f}s (baseline: <{self.baseline.UNIT_TEST_AVG_TIME}s)"
                )

            except Exception as e:
                print(f"⚠️  Pytest results analysis failed: {e}")

        # Detect regressions
        metrics.regression_detected = self._detect_regressions(metrics)

        # Evaluate quality gates
        metrics.quality_gates_passed = self._evaluate_quality_gates(metrics)

        return metrics

    def _detect_regressions(self, current_metrics: PerformanceMetrics) -> bool:
        """Detect performance regressions compared to baseline and trends."""
        regressions = []

        # Success rate regression
        if (
            current_metrics.success_rate < self.baseline.SUCCESS_RATE * 0.95
        ):  # 5% tolerance
            regressions.append(
                f"Success rate regression: {current_metrics.success_rate:.1f}% < {self.baseline.SUCCESS_RATE * 0.95:.1f}%"
            )

        # Performance regression (unit test time)
        if (
            current_metrics.avg_test_time
            > self.baseline.UNIT_TEST_AVG_TIME * self.baseline.MAX_REGRESSION_THRESHOLD
        ):
            regressions.append(
                f"Performance regression: {current_metrics.avg_test_time:.3f}s > {self.baseline.UNIT_TEST_AVG_TIME * self.baseline.MAX_REGRESSION_THRESHOLD:.3f}s"
            )

        # Coverage regression
        if (
            current_metrics.coverage_percent < self.baseline.COVERAGE_PERCENT * 0.90
        ):  # 10% tolerance
            regressions.append(
                f"Coverage regression: {current_metrics.coverage_percent:.1f}% < {self.baseline.COVERAGE_PERCENT * 0.90:.1f}%"
            )

        if regressions:
            print("\n🚨 PERFORMANCE REGRESSIONS DETECTED:")
            for regression in regressions:
                print(f"   ❌ {regression}")
            return True
        else:
            print("\n✅ No performance regressions detected")
            return False

    def _evaluate_quality_gates(self, metrics: PerformanceMetrics) -> bool:
        """Evaluate quality gates based on Phase 5A standards."""
        gates_passed = []
        gates_failed = []

        # Success rate gate
        if metrics.success_rate >= self.baseline.MIN_SUCCESS_RATE:
            gates_passed.append(
                f"Success Rate: {metrics.success_rate:.1f}% >= {self.baseline.MIN_SUCCESS_RATE}%"
            )
        else:
            gates_failed.append(
                f"Success Rate: {metrics.success_rate:.1f}% < {self.baseline.MIN_SUCCESS_RATE}%"
            )

        # Coverage gate
        if metrics.coverage_percent >= self.baseline.MIN_COVERAGE:
            gates_passed.append(
                f"Coverage: {metrics.coverage_percent:.1f}% >= {self.baseline.MIN_COVERAGE}%"
            )
        else:
            gates_failed.append(
                f"Coverage: {metrics.coverage_percent:.1f}% < {self.baseline.MIN_COVERAGE}%"
            )

        # Performance gate
        if metrics.avg_test_time <= self.baseline.MAX_UNIT_TEST_TIME:
            gates_passed.append(
                f"Performance: {metrics.avg_test_time:.3f}s <= {self.baseline.MAX_UNIT_TEST_TIME}s"
            )
        else:
            gates_failed.append(
                f"Performance: {metrics.avg_test_time:.3f}s > {self.baseline.MAX_UNIT_TEST_TIME}s"
            )

        print("\n🎯 QUALITY GATE EVALUATION:")
        for gate in gates_passed:
            print(f"   ✅ {gate}")
        for gate in gates_failed:
            print(f"   ❌ {gate}")

        return len(gates_failed) == 0

    def store_metrics(
        self,
        metrics: PerformanceMetrics,
        ci_mode: str = None,
        git_commit: str = None,
        build_number: str = None,
    ) -> None:
        """Store performance metrics in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO performance_metrics (
                    timestamp, success_rate, avg_test_time, coverage_percent,
                    total_tests, passed_tests, failed_tests, regression_detected,
                    quality_gates_passed, ci_mode, git_commit, build_number
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    metrics.timestamp.isoformat(),
                    metrics.success_rate,
                    metrics.avg_test_time,
                    metrics.coverage_percent,
                    metrics.total_tests,
                    metrics.passed_tests,
                    metrics.failed_tests,
                    metrics.regression_detected,
                    metrics.quality_gates_passed,
                    ci_mode,
                    git_commit,
                    build_number,
                ),
            )
            conn.commit()

        print(f"📊 Metrics stored for {metrics.timestamp.isoformat()}")

    def generate_trend_analysis(self, days: int = 30) -> dict:
        """Generate performance trend analysis for the last N days."""
        print(f"\n📈 Generating {days}-day performance trend analysis")

        cutoff_date = datetime.utcnow() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM performance_metrics 
                WHERE timestamp >= ? 
                ORDER BY timestamp DESC
            """,
                (cutoff_date.isoformat(),),
            )

            records = cursor.fetchall()

        if not records:
            print("⚠️  No performance data available for trend analysis")
            return {}

        # Calculate trends
        success_rates = [r["success_rate"] for r in records]
        avg_test_times = [r["avg_test_time"] for r in records]
        coverage_percentages = [r["coverage_percent"] for r in records]

        trend_analysis = {
            "period_days": days,
            "total_builds": len(records),
            "success_rate": {
                "current": success_rates[0] if success_rates else 0,
                "average": sum(success_rates) / len(success_rates)
                if success_rates
                else 0,
                "trend": "stable",
                "phase5a_comparison": success_rates[0] - self.baseline.SUCCESS_RATE
                if success_rates
                else 0,
            },
            "performance": {
                "current": avg_test_times[0] if avg_test_times else 0,
                "average": sum(avg_test_times) / len(avg_test_times)
                if avg_test_times
                else 0,
                "trend": "stable",
                "phase5a_comparison": avg_test_times[0]
                - self.baseline.UNIT_TEST_AVG_TIME
                if avg_test_times
                else 0,
            },
            "coverage": {
                "current": coverage_percentages[0] if coverage_percentages else 0,
                "average": sum(coverage_percentages) / len(coverage_percentages)
                if coverage_percentages
                else 0,
                "trend": "stable",
                "phase5a_comparison": coverage_percentages[0]
                - self.baseline.COVERAGE_PERCENT
                if coverage_percentages
                else 0,
            },
            "quality_gates": {
                "pass_rate": sum(1 for r in records if r["quality_gates_passed"])
                / len(records)
                * 100,
                "regression_rate": sum(1 for r in records if r["regression_detected"])
                / len(records)
                * 100,
            },
        }

        # Determine trends
        if len(success_rates) >= 2:
            if success_rates[0] > success_rates[-1]:
                trend_analysis["success_rate"]["trend"] = "improving"
            elif success_rates[0] < success_rates[-1]:
                trend_analysis["success_rate"]["trend"] = "declining"

        if len(avg_test_times) >= 2:
            if avg_test_times[0] < avg_test_times[-1]:
                trend_analysis["performance"]["trend"] = "improving"
            elif avg_test_times[0] > avg_test_times[-1]:
                trend_analysis["performance"]["trend"] = "declining"

        if len(coverage_percentages) >= 2:
            if coverage_percentages[0] > coverage_percentages[-1]:
                trend_analysis["coverage"]["trend"] = "improving"
            elif coverage_percentages[0] < coverage_percentages[-1]:
                trend_analysis["coverage"]["trend"] = "declining"

        return trend_analysis

    def print_trend_report(self, trend_analysis: dict) -> None:
        """Print a formatted trend analysis report."""
        print(f"\n📊 PERFORMANCE TREND ANALYSIS ({trend_analysis['period_days']} days)")
        print("=" * 60)

        print("\n📈 Success Rate Trends:")
        print(f"   Current: {trend_analysis['success_rate']['current']:.1f}%")
        print(f"   Average: {trend_analysis['success_rate']['average']:.1f}%")
        print(f"   Trend: {trend_analysis['success_rate']['trend']}")
        print(
            f"   vs Phase 5A: {trend_analysis['success_rate']['phase5a_comparison']:+.1f}%"
        )

        print("\n⚡ Performance Trends:")
        print(f"   Current: {trend_analysis['performance']['current']:.3f}s")
        print(f"   Average: {trend_analysis['performance']['average']:.3f}s")
        print(f"   Trend: {trend_analysis['performance']['trend']}")
        print(
            f"   vs Phase 5A: {trend_analysis['performance']['phase5a_comparison']:+.3f}s"
        )

        print("\n📊 Coverage Trends:")
        print(f"   Current: {trend_analysis['coverage']['current']:.1f}%")
        print(f"   Average: {trend_analysis['coverage']['average']:.1f}%")
        print(f"   Trend: {trend_analysis['coverage']['trend']}")
        print(
            f"   vs Phase 5A: {trend_analysis['coverage']['phase5a_comparison']:+.1f}%"
        )

        print("\n🎯 Quality Gate Statistics:")
        print(f"   Pass Rate: {trend_analysis['quality_gates']['pass_rate']:.1f}%")
        print(
            f"   Regression Rate: {trend_analysis['quality_gates']['regression_rate']:.1f}%"
        )

    def generate_dashboard(self) -> None:
        """Generate a performance dashboard report."""
        print("\n🚀 DOCMIND AI CI/CD PERFORMANCE DASHBOARD")
        print("=" * 60)

        # Current status
        current_metrics = self.analyze_latest_results()

        print(
            f"\n📊 CURRENT STATUS (as of {current_metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC)"
        )
        print(
            f"   📈 Success Rate: {current_metrics.success_rate:.1f}% (target: {self.baseline.MIN_SUCCESS_RATE}%+)"
        )
        print(
            f"   ⚡ Avg Test Time: {current_metrics.avg_test_time:.3f}s (target: <{self.baseline.MAX_UNIT_TEST_TIME}s)"
        )
        print(
            f"   📊 Coverage: {current_metrics.coverage_percent:.1f}% (target: {self.baseline.MIN_COVERAGE}%+)"
        )
        print(
            f"   🎯 Quality Gates: {'✅ PASSED' if current_metrics.quality_gates_passed else '❌ FAILED'}"
        )
        print(
            f"   🚨 Regressions: {'❌ DETECTED' if current_metrics.regression_detected else '✅ NONE'}"
        )

        # Phase 5A comparison
        print("\n📈 PHASE 5A EXCEPTIONAL RESULTS COMPARISON")
        print(
            f"   Success Rate: {current_metrics.success_rate:.1f}% vs {self.baseline.SUCCESS_RATE}% (Phase 5A)"
        )
        print(
            f"   Unit Test Time: {current_metrics.avg_test_time:.3f}s vs <{self.baseline.UNIT_TEST_AVG_TIME}s (Phase 5A)"
        )
        print(
            f"   Coverage: {current_metrics.coverage_percent:.1f}% vs {self.baseline.COVERAGE_PERCENT}% (Phase 5A)"
        )

        # Generate trend analysis
        trend_analysis = self.generate_trend_analysis(days=30)
        if trend_analysis:
            self.print_trend_report(trend_analysis)

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        if (
            current_metrics.quality_gates_passed
            and not current_metrics.regression_detected
        ):
            print(
                "   🎉 Excellent! All quality gates passed and no regressions detected."
            )
            print(
                "   🚀 System maintaining Phase 5A exceptional performance standards."
            )
        else:
            if not current_metrics.quality_gates_passed:
                print(
                    "   🔧 Address quality gate failures to maintain Phase 5A standards"
                )
            if current_metrics.regression_detected:
                print("   ⚡ Investigate and resolve performance regressions")
            print("   📊 Monitor trends and implement improvements as needed")


def main():
    """Main entry point for CI/CD performance monitoring."""
    parser = argparse.ArgumentParser(
        description="DocMind AI CI/CD Performance Monitor (Phase 5A Optimized)",
        epilog="""Phase 5A Exceptional Results Baseline:
  - Success Rate: 95.4% (229 passed, 11 failed)
  - Coverage: 29.71% (trending upward)
  - Unit Test Performance: <0.1s average
  - Production Readiness: 80%+

Examples:
  python scripts/performance_monitor_ci.py --dashboard           # Performance dashboard
  python scripts/performance_monitor_ci.py --analyze-latest     # Analyze latest results
  python scripts/performance_monitor_ci.py --regression-check   # Check for regressions
  python scripts/performance_monitor_ci.py --trend-analysis     # Trend analysis
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--analyze-latest",
        action="store_true",
        help="Analyze the latest test results and extract performance metrics",
    )
    parser.add_argument(
        "--regression-check",
        action="store_true",
        help="Check for performance regressions against Phase 5A baseline",
    )
    parser.add_argument(
        "--quality-gates",
        action="store_true",
        help="Enforce quality gates based on Phase 5A standards",
    )
    parser.add_argument(
        "--trend-analysis",
        action="store_true",
        help="Generate performance trend analysis report",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Generate comprehensive performance dashboard",
    )
    parser.add_argument(
        "--store-metrics",
        action="store_true",
        help="Store current metrics in the database",
    )
    parser.add_argument(
        "--ci-mode",
        type=str,
        help="CI/CD mode for context (pre-commit, ci, pr-validation, deployment-gate)",
    )
    parser.add_argument(
        "--git-commit",
        type=str,
        help="Git commit hash for tracking",
    )
    parser.add_argument(
        "--build-number",
        type=str,
        help="Build number for tracking",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    monitor = PerformanceMonitor(project_root)

    print("📊 DocMind AI CI/CD Performance Monitor")
    print("=" * 50)
    print(
        f"🎯 Phase 5A Baseline: {Phase5ABaseline.SUCCESS_RATE}% success, {Phase5ABaseline.COVERAGE_PERCENT}% coverage, <{Phase5ABaseline.UNIT_TEST_AVG_TIME}s unit tests"
    )

    try:
        if args.dashboard:
            monitor.generate_dashboard()
        elif args.analyze_latest:
            metrics = monitor.analyze_latest_results()
            if args.store_metrics:
                monitor.store_metrics(
                    metrics, args.ci_mode, args.git_commit, args.build_number
                )
        elif args.regression_check:
            metrics = monitor.analyze_latest_results()
            if metrics.regression_detected:
                print("\n🚨 REGRESSION CHECK FAILED")
                sys.exit(1)
            else:
                print("\n✅ REGRESSION CHECK PASSED")
        elif args.quality_gates:
            metrics = monitor.analyze_latest_results()
            if not metrics.quality_gates_passed:
                print("\n❌ QUALITY GATES FAILED")
                sys.exit(1)
            else:
                print("\n✅ QUALITY GATES PASSED")
        elif args.trend_analysis:
            trend_data = monitor.generate_trend_analysis()
            if trend_data:
                monitor.print_trend_report(trend_data)
        else:
            # Default: comprehensive dashboard
            monitor.generate_dashboard()

    except KeyboardInterrupt:
        print("\n⚠️  Performance monitoring interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

    print("\n🎯 Performance monitoring completed")


if __name__ == "__main__":
    main()
