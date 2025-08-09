#!/usr/bin/env python
"""Comprehensive test runner with coverage reporting for DocMind AI.

This script provides a comprehensive testing framework with the following features:
- Organized test execution by category (unit, integration, performance)
- Detailed coverage reporting (HTML, JSON, XML, terminal)
- Test failure analysis with detailed reporting
- Performance benchmarking for critical components
- Support for different test environments (fast, full, CI/CD)

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Run only fast unit tests
    python run_tests.py --unit       # Run unit tests only
    python run_tests.py --integration # Run integration tests only
    python run_tests.py --performance # Run performance tests only
    python run_tests.py --coverage    # Generate detailed coverage report
    python run_tests.py --clean       # Clean test artifacts before running
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


class TestResult:
    """Container for test execution results."""

    def __init__(self):
        """Initialize a TestResult instance.

        Tracks comprehensive test execution metrics including pass/fail status,
        timing, command details, and execution outputs.

        Attributes:
            passed (int): Number of tests passed.
            failed (int): Number of tests failed.
            skipped (int): Number of tests skipped.
            errors (int): Number of test errors.
            duration (float): Total test execution time in seconds.
            command (str): Command that was executed.
            output (str): Complete test execution output.
            exit_code (int): Exit code of the test execution.
        """
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors = 0
        self.duration = 0.0
        self.command = ""
        self.output = ""
        self.exit_code = 0


class TestRunner:
    """Comprehensive test runner for DocMind AI."""

    def __init__(self, project_root: Path):
        """Initialize a TestRunner instance.

        Manages comprehensive test execution for a specific project root.

        Args:
            project_root (Path): Absolute path to the project's root directory.

        Attributes:
            project_root (Path): Directory containing the project being tested.
            results (list[TestResult]): Collected test results from various test runs.
        """
        self.project_root = project_root
        self.results: list[TestResult] = []

    def clean_artifacts(self) -> None:
        """Clean test artifacts and caches."""
        print("🧹 Cleaning test artifacts...")

        artifacts_to_clean = [
            ".pytest_cache",
            "htmlcov",
            "coverage.xml",
            "coverage.json",
            ".coverage",
            "__pycache__",
            "*.pyc",
        ]

        for pattern in artifacts_to_clean:
            if pattern.startswith("*."):
                # Use find for file patterns
                subprocess.run(
                    ["find", ".", "-name", pattern, "-delete"],
                    cwd=self.project_root,
                    capture_output=True,
                )
            else:
                # Remove directories
                path = self.project_root / pattern
                if path.exists():
                    if path.is_file():
                        path.unlink()
                    else:
                        subprocess.run(["rm", "-rf", str(path)], cwd=self.project_root)

        print("✅ Artifacts cleaned")

    def run_command(self, command: list[str], description: str) -> TestResult:
        """Run a test command and capture results."""
        result = TestResult()
        result.command = " ".join(command)

        print(f"\n{'=' * 60}")
        print(f"🧪 {description}")
        print(f"📋 Command: {result.command}")
        print("=" * 60)

        start_time = time.time()

        try:
            process = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes timeout
            )

            result.duration = time.time() - start_time
            result.exit_code = process.returncode
            result.output = process.stdout

            if process.stderr:
                result.output += f"\n--- STDERR ---\n{process.stderr}"

            # Parse pytest output for statistics
            if "pytest" in command[0] or "pytest" in command[1]:
                self._parse_pytest_output(result)

            if result.exit_code == 0:
                print(f"✅ {description} completed successfully")
                if result.passed > 0:
                    print(
                        f"   📊 {result.passed} passed, "
                        f"{result.failed} failed, {result.skipped} skipped"
                    )
            else:
                print(f"❌ {description} failed with exit code {result.exit_code}")
                if result.failed > 0:
                    print(
                        f"   📊 {result.passed} passed, "
                        f"{result.failed} failed, {result.skipped} skipped"
                    )

            print(f"   ⏱️  Duration: {result.duration:.2f}s")

        except subprocess.TimeoutExpired:
            result.duration = time.time() - start_time
            result.exit_code = -1
            result.output = "Test execution timed out after 30 minutes"
            print(f"⏰ {description} timed out")

        except Exception as e:
            result.duration = time.time() - start_time
            result.exit_code = -1
            result.output = f"Execution error: {e}"
            print(f"💥 {description} failed with error: {e}")

        self.results.append(result)
        return result

    def _parse_pytest_output(self, result: TestResult) -> None:
        """Parse pytest output to extract test statistics."""
        output = result.output

        # Look for test summary line like "5 passed, 2 failed, 1 skipped"
        import re

        summary_pattern = (
            r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)?"
        )
        match = re.search(summary_pattern, output)

        if match:
            result.passed = int(match.group(1)) if match.group(1) else 0
            result.failed = int(match.group(2)) if match.group(2) else 0
            result.skipped = int(match.group(3)) if match.group(3) else 0
            result.errors = int(match.group(4)) if match.group(4) else 0

        # Alternative parsing for different pytest output formats
        if result.passed == 0 and result.failed == 0:
            passed_match = re.search(r"(\d+) passed", output)
            failed_match = re.search(r"(\d+) failed", output)
            skipped_match = re.search(r"(\d+) skipped", output)
            error_match = re.search(r"(\d+) error", output)

            if passed_match:
                result.passed = int(passed_match.group(1))
            if failed_match:
                result.failed = int(failed_match.group(1))
            if skipped_match:
                result.skipped = int(skipped_match.group(1))
            if error_match:
                result.errors = int(error_match.group(1))

    def run_unit_tests(self) -> TestResult:
        """Run fast unit tests."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/test_models.py",
            "tests/test_utils.py",
            "tests/test_agent_utils.py",
            "-v",
            "--tb=short",
            "--cov=models",
            "--cov=utils",
            "--cov-report=term-missing",
            "--durations=10",
            "-m",
            "not slow and not integration and not requires_gpu",
        ]
        return self.run_command(command, "Unit Tests (Fast)")

    def run_integration_tests(self) -> TestResult:
        """Run integration tests."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/test_integration.py",
            "tests/test_document_loader.py",
            "tests/test_index_builder.py",
            "tests/test_hybrid_search.py",
            "-v",
            "--tb=short",
            "--cov=agent_factory",
            "--cov=utils",
            "--cov-report=term-missing",
            "--durations=10",
            "-m",
            "integration or not slow",
        ]
        return self.run_command(command, "Integration Tests")

    def run_performance_tests(self) -> TestResult:
        """Run performance and benchmark tests."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/test_performance.py",
            "tests/test_performance_integration.py",
            "-v",
            "--tb=short",
            "--benchmark-only",
            "--benchmark-columns=min,max,mean,stddev,rounds,iterations",
            "--benchmark-sort=mean",
            "--durations=10",
            "-m",
            "performance",
        ]
        return self.run_command(command, "Performance Tests")

    def run_slow_tests(self) -> TestResult:
        """Run slow/expensive tests (GPU, network, model downloads)."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/test_embeddings.py",
            "tests/test_gpu_optimization.py",
            "tests/test_multimodal.py",
            "-v",
            "--tb=short",
            "--cov-report=term-missing",
            "--durations=10",
            "-m",
            "slow or requires_gpu or requires_network",
            "--timeout=600",  # 10 minute timeout for slow tests
        ]
        return self.run_command(command, "Slow Tests (GPU/Network/Models)")

    def run_all_tests(self) -> TestResult:
        """Run all tests with comprehensive coverage."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--cov=.",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "--cov-branch",
            "--durations=20",
            "--maxfail=5",  # Stop after 5 failures
        ]
        return self.run_command(command, "All Tests with Coverage")

    def run_smoke_tests(self) -> TestResult:
        """Run basic smoke tests to verify system health."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/test_models.py::test_app_settings_clean_defaults",
            "tests/test_utils.py::test_setup_logging",
            "tests/test_agent_utils.py::test_agent_creation_with_default_settings",
            "-v",
            "--tb=line",
            "-m",
            "not slow",
        ]
        return self.run_command(command, "Smoke Tests")

    def validate_imports(self) -> TestResult:
        """Validate that all modules can be imported."""
        command = [
            "uv",
            "run",
            "python",
            "-c",
            """
import sys
import importlib

modules = [
    'models', 'utils.utils', 'utils.document_loader', 
    'utils.index_builder', 'utils.model_manager',
    'agents.agent_utils', 'agent_factory'
]

failed = []
for module in modules:
    try:
        importlib.import_module(module)
        print(f'✅ {module}')
    except Exception as e:
        print(f'❌ {module}: {e}')
        failed.append(module)

if failed:
    print(f'Failed to import: {failed}')
    sys.exit(1)
else:
    print('All imports successful!')
            """,
        ]
        return self.run_command(command, "Import Validation")

    def generate_coverage_report(self) -> None:
        """Generate detailed coverage analysis."""
        print("\n" + "=" * 60)
        print("📊 COVERAGE ANALYSIS")
        print("=" * 60)

        # Run coverage report command
        coverage_cmd = ["uv", "run", "coverage", "report", "--show-missing"]
        self.run_command(coverage_cmd, "Coverage Report")

        # Generate coverage analysis if coverage.json exists
        coverage_json = self.project_root / "coverage.json"
        if coverage_json.exists():
            self._analyze_coverage_data(coverage_json)

    def _analyze_coverage_data(self, coverage_json: Path) -> None:
        """Analyze coverage data and provide insights."""
        try:
            with coverage_json.open() as f:
                data = json.load(f)

            files = data.get("files", {})
            if not files:
                print("❌ No coverage data found")
                return

            # Calculate overall stats
            total_lines = sum(f["summary"]["num_statements"] for f in files.values())
            covered_lines = sum(f["summary"]["covered_lines"] for f in files.values())
            overall_pct = (covered_lines / total_lines) * 100 if total_lines > 0 else 0

            print(f"\n📈 OVERALL COVERAGE: {overall_pct:.1f}%")
            print(f"   Total lines: {total_lines}")
            print(f"   Covered lines: {covered_lines}")
            print(f"   Missing lines: {total_lines - covered_lines}")

            # Find low coverage files
            low_coverage_files = []
            for filename, file_data in files.items():
                coverage_pct = file_data["summary"]["percent_covered"]
                if coverage_pct < 80 and not any(
                    skip in filename for skip in ["test_", "__pycache__", ".pyc"]
                ):
                    low_coverage_files.append((filename, coverage_pct))

            if low_coverage_files:
                print("\n🔍 FILES WITH LOW COVERAGE (<80%):")
                print("-" * 50)
                for filename, coverage_pct in sorted(
                    low_coverage_files, key=lambda x: x[1]
                ):
                    print(f"   {filename}: {coverage_pct:.1f}%")

            # Identify critical files
            critical_files = [
                "models.py",
                "utils/utils.py",
                "utils/document_loader.py",
                "utils/index_builder.py",
                "agent_factory.py",
                "agents/agent_utils.py",
            ]

            print("\n🎯 CRITICAL FILE COVERAGE:")
            print("-" * 50)
            for critical in critical_files:
                found = False
                for filename, file_data in files.items():
                    if critical in filename:
                        coverage_pct = file_data["summary"]["percent_covered"]
                        status = "✅" if coverage_pct >= 80 else "⚠️"
                        print(f"   {status} {filename}: {coverage_pct:.1f}%")
                        found = True
                        break
                if not found:
                    print(f"   ❓ {critical}: Not found in coverage data")

        except Exception as e:
            print(f"❌ Error analyzing coverage data: {e}")

    def print_summary(self) -> None:
        """Print comprehensive test execution summary."""
        print("\n" + "=" * 80)
        print("📋 TEST EXECUTION SUMMARY")
        print("=" * 80)

        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        print("📊 OVERALL STATISTICS:")
        print(f"   ✅ Passed:  {total_passed}")
        print(f"   ❌ Failed:  {total_failed}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        print(f"   💥 Errors:  {total_errors}")
        print(f"   ⏱️  Total Duration: {total_duration:.1f}s")

        print("\n🔍 DETAILED RESULTS:")
        print("-" * 60)

        for result in self.results:
            status = "✅" if result.exit_code == 0 else "❌"
            print(f"   {status} {result.command}")
            if result.passed > 0 or result.failed > 0:
                print(
                    f"      📊 {result.passed}P {result.failed}F "
                    f"{result.skipped}S - {result.duration:.1f}s"
                )
            else:
                print(f"      ⏱️  {result.duration:.1f}s")

        # Show failures
        failed_results = [r for r in self.results if r.exit_code != 0]
        if failed_results:
            print("\n❌ FAILED TESTS:")
            print("-" * 60)
            for result in failed_results:
                print(f"   Command: {result.command}")
                print(f"   Exit Code: {result.exit_code}")
                if result.output:
                    # Show last few lines of output
                    output_lines = result.output.split("\n")[-20:]
                    print("   Last output lines:")
                    for line in output_lines:
                        if line.strip():
                            print(f"     {line}")
                print()

        # Print recommended actions
        print("\n💡 RECOMMENDATIONS:")
        if total_failed > 0:
            print("   🔧 Fix failing tests before deployment")
        if total_failed == 0 and total_passed > 0:
            print("   🎉 All tests passing! Ready for deployment")

        print("   📄 Coverage report: htmlcov/index.html")
        print("   📊 Coverage data: coverage.json")
        print("   🧪 Re-run specific tests: pytest tests/test_<name>.py -v")


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="DocMind AI Test Runner")
    parser.add_argument("--fast", action="store_true", help="Run only fast unit tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance tests only"
    )
    parser.add_argument(
        "--slow", action="store_true", help="Run slow tests (GPU/Network)"
    )
    parser.add_argument("--smoke", action="store_true", help="Run basic smoke tests")
    parser.add_argument(
        "--coverage", action="store_true", help="Generate detailed coverage report"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean artifacts before running"
    )
    parser.add_argument(
        "--validate-imports", action="store_true", help="Validate module imports"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent
    runner = TestRunner(project_root)

    print("🧪 DocMind AI Test Suite")
    print("=" * 50)

    # Clean artifacts if requested
    if args.clean:
        runner.clean_artifacts()

    try:
        # Run specific test categories
        if args.validate_imports:
            runner.validate_imports()
        elif args.smoke:
            runner.run_smoke_tests()
        elif args.fast or args.unit:
            runner.run_unit_tests()
        elif args.integration:
            runner.run_integration_tests()
        elif args.performance:
            runner.run_performance_tests()
        elif args.slow:
            runner.run_slow_tests()
        else:
            # Run comprehensive test suite
            runner.validate_imports()
            runner.run_unit_tests()
            runner.run_integration_tests()
            # Skip slow tests by default unless specifically requested
            if not args.fast:
                runner.run_slow_tests()

        # Generate coverage report if requested or if running all tests
        if args.coverage or (
            not any(
                [
                    args.fast,
                    args.unit,
                    args.integration,
                    args.performance,
                    args.slow,
                    args.smoke,
                    args.validate_imports,
                ]
            )
        ):
            runner.generate_coverage_report()

    except KeyboardInterrupt:
        print("\n⚠️  Test execution interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)

    # Print summary
    runner.print_summary()

    # Exit with appropriate code
    failed_runs = sum(1 for r in runner.results if r.exit_code != 0)
    if failed_runs > 0:
        print(f"\n❌ {failed_runs} test run(s) failed")
        sys.exit(1)
    else:
        print("\n✅ All test runs completed successfully")


if __name__ == "__main__":
    main()
