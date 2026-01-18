#!/usr/bin/env python
"""Comprehensive test runner with tiered testing strategy for DocMind AI.

This script implements a two-tier testing strategy based on ML engineering
best practices:

Tier 1 - Unit Tests (Fast):
    - Mocked dependencies, no external services
    - <5 seconds per test, total suite <30 seconds
    - Run on every code change

Tier 2 - Integration Tests:
    - Lightweight models, minimal GPU usage
    - <30 seconds per test, total suite <5 minutes
    - Run on feature branches and PRs

GPU Smoke Tests:
    - Optional manual validation
    - Outside CI/CD pipeline
    - Real hardware testing for releases

Usage:
    uv run python run_tests.py                  # Run tiered tests (unit -> integration)
    uv run python run_tests.py --unit           # Run unit tests only
    uv run python run_tests.py --integration    # Run integration tests only
    uv run python run_tests.py --gpu            # Run GPU tests only
    uv run python run_tests.py --fast           # Run unit + integration tests
    uv run python run_tests.py --performance    # Run performance benchmarks
    uv run python run_tests.py --smoke          # Run basic smoke tests
    uv run python run_tests.py --coverage       # Generate detailed coverage report
    uv run python run_tests.py --clean          # Clean test artifacts
"""

import argparse
import contextlib
import importlib.util
import json
import os
import re
import shutil
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

    def _has_xdist(self) -> bool:
        """Return True when pytest-xdist is available."""
        try:
            return importlib.util.find_spec("xdist") is not None
        except (ImportError, AttributeError, ValueError):
            return False

    def _reset_coverage_artifacts(self) -> None:
        """Remove stale coverage artifacts to keep reports reproducible."""
        coverage_dir = self.project_root / "coverage"
        coverage_dir.mkdir(parents=True, exist_ok=True)

        stale_paths = [
            self.project_root / ".coverage",
            coverage_dir / ".coverage",
            self.project_root / "coverage.json",
            self.project_root / "coverage.xml",
            self.project_root / "htmlcov",
        ]
        for path in stale_paths:
            if not path.exists():
                continue
            if path.is_file():
                path.unlink(missing_ok=True)
            else:
                shutil.rmtree(path, ignore_errors=True)

    def clean_artifacts(self) -> None:
        """Clean test artifacts and caches."""
        print("Cleaning test artifacts...")

        # Targets that should be searched for recursively
        recursive_targets = {
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".coverage",
        }

        # Explicit paths relative to project root
        explicit_targets = [
            "htmlcov",
            "coverage.xml",
            "coverage.json",
            "coverage/.coverage",
        ]

        # Clean recursive targets
        for pattern in recursive_targets:
            for path in self.project_root.rglob(pattern):
                with contextlib.suppress(OSError):
                    if path.is_file():
                        path.unlink()
                    else:
                        # Use ignore_errors=True for robust rmtree
                        shutil.rmtree(path, ignore_errors=True)

        # Clean explicit targets
        for rel_path in explicit_targets:
            path = self.project_root / rel_path
            if path.exists():
                with contextlib.suppress(OSError):
                    if path.is_file():
                        path.unlink()
                    else:
                        shutil.rmtree(path, ignore_errors=True)

        print("OK: Artifacts cleaned")

    def run_command(self, command: list[str], description: str) -> TestResult:
        """Run a test command and capture results."""
        result = TestResult()
        result.command = " ".join(command)

        print(f"\n{'=' * 60}")
        print(f"TEST: {description}")
        print(f"Command: {result.command}")
        print("=" * 60)

        start_time = time.time()

        try:
            process = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=False,
                timeout=1800,  # 30 minutes timeout
            )

            result.duration = time.time() - start_time
            result.exit_code = process.returncode
            result.output = process.stdout

            if process.stderr:
                result.output += f"\n--- STDERR ---\n{process.stderr}"

            # Parse pytest output for statistics
            if any("pytest" in part for part in command):
                self._parse_pytest_output(result)

            if result.exit_code == 0:
                print(f"OK: {description} completed successfully")
                if result.passed > 0:
                    print(
                        f"   Stats: {result.passed} passed, "
                        f"{result.failed} failed, {result.skipped} skipped"
                    )
            else:
                print(f"FAIL: {description} failed with exit code {result.exit_code}")
                if result.failed > 0:
                    print(
                        f"   Stats: {result.passed} passed, "
                        f"{result.failed} failed, {result.skipped} skipped"
                    )

            print(f"   Duration: {result.duration:.2f}s")

        except subprocess.TimeoutExpired:
            result.duration = time.time() - start_time
            result.exit_code = -1
            result.output = "Test execution timed out after 30 minutes"
            print(f"TIMEOUT: {description} timed out")

        except (subprocess.SubprocessError, OSError, ValueError) as e:
            result.duration = time.time() - start_time
            result.exit_code = -1
            result.output = f"Execution error: {e}"
            print(f"ERROR: {description} failed with error: {e}")

        self.results.append(result)
        return result

    def _parse_pytest_output(self, result: TestResult) -> None:
        """Parse pytest output to extract test statistics."""
        output = result.output

        # Look for test summary line like "5 passed, 2 failed, 1 skipped"

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
        """Run fast unit tests with mocked dependencies (<5s each)."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/unit/",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-fail-under=0",
            "--cov-report=term-missing",
            "--durations=10",
        ]
        # Run unit tests serially in CI for stability across environments
        return self.run_command(command, "Unit Tests (Tier 1 - Fast with mocks)")

    def run_integration_tests(self) -> TestResult:
        """Run integration tests with lightweight models (<30s each)."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/integration/",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-fail-under=0",
            "--cov-report=term-missing",
            "--durations=10",
        ]
        return self.run_command(
            command, "Integration Tests (Tier 2 - Lightweight models)"
        )

    def run_extras_tests(self) -> TestResult:
        """Run tests that require optional llama_index extras."""
        description = "Extras Tests (llama_index extras)"
        try:
            has_extras = (
                importlib.util.find_spec("llama_index.program.openai") is not None
            )
        except ModuleNotFoundError:
            has_extras = False

        if not has_extras:
            print(f"\n{'=' * 60}")
            print(f"TEST: {description}")
            print("=" * 60)
            print(
                "SKIP: llama_index.program.openai not installed; extras tests will be "
                "skipped."
            )
            result = TestResult()
            result.command = "pytest -m requires_llama (skipped - dependency missing)"
            result.exit_code = 0
            result.skipped = 1
            result.output = "Dependencies missing; skipped extras test lane."
            self.results.append(result)
            return result

        command = [
            "uv",
            "run",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--no-cov",
            "-m",
            "requires_llama",
        ]
        return self.run_command(command, description)

    def run_performance_tests(self) -> TestResult:
        """Run performance and benchmark tests."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/performance/",
            "-v",
            "--tb=short",
            "--durations=10",
            "-m",
            "performance",
        ]
        return self.run_command(command, "Performance Tests")

    def run_gpu_tests(self) -> TestResult:
        """Run GPU-required tests with hardware validation."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/",
            "-v",
            "--tb=short",
            "--durations=10",
            "-m",
            "requires_gpu or gpu_required",
        ]
        return self.run_command(command, "GPU Tests (Hardware validation)")

    def run_all_tests(self) -> TestResult:
        """Run unit+integration tests with coverage (exclude performance/e2e)."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/unit",
            "tests/integration",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml",
            "--cov-report=json:coverage.json",
            "--cov-report=term-missing",
            "--cov-branch",
            "--durations=20",
            "--maxfail=5",
        ]
        ci_env = os.getenv("CI") or os.getenv("GITHUB_ACTIONS")
        if ci_env and sys.version_info >= (3, 11):
            if self._has_xdist():
                command += ["-n", "auto"]
            else:
                print("pytest-xdist not available; running coverage tests serially.")
        return self.run_command(command, "All Tests with Coverage (unit+integration)")

    def run_smoke_tests(self) -> TestResult:
        """Run basic smoke tests to verify system health."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/unit/models/test_models.py",
            "tests/unit/config/test_validation.py",
            "-v",
            "--tb=line",
            "--maxfail=3",  # Stop after 3 failures for smoke tests
        ]
        return self.run_command(command, "Smoke Tests (Basic system health)")

    def run_fast_tests(self) -> TestResult:
        """Run unit and integration tests (excludes system tests)."""
        command = [
            "uv",
            "run",
            "pytest",
            "tests/unit",
            "tests/integration",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-report=term-missing",
            "--durations=10",
        ]
        return self.run_command(command, "Fast Tests (Unit + Integration)")

    def run_tiered_tests(self) -> None:
        """Run all tests in pyramid order: unit -> integration (no system tests)."""
        print("\nRunning Tiered Test Strategy")
        print("=" * 50)
        print("Tier 1: Unit Tests (mocked dependencies)")
        self._reset_coverage_artifacts()
        result_unit = self.run_unit_tests()

        if result_unit.exit_code != 0:
            print("Unit tests failed. Stopping tiered execution.")
            return

        print("\nTier 2: Integration Tests (lightweight models)")
        # Append coverage from Tier 1 so coverage reporting reflects both tiers.
        command = [
            "uv",
            "run",
            "pytest",
            "tests/integration/",
            "-v",
            "--tb=short",
            "--cov=src",
            "--cov-append",
            "--cov-fail-under=0",
            "--cov-report=term-missing",
            "--durations=10",
        ]
        result_integration = self.run_command(
            command, "Integration Tests (Tier 2 - Lightweight models)"
        )

        if result_integration.exit_code != 0:
            print("Integration tests failed. Stopping tiered execution.")
            return

        print("\nAll tiered tests passed! GPU smoke tests available via --gpu")

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
    'src.utils.core', 'src.utils.monitoring', 'src.processing.ingestion_api',
    'src.agents.coordinator', 'src.agents.tool_factory', 'src.agents.tools',
    'src.config.settings'
]

failed = []
for module in modules:
    try:
        importlib.import_module(module)
        print(f'OK {module}')
    except ImportError as e:
        print(f'FAIL {module}: {e}')
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
        print("COVERAGE ANALYSIS")
        print("=" * 60)

        # Run coverage report command
        coverage_cmd = [
            "uv",
            "run",
            "coverage",
            "report",
            "--show-missing",
            "--fail-under=0",
        ]
        self.run_command(coverage_cmd, "Coverage Report")

        coverage_json = self.project_root / "coverage.json"
        coverage_xml = self.project_root / "coverage.xml"
        coverage_html = self.project_root / "htmlcov"
        if not (
            coverage_json.exists() and coverage_xml.exists() and coverage_html.exists()
        ):
            self.run_command(
                [
                    "uv",
                    "run",
                    "coverage",
                    "json",
                    "--fail-under=0",
                    "-o",
                    str(coverage_json),
                ],
                "Coverage JSON",
            )
            self.run_command(
                [
                    "uv",
                    "run",
                    "coverage",
                    "xml",
                    "--fail-under=0",
                    "-o",
                    str(coverage_xml),
                ],
                "Coverage XML",
            )
            self.run_command(
                [
                    "uv",
                    "run",
                    "coverage",
                    "html",
                    "--fail-under=0",
                    "-d",
                    str(coverage_html),
                ],
                "Coverage HTML",
            )

        # Generate coverage analysis if coverage.json exists
        if coverage_json.exists():
            self._analyze_coverage_data(coverage_json)

    def _analyze_coverage_data(self, coverage_json: Path) -> None:
        """Analyze coverage data and provide insights."""
        try:
            with coverage_json.open() as f:
                data = json.load(f)

            files = data.get("files", {})
            if not files:
                print("ERROR: No coverage data found")
                return

            # Calculate overall stats
            total_lines = sum(f["summary"]["num_statements"] for f in files.values())
            covered_lines = sum(f["summary"]["covered_lines"] for f in files.values())
            overall_pct = (covered_lines / total_lines) * 100 if total_lines > 0 else 0

            print(f"\nOVERALL COVERAGE: {overall_pct:.1f}%")
            print(f"  Total lines: {total_lines}")
            print(f"  Covered lines: {covered_lines}")
            print(f"  Missing lines: {total_lines - covered_lines}")

            # Find low coverage files
            low_coverage_files = []
            for filename, file_data in files.items():
                coverage_pct = file_data["summary"]["percent_covered"]
                if coverage_pct < 80 and not any(
                    skip in filename for skip in ["test_", "__pycache__", ".pyc"]
                ):
                    low_coverage_files.append((filename, coverage_pct))

            if low_coverage_files:
                print("\nFILES WITH LOW COVERAGE (<80%):")
                print("-" * 50)
                for filename, coverage_pct in sorted(
                    low_coverage_files, key=lambda x: x[1]
                ):
                    print(f"   {filename}: {coverage_pct:.1f}%")

            # Identify critical files
            critical_files = [
                "src/config/settings.py",
                "src/models/embeddings.py",
                "src/persistence/chat_db.py",
                "src/persistence/memory_store.py",
                "src/persistence/snapshot.py",
                "src/retrieval/hybrid.py",
                "src/retrieval/router_factory.py",
                "src/agents/coordinator.py",
                "src/agents/tool_factory.py",
                "src/agents/tools/router_tool.py",
                "src/agents/tools/planning.py",
                "src/agents/tools/retrieval.py",
                "src/agents/tools/synthesis.py",
                "src/agents/tools/validation.py",
                "src/utils/core.py",
                "src/processing/ingestion_api.py",
                "src/utils/monitoring.py",
                "src/app.py",
            ]

            print("\nCRITICAL FILE COVERAGE:")
            print("-" * 50)
            for critical in critical_files:
                found = False
                for filename, file_data in files.items():
                    if critical in filename:
                        coverage_pct = file_data["summary"]["percent_covered"]
                        status = "OK" if coverage_pct >= 80 else "WARN"
                        print(f"   {status} {filename}: {coverage_pct:.1f}%")
                        found = True
                        break
                if not found:
                    print(f"   WARN: {critical}: Not found in coverage data")

        except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
            print(f"ERROR: Error analyzing coverage data: {e}")

    def print_summary(self) -> None:
        """Print comprehensive test execution summary."""
        print("\n" + "=" * 80)
        print("TEST EXECUTION SUMMARY")
        print("=" * 80)

        total_passed = sum(r.passed for r in self.results)
        total_failed = sum(r.failed for r in self.results)
        total_skipped = sum(r.skipped for r in self.results)
        total_errors = sum(r.errors for r in self.results)
        total_duration = sum(r.duration for r in self.results)

        print("OVERALL STATS:")
        print(f"   Passed:  {total_passed}")
        print(f"   Failed:  {total_failed}")
        print(f"   Skipped: {total_skipped}")
        print(f"   Errors:  {total_errors}")
        print(f"   Total Duration: {total_duration:.1f}s")

        print("\nDETAILED RESULTS:")
        print("-" * 60)

        for result in self.results:
            status = "OK" if result.exit_code == 0 else "FAIL"
            print(f"   {status} {result.command}")
            if result.passed > 0 or result.failed > 0:
                print(
                    f"      {result.passed}P {result.failed}F "
                    f"{result.skipped}S - {result.duration:.1f}s"
                )
            else:
                print(f"      Duration: {result.duration:.1f}s")

        # Show failures
        failed_results = [r for r in self.results if r.exit_code != 0]
        if failed_results:
            print("\nFAILED TESTS:")
            print("-" * 60)
            for result in failed_results:
                print(f"   Command: {result.command}")
                print(f"   Exit Code: {result.exit_code}")
                if result.output:
                    # Show a generous tail of output for diagnosis
                    output_lines = result.output.split("\n")
                    tail = 300 if len(output_lines) > 300 else len(output_lines)
                    print("   Last output (tail):")
                    for line in output_lines[-tail:]:
                        if line.strip():
                            print(f"     {line}")
                print()

        # Print recommended actions
        print("\nRECOMMENDATIONS:")
        if total_failed > 0:
            print("   - Fix failing tests before deployment")
        if total_failed == 0 and total_passed > 0:
            print("   All tests passing! Ready for deployment")

        print("   Coverage report: htmlcov/index.html")
        print("   Coverage data: coverage.json")
        print("   Re-run specific tests: uv run pytest tests/test_<name>.py -v")


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the tiered test runner."""
    parser = argparse.ArgumentParser(
        description="DocMind AI Tiered Test Runner",
        epilog="""Three-Tier Testing Strategy:
  Tier 1 (Unit): Fast tests with mocks (<5s each)
  Tier 2 (Integration): Lightweight models (<30s each)
  Tier 3 (System): Real models + GPU (<5min each)

Examples:
  uv run python scripts/run_tests.py                    # Run all tiers in sequence
  uv run python scripts/run_tests.py --unit --fast      # Quick unit test validation
  uv run python scripts/run_tests.py --integration      # Integration tests only
  uv run python scripts/run_tests.py --gpu              # GPU tests only (requires GPU)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run unit tests only (Tier 1 - mocked dependencies)",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration tests only (Tier 2 - lightweight models)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run unit + integration tests (exclude system)",
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run performance benchmark tests"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run GPU-required tests with hardware validation",
    )
    parser.add_argument(
        "--extras",
        action="store_true",
        help="Run tests that require optional llama_index extras",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run basic smoke tests (quick system health check)",
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate detailed coverage report"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean test artifacts before running"
    )
    parser.add_argument(
        "--validate-imports",
        action="store_true",
        help="Validate that all modules can be imported",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional test paths or patterns to pass directly to pytest.",
    )
    return parser


def _print_default_tiers() -> None:
    """Print guidance for the default tiered test strategy."""
    print("\nRunning Default Tiered Test Strategy")
    print("\nTiers:")
    print("   --unit: Fast tests with mocks (development)")
    print("   --integration: Lightweight models (PR validation)")
    print("   --gpu: Manual GPU smoke tests (staging/release)")
    print("   --fast: Unit + Integration only")
    print("   --extras: Optional-dependency tests (requires llama_index extras)")


def _run_direct_paths(runner: TestRunner, args: argparse.Namespace) -> None:
    """Run pytest directly for the provided paths."""
    cmd = ["uv", "run", "pytest", *args.paths, "-v", "--tb=short"]
    if args.coverage:
        cmd += ["--cov=src", "--cov-report=term-missing"]
    else:
        cmd += ["--no-cov"]
    runner.run_command(cmd, "Direct pytest (paths)")


def _run_selected_tests(runner: TestRunner, args: argparse.Namespace) -> None:
    """Dispatch to the appropriate test execution flow."""
    if args.paths:
        _run_direct_paths(runner, args)
        return
    if args.validate_imports:
        runner.validate_imports()
    elif args.smoke:
        runner.run_smoke_tests()
    elif args.unit:
        runner.run_unit_tests()
    elif args.integration:
        runner.run_integration_tests()
    elif args.extras:
        runner.run_extras_tests()
    elif args.fast:
        runner.run_fast_tests()
    elif args.performance:
        runner.run_performance_tests()
    elif args.gpu:
        runner.run_gpu_tests()
    elif args.coverage:
        runner.run_all_tests()
    else:
        _print_default_tiers()
        runner.validate_imports()
        runner.run_tiered_tests()


def _should_generate_coverage(args: argparse.Namespace) -> bool:
    """Return True when a coverage report should be generated."""
    if args.coverage:
        return True
    if args.paths:
        return False
    return not any(
        (
            args.fast,
            args.unit,
            args.integration,
            args.extras,
            args.performance,
            args.gpu,
            args.smoke,
            args.validate_imports,
        )
    )


def _finalize_run(runner: TestRunner) -> int:
    """Print summary and return exit code.

    Returns:
        0 if all test runs succeeded, 1 if any runs failed.
    """
    runner.print_summary()
    failed_runs = sum(1 for r in runner.results if r.exit_code != 0)
    if failed_runs > 0:
        print(f"\n{failed_runs} test run(s) failed")
        return 1
    print("\nAll test runs completed successfully")
    return 0


def main() -> int:
    """Main entry point for tiered test runner.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    runner = TestRunner(project_root)

    print("DocMind AI Test Suite")
    print("=" * 50)

    if args.clean:
        runner.clean_artifacts()

    try:
        _run_selected_tests(runner, args)
        if _should_generate_coverage(args):
            runner.generate_coverage_report()
    except KeyboardInterrupt:
        print("\nWARN: Test execution interrupted by user")
        return 1
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"\nUnexpected error: {e}")
        return 1

    return _finalize_run(runner)


if __name__ == "__main__":
    sys.exit(main())
