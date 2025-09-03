#!/usr/bin/env python3
"""Structural Changes Validation Runner.

This script provides comprehensive validation of structural improvements to ensure
no performance regressions or integration issues were introduced by:

- Configuration unification (76% complexity reduction)
- Directory flattening (6 levels → 2 levels)
- Import resolution (64 errors fixed)
- Code quality improvements (174 → 49 ruff errors)
- Test recovery (17.4% → 81.2% pass rate)

Usage:
    python scripts/validate_structural_changes.py [--quick] [--report-file output.json]

Options:
    --quick         Run only critical validation tests (faster)
    --report-file   Save detailed report to JSON file
    --verbose       Enable verbose output
    --no-cleanup    Skip test artifact cleanup
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class StructuralValidationRunner:
    """Runs comprehensive structural validation tests and generates reports."""

    def __init__(self, quick_mode: bool = False, verbose: bool = False):
        """Initialize the structural changes validator.

        Args:
            quick_mode: If True, run only critical tests for faster validation.
            verbose: If True, provide detailed output during validation.
        """
        self.quick_mode = quick_mode
        self.verbose = verbose
        self.results = {
            "validation_timestamp": datetime.now().isoformat(),
            "quick_mode": quick_mode,
            "test_results": {},
            "performance_metrics": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "skipped_tests": 0,
                "critical_failures": [],
                "performance_regressions": [],
                "overall_status": "PENDING",
            },
        }

    def run_validation(self) -> dict:
        """Run comprehensive structural validation."""
        print("🔍 DocMind AI Structural Changes Validation")
        print("=" * 60)
        print(f"Mode: {'Quick' if self.quick_mode else 'Comprehensive'}")
        print(f"Started: {self.results['validation_timestamp']}")
        print()

        # 1. Import Performance Validation
        self._run_import_performance_tests()

        # 2. Configuration System Validation
        self._run_configuration_tests()

        # 3. Integration Workflow Validation
        self._run_integration_workflow_tests()

        # 4. Memory Usage Validation
        self._run_memory_validation_tests()

        # 5. Performance Regression Detection
        if not self.quick_mode:
            self._run_performance_regression_tests()

        # 6. Error Handling Validation
        self._run_error_handling_tests()

        # Generate final report
        self._generate_final_report()

        return self.results

    def _run_import_performance_tests(self):
        """Run import performance validation tests."""
        print("📦 Testing Import Performance After Directory Flattening")
        print("-" * 50)

        test_command = [
            "python",
            "-m",
            "pytest",
            "tests/performance/test_structural_performance_validation.py::TestImportPerformancePostFlattening",
            "-v",
            "--tb=short",
        ]

        if self.quick_mode:
            # Run only critical import tests in quick mode
            test_command.extend(
                [
                    "-k",
                    (
                        "test_core_module_import_performance "
                        "or test_import_memory_overhead"
                    ),
                ]
            )

        result = self._run_pytest_command(test_command, "import_performance")

        if result["status"] == "PASSED":
            print("✅ Import performance validation: PASSED")
            print("   - Import times within targets")
            print("   - Memory overhead acceptable")
        else:
            print("❌ Import performance validation: FAILED")
            self.results["summary"]["critical_failures"].append(
                "Import performance regression"
            )

        print()

    def _run_configuration_tests(self):
        """Run unified configuration system validation."""
        print("⚙️  Testing Unified Configuration System Performance")
        print("-" * 50)

        test_command = [
            "python",
            "-m",
            "pytest",
            "tests/performance/test_structural_performance_validation.py::TestUnifiedConfigurationPerformance",
            "-v",
            "--tb=short",
        ]

        result = self._run_pytest_command(test_command, "configuration_performance")

        if result["status"] == "PASSED":
            print("✅ Configuration system validation: PASSED")
            print("   - Configuration loading within 50ms target")
            print("   - Environment variable processing efficient")
            print("   - Nested model synchronization fast")
        else:
            print("❌ Configuration system validation: FAILED")
            self.results["summary"]["critical_failures"].append(
                "Configuration performance regression"
            )

        # Test configuration integration
        integration_test_command = [
            "python",
            "-m",
            "pytest",
            "tests/integration/test_structural_integration_workflows.py::TestConfigurationPropagationIntegration",
            "-v",
            "--tb=short",
        ]

        integration_result = self._run_pytest_command(
            integration_test_command, "configuration_integration"
        )

        if integration_result["status"] == "PASSED":
            print("✅ Configuration integration: PASSED")
        else:
            print("❌ Configuration integration: FAILED")
            self.results["summary"]["critical_failures"].append(
                "Configuration integration failure"
            )

        print()

    def _run_integration_workflow_tests(self):
        """Run integration workflow validation."""
        print("🔗 Testing Integration Workflows After Reorganization")
        print("-" * 50)

        test_suites = [
            (
                "Document Processing Pipeline",
                "tests/integration/test_structural_integration_workflows.py::TestDocumentProcessingPipelineIntegration",
            ),
            (
                "Multi-Agent Coordination",
                "tests/integration/test_structural_integration_workflows.py::TestMultiAgentCoordinationIntegration",
            ),
            (
                "Retrieval System Integration",
                "tests/integration/test_structural_integration_workflows.py::TestRetrievalSystemIntegration",
            ),
        ]

        all_passed = True

        for suite_name, test_path in test_suites:
            if self.quick_mode and "Retrieval" in suite_name:
                # Skip retrieval system in quick mode
                continue

            test_command = ["python", "-m", "pytest", test_path, "-v", "--tb=short"]

            result = self._run_pytest_command(
                test_command, f"integration_{suite_name.lower().replace(' ', '_')}"
            )

            if result["status"] == "PASSED":
                print(f"✅ {suite_name}: PASSED")
            else:
                print(f"❌ {suite_name}: FAILED")
                all_passed = False
                self.results["summary"]["critical_failures"].append(
                    f"{suite_name} integration failure"
                )

        if all_passed:
            print("✅ All integration workflows: PASSED")
        else:
            print("❌ Some integration workflows: FAILED")

        print()

    def _run_memory_validation_tests(self):
        """Run memory usage validation tests."""
        print("🧠 Testing Memory Usage After Structural Changes")
        print("-" * 50)

        # Test that structural changes haven't increased memory usage
        test_command = [
            "python",
            "-m",
            "pytest",
            "tests/performance/test_memory_benchmarks.py::TestMemoryLeakDetection::test_embedding_memory_stability",
            "tests/performance/test_memory_benchmarks.py::TestMemoryBenchmarks::test_memory_usage_scaling",
            "-v",
            "--tb=short",
        ]

        result = self._run_pytest_command(test_command, "memory_validation")

        # Also run structural memory overhead test
        structural_memory_command = [
            "python",
            "-m",
            "pytest",
            "tests/performance/test_structural_performance_validation.py::TestImportPerformancePostFlattening::test_import_memory_overhead",
            "-v",
            "--tb=short",
        ]

        structural_result = self._run_pytest_command(
            structural_memory_command, "structural_memory"
        )

        if result["status"] == "PASSED" and structural_result["status"] == "PASSED":
            print("✅ Memory usage validation: PASSED")
            print("   - No memory leaks detected")
            print("   - Import memory overhead within limits")
            print("   - Memory scaling remains efficient")
        else:
            print("❌ Memory usage validation: FAILED")
            self.results["summary"]["performance_regressions"].append(
                "Memory usage increased"
            )

        print()

    def _run_performance_regression_tests(self):
        """Run comprehensive performance regression detection."""
        print("🚀 Testing Performance Regression Detection")
        print("-" * 50)

        test_command = [
            "python",
            "-m",
            "pytest",
            "tests/performance/test_structural_performance_validation.py::TestStructuralPerformanceRegression",
            "-v",
            "--tb=short",
        ]

        result = self._run_pytest_command(test_command, "performance_regression")

        if result["status"] == "PASSED":
            print("✅ Performance regression detection: PASSED")
            print("   - Comprehensive workflow performance maintained")
            print("   - All performance targets met")
        else:
            print("❌ Performance regression detection: FAILED")
            self.results["summary"]["performance_regressions"].append(
                "Comprehensive workflow regression"
            )

        print()

    def _run_error_handling_tests(self):
        """Run error handling and resilience validation."""
        print("🛡️  Testing Error Handling and Resilience")
        print("-" * 50)

        test_command = [
            "python",
            "-m",
            "pytest",
            "tests/integration/test_structural_integration_workflows.py::TestErrorHandlingAndResilienceIntegration",
            "-v",
            "--tb=short",
        ]

        result = self._run_pytest_command(test_command, "error_handling")

        if result["status"] == "PASSED":
            print("✅ Error handling validation: PASSED")
            print("   - Component failures handled gracefully")
            print("   - Configuration validation works")
            print("   - Async error propagation preserved")
        else:
            print("❌ Error handling validation: FAILED")
            self.results["summary"]["critical_failures"].append(
                "Error handling regression"
            )

        print()

    def _run_pytest_command(self, command: list[str], test_category: str) -> dict:
        """Run a pytest command and capture results."""
        start_time = time.perf_counter()

        try:
            if self.verbose:
                print(f"Running: {' '.join(command)}")

            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per test suite
            )

            duration = time.perf_counter() - start_time

            # Parse pytest output for test counts
            output_lines = result.stdout.split("\n")
            passed = failed = skipped = 0

            for line in output_lines:
                if "passed" in line and "failed" in line:
                    # Parse line like "5 passed, 2 failed, 1 skipped"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed,":
                            passed = int(parts[i - 1])
                        elif part == "failed,":
                            failed = int(parts[i - 1])
                        elif part == "skipped":
                            skipped = int(parts[i - 1])
                elif line.strip().endswith("passed"):
                    # Parse line like "5 passed"
                    parts = line.split()
                    passed = int(parts[0])

            status = "PASSED" if result.returncode == 0 else "FAILED"

            test_result = {
                "status": status,
                "duration_seconds": round(duration, 2),
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

            self.results["test_results"][test_category] = test_result

            # Update summary
            self.results["summary"]["total_tests"] += passed + failed + skipped
            self.results["summary"]["passed_tests"] += passed
            self.results["summary"]["failed_tests"] += failed
            self.results["summary"]["skipped_tests"] += skipped

            if self.verbose and result.stdout:
                print(f"Test output:\n{result.stdout}")

            return test_result

        except subprocess.TimeoutExpired:
            duration = time.perf_counter() - start_time
            print(
                f"⚠️  Test suite {test_category} timed out after {duration:.1f} seconds"
            )

            test_result = {
                "status": "TIMEOUT",
                "duration_seconds": round(duration, 2),
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "stdout": "",
                "stderr": "Test suite timed out",
                "return_code": -1,
            }

            self.results["test_results"][test_category] = test_result
            self.results["summary"]["failed_tests"] += 1
            self.results["summary"]["total_tests"] += 1

            return test_result

        except Exception as e:
            duration = time.perf_counter() - start_time
            print(f"❌ Error running test suite {test_category}: {e}")

            test_result = {
                "status": "ERROR",
                "duration_seconds": round(duration, 2),
                "passed": 0,
                "failed": 1,
                "skipped": 0,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
            }

            self.results["test_results"][test_category] = test_result
            self.results["summary"]["failed_tests"] += 1
            self.results["summary"]["total_tests"] += 1

            return test_result

    def _generate_final_report(self):
        """Generate final validation report."""
        summary = self.results["summary"]

        # Calculate overall status
        if summary["failed_tests"] == 0 and len(summary["critical_failures"]) == 0:
            summary["overall_status"] = "PASSED"
        elif len(summary["critical_failures"]) > 0:
            summary["overall_status"] = "CRITICAL_FAILURE"
        else:
            summary["overall_status"] = "FAILED"

        # Calculate performance metrics
        total_duration = sum(
            result.get("duration_seconds", 0)
            for result in self.results["test_results"].values()
        )

        self.results["performance_metrics"] = {
            "total_validation_duration_seconds": round(total_duration, 2),
            "test_suites_run": len(self.results["test_results"]),
            "average_suite_duration_seconds": round(
                total_duration / max(len(self.results["test_results"]), 1), 2
            ),
        }

        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed_tests']} ✅")
        print(f"Failed: {summary['failed_tests']} ❌")
        print(f"Skipped: {summary['skipped_tests']} ⏭️")
        print(f"Total Duration: {total_duration:.1f}s")
        print()

        if summary["critical_failures"]:
            print("🚨 CRITICAL FAILURES:")
            for failure in summary["critical_failures"]:
                print(f"   - {failure}")
            print()

        if summary["performance_regressions"]:
            print("⚠️  PERFORMANCE REGRESSIONS:")
            for regression in summary["performance_regressions"]:
                print(f"   - {regression}")
            print()

        # Test suite results
        print("📋 TEST SUITE RESULTS:")
        for suite_name, result in self.results["test_results"].items():
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(
                f"   {status_icon} {suite_name.replace('_', ' ').title()}: "
                f"{result['status']} ({result['duration_seconds']}s)"
            )

        print()

        if summary["overall_status"] == "PASSED":
            print("🎉 STRUCTURAL CHANGES VALIDATION: SUCCESS")
            print("   All performance and integration tests passed!")
            print("   The structural improvements have not introduced regressions.")
        elif summary["overall_status"] == "CRITICAL_FAILURE":
            print("💥 STRUCTURAL CHANGES VALIDATION: CRITICAL FAILURE")
            print("   Critical functionality has been broken by structural changes.")
            print("   Immediate investigation and fixes required.")
        else:
            print("⚠️  STRUCTURAL CHANGES VALIDATION: PARTIAL FAILURE")
            print("   Some tests failed but no critical functionality broken.")
            print("   Review failures and address as needed.")

        print()

    def save_report(self, file_path: Path):
        """Save detailed validation report to JSON file."""
        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"📄 Detailed report saved to: {file_path}")


def main():
    """Main entry point for structural validation."""
    parser = argparse.ArgumentParser(
        description="Validate DocMind AI structural changes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only critical validation tests (faster)",
    )

    parser.add_argument(
        "--report-file", type=Path, help="Save detailed report to JSON file"
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--no-cleanup", action="store_true", help="Skip test artifact cleanup"
    )

    args = parser.parse_args()

    # Run validation
    runner = StructuralValidationRunner(quick_mode=args.quick, verbose=args.verbose)

    try:
        results = runner.run_validation()

        # Save report if requested
        if args.report_file:
            runner.save_report(args.report_file)

        # Cleanup test artifacts unless disabled
        if not args.no_cleanup:
            print("🧹 Cleaning up test artifacts...")
            cleanup_command = [
                "python",
                "-c",
                "import shutil, pathlib; "
                "[shutil.rmtree(p, ignore_errors=True) for p in "
                "pathlib.Path('.').glob('**/__pycache__')] + "
                "[shutil.rmtree(p, ignore_errors=True) for p in "
                "pathlib.Path('.').glob('**/.pytest_cache')]",
            ]
            subprocess.run(cleanup_command, capture_output=True)

        # Exit with appropriate code
        if results["summary"]["overall_status"] == "PASSED":
            print("✨ Validation completed successfully!")
            sys.exit(0)
        elif results["summary"]["overall_status"] == "CRITICAL_FAILURE":
            print("💥 Critical validation failures detected!")
            sys.exit(2)
        else:
            print("⚠️  Validation completed with some failures.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Validation interrupted by user")
        sys.exit(130)

    except Exception as e:
        print(f"💥 Validation failed with error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
