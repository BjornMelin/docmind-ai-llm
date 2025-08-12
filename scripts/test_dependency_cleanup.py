#!/usr/bin/env python3
"""Test runner for dependency cleanup validation.

This script runs all tests related to PR #2 dependency cleanup validation
and provides a comprehensive report on the status of the dependency cleanup.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Determine package manager to use
PACKAGE_MANAGER = os.environ.get("PACKAGE_MANAGER", "uv").lower()


def get_test_command(test_path: str) -> list[str]:
    """Build the test command based on the selected package manager."""
    if PACKAGE_MANAGER == "uv":
        return ["uv", "run", "pytest", test_path, "-v", "--tb=short"]
    elif PACKAGE_MANAGER == "poetry":
        if shutil.which("poetry"):
            return ["poetry", "run", "pytest", test_path, "-v", "--tb=short"]
        else:
            raise RuntimeError("Poetry is not installed or not in PATH")
    elif PACKAGE_MANAGER == "pip":
        # Assume pytest is installed in the current environment
        return [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]
    else:
        raise RuntimeError(f"Unsupported package manager: {PACKAGE_MANAGER}")


def run_test_suite(test_path: str, description: str) -> tuple[bool, str]:
    """Run a test suite and return success status and output."""
    print(f"\n{'=' * 60}")
    print(f"Running {description} (using {PACKAGE_MANAGER})")
    print(f"{'=' * 60}")

    try:
        cmd = get_test_command(test_path)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return result.returncode == 0, result.stdout

    except Exception as e:
        error_msg = f"Failed to run tests: {e}"
        print(error_msg)
        return False, error_msg


def main():
    """Run all dependency cleanup validation tests."""
    print("DocMind AI - Dependency Cleanup Validation Test Suite")
    print("=" * 60)
    print("Validating PR #2 dependency cleanup...")

    test_suites = [
        ("tests/unit/test_dependencies.py", "Core Dependency Tests"),
        (
            "tests/unit/test_edge_cases_dependency_cleanup.py",
            "Edge Cases & Fallback Tests",
        ),
        ("tests/integration/test_app_startup.py", "App Startup Integration Tests"),
    ]

    results = []
    total_tests = 0
    total_passed = 0

    for test_path, description in test_suites:
        success, output = run_test_suite(test_path, description)
        results.append((description, success, output))

        # Extract test counts from output
        if "passed" in output:
            try:
                # Extract numbers from pytest output
                lines = output.split("\n")
                summary_line = [
                    line
                    for line in lines
                    if "passed" in line
                    and ("failed" in line or "skipped" in line or "error" in line)
                ]
                if summary_line:
                    # Parse something like "13 passed, 2 skipped in 4.21s"
                    parts = summary_line[0].split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed = int(parts[i - 1])
                            total_passed += passed
                            break

                    # Count total tests (rough estimate)
                    for i, part in enumerate(parts):
                        if "failed" in part:
                            failed = int(parts[i - 1])
                            total_tests += failed
                        elif "skipped" in part:
                            skipped = int(parts[i - 1])
                            total_tests += skipped

                    total_tests += passed

            except (ValueError, IndexError):
                pass  # Couldn't parse test counts

    # Print summary
    print(f"\n{'=' * 80}")
    print("DEPENDENCY CLEANUP VALIDATION SUMMARY")
    print(f"{'=' * 80}")

    all_passed = True
    for description, success, _output in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {description}")
        if not success:
            all_passed = False

    print(
        f"\nOverall Status: "
        f"{'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}"
    )

    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(
            f"Test Results: {total_passed}/{total_tests} passed ({success_rate:.1f}%)"
        )

    print(f"\n{'=' * 80}")
    print("VALIDATION CONCLUSIONS:")
    print(f"{'=' * 80}")

    if all_passed:
        print("✓ All dependency cleanup validation tests passed!")
        print("✓ LlamaIndex modular imports are working correctly")
        print("✓ Core dependencies are available and functional")
        print("✓ Optional dependencies are handled gracefully")
        print("✓ App startup components are working")
        print("✓ Error handling and fallback mechanisms are in place")
        print("\n🎉 PR #2 dependency cleanup appears to be successful!")
    else:
        print(
            "⚠ Some tests failed - this may indicate issues with the dependency cleanup"
        )
        print("⚠ Check the test output above for specific failure details")
        print("⚠ Some failures may be due to test environment limitations")

        # Provide guidance on common issues
        print("\nCommon issues and solutions:")
        print(
            "- LlamaCPP loading errors: Usually due to missing system libraries "
            "(expected in test environments)"
        )
        print(
            "- Import errors: May indicate missing optional dependencies "
            "(check if they're truly optional)"
        )
        print(
            "- Async function detection: May indicate functions that should be "
            "async but aren't"
        )
        print("- Module not found: May indicate core dependencies are missing")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
