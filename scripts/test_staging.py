#!/usr/bin/env python
"""Staging Validation Test Runner for DocMind AI.

This script runs the complete tiered test suite including system tests with real models.
Suitable for staging environments where GPU resources are available and comprehensive
validation is required before production deployment.

Test Strategy:
- Tier 1: Unit Tests (mocked dependencies, <5s each)
- Tier 2: Integration Tests (lightweight models, <30s each)
- Tier 3: System Tests (real models + GPU, <5min each)
- Performance validation and benchmarking

Usage:
    python scripts/test_staging.py                    # Full staging validation
    python scripts/test_staging.py --skip-system      # Skip GPU-intensive tests
    python scripts/test_staging.py --performance      # Include performance benchmarks
    python scripts/test_staging.py --coverage         # Generate coverage reports
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def check_gpu_availability() -> bool:
    """Check if GPU is available for system tests."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "name" in result.stdout:
            print("✅ GPU detected for system tests")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    print("⚠️  No GPU detected - system tests will be skipped")
    return False


def run_command(
    command: list[str], description: str, timeout: int = 1800
) -> tuple[int, str]:
    """Run a command and return exit code and output."""
    print(f"\n{'=' * 60}")
    print(f"🧪 {description}")
    print(f"📋 Command: {' '.join(command)}")
    print("=" * 60)

    start_time = time.time()

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} completed successfully ({duration:.1f}s)")
        else:
            print(f"❌ {description} failed ({duration:.1f}s)")
            if result.stderr:
                print("STDERR:", result.stderr[-1000:])  # Last 1000 chars

        return result.returncode, result.stdout

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"⏰ {description} timed out after {duration:.1f}s")
        return -1, "Timeout"
    except Exception as e:
        duration = time.time() - start_time
        print(f"💥 {description} error after {duration:.1f}s: {e}")
        return -1, str(e)


def main():
    """Main staging validation runner."""
    parser = argparse.ArgumentParser(description="DocMind AI Staging Test Runner")
    parser.add_argument(
        "--skip-system", action="store_true", help="Skip system tests (GPU-intensive)"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Include performance benchmark tests"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate comprehensive coverage report"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first test failure"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    print("🚀 DocMind AI Staging Validation Suite")
    print("=" * 60)
    print("Target: Comprehensive validation for staging/production")
    print("Scope: Full tiered test strategy + system validation")

    # Check GPU availability
    gpu_available = check_gpu_availability()
    if not gpu_available and not args.skip_system:
        print("\n⚠️  Warning: System tests require GPU but none detected.")
        print("   Use --skip-system to run without GPU-intensive tests.")

    # Change to project root
    import os

    os.chdir(project_root)

    exit_codes = []
    test_results = {}

    # Step 1: Import validation
    print("\n🔍 Step 1: Import Validation")
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--validate-imports"]
    exit_code, _ = run_command(cmd, "Import Validation", timeout=300)
    exit_codes.append(exit_code)
    test_results["imports"] = exit_code == 0

    if exit_code != 0 and args.fail_fast:
        print("❌ Import validation failed. Stopping staging validation.")
        sys.exit(1)

    # Step 2: Unit tests
    print("\n🧪 Step 2: Unit Tests (Tier 1)")
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--unit"]
    exit_code, _ = run_command(cmd, "Unit Tests", timeout=600)
    exit_codes.append(exit_code)
    test_results["unit"] = exit_code == 0

    if exit_code != 0 and args.fail_fast:
        print("❌ Unit tests failed. Stopping staging validation.")
        sys.exit(1)

    # Step 3: Integration tests
    print("\n⚙️ Step 3: Integration Tests (Tier 2)")
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--integration"]
    exit_code, _ = run_command(cmd, "Integration Tests", timeout=900)
    exit_codes.append(exit_code)
    test_results["integration"] = exit_code == 0

    if exit_code != 0 and args.fail_fast:
        print("❌ Integration tests failed. Stopping staging validation.")
        sys.exit(1)

    # Step 4: System tests (if GPU available and not skipped)
    if gpu_available and not args.skip_system:
        print("\n🎯 Step 4: System Tests (Tier 3)")
        cmd = ["uv", "run", "python", "scripts/run_tests.py", "--system"]
        exit_code, _ = run_command(cmd, "System Tests", timeout=1800)
        exit_codes.append(exit_code)
        test_results["system"] = exit_code == 0

        if exit_code != 0 and args.fail_fast:
            print("❌ System tests failed. Stopping staging validation.")
            sys.exit(1)
    else:
        test_results["system"] = "skipped"

    # Step 5: Performance tests (if requested)
    if args.performance:
        print("\n📊 Step 5: Performance Benchmarks")
        cmd = ["uv", "run", "python", "scripts/run_tests.py", "--performance"]
        exit_code, _ = run_command(cmd, "Performance Tests", timeout=1200)
        exit_codes.append(exit_code)
        test_results["performance"] = exit_code == 0

        if exit_code != 0 and args.fail_fast:
            print("❌ Performance tests failed. Stopping staging validation.")
            sys.exit(1)

    # Step 6: Coverage report (if requested)
    if args.coverage:
        print("\n📈 Step 6: Coverage Analysis")
        cmd = ["uv", "run", "python", "scripts/run_tests.py", "--coverage"]
        exit_code, _ = run_command(cmd, "Coverage Report", timeout=600)
        exit_codes.append(exit_code)
        test_results["coverage"] = exit_code == 0

    # Final validation summary
    total_failures = sum(1 for code in exit_codes if code != 0)

    print("\n" + "=" * 80)
    print("📋 STAGING VALIDATION SUMMARY")
    print("=" * 80)

    # Show detailed results
    print("\n📊 Test Results:")
    for test_name, result in test_results.items():
        if result == "skipped":
            print(f"   ⏭️  {test_name.title()}: Skipped")
        elif result:
            print(f"   ✅ {test_name.title()}: Passed")
        else:
            print(f"   ❌ {test_name.title()}: Failed")

    # Final assessment
    if total_failures == 0:
        print("\n🎉 All staging tests passed!")
        print("✅ System is ready for production deployment")

        # Provide deployment recommendations
        print("\n💡 Deployment Recommendations:")
        print("   🚀 Ready to deploy to production")
        print("   📊 Monitor performance metrics post-deployment")
        print("   🔄 Consider running performance tests in production")

        if args.coverage:
            print("   📄 Coverage report: htmlcov/index.html")

        sys.exit(0)
    else:
        print(f"\n❌ {total_failures} test suite(s) failed")
        print("🔧 Address issues before production deployment")

        # Show recommendations based on failures
        print("\n💡 Remediation Steps:")
        if not test_results.get("unit", True):
            print("   🧪 Fix unit test failures (critical for stability)")
        if not test_results.get("integration", True):
            print("   ⚙️  Fix integration issues (component interactions)")
        if not test_results.get("system", True) and test_results["system"] != "skipped":
            print("   🎯 Fix system test failures (end-to-end functionality)")

        print("   📄 Check test logs for detailed failure information")

        sys.exit(1)


if __name__ == "__main__":
    main()
