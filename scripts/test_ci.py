#!/usr/bin/env python
"""CI/CD Pipeline Test Runner for DocMind AI.

This script runs unit and integration tests only - suitable for continuous integration
pipelines where GPU resources are not available and fast feedback is critical.

Test Strategy:
- Unit Tests (Tier 1): Mocked dependencies, <5s each
- Integration Tests (Tier 2): Lightweight models, <30s each
- No System Tests (Tier 3): Excluded to avoid GPU requirements

Usage:
    python scripts/test_ci.py                 # Run CI test suite
    python scripts/test_ci.py --unit-only     # Run only unit tests
    python scripts/test_ci.py --coverage      # Include coverage reporting
    python scripts/test_ci.py --fail-fast     # Stop on first failure
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(command: list[str], description: str) -> tuple[int, str]:
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
            timeout=900,  # 15 minute timeout for CI
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ {description} completed successfully ({duration:.1f}s)")
        else:
            print(f"❌ {description} failed ({duration:.1f}s)")
            if result.stderr:
                print("STDERR:", result.stderr[-500:])  # Last 500 chars

        return result.returncode, result.stdout

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return -1, "Timeout"
    except Exception as e:
        print(f"💥 {description} error: {e}")
        return -1, str(e)


def main():
    """Main CI test runner."""
    parser = argparse.ArgumentParser(description="DocMind AI CI Test Runner")
    parser.add_argument(
        "--unit-only", action="store_true", help="Run only unit tests (fastest)"
    )
    parser.add_argument(
        "--coverage", action="store_true", help="Generate coverage report"
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first test failure"
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent

    print("🏗️  DocMind AI CI Test Suite")
    print("=" * 50)
    print("Target: Fast feedback for continuous integration")
    print("Scope: Unit + Integration tests (no GPU required)")

    # Change to project root
    import os

    os.chdir(project_root)

    exit_codes = []

    # Run import validation first
    print("\n🔍 Step 1: Import Validation")
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--validate-imports"]
    exit_code, _ = run_command(cmd, "Import Validation")
    exit_codes.append(exit_code)

    if exit_code != 0 and args.fail_fast:
        print("❌ Import validation failed. Stopping CI execution.")
        sys.exit(1)

    # Run unit tests
    print("\n🧪 Step 2: Unit Tests (Tier 1)")
    cmd = ["uv", "run", "python", "scripts/run_tests.py", "--unit"]
    if args.coverage:
        cmd.append("--coverage")

    exit_code, _ = run_command(cmd, "Unit Tests")
    exit_codes.append(exit_code)

    if exit_code != 0 and args.fail_fast:
        print("❌ Unit tests failed. Stopping CI execution.")
        sys.exit(1)

    # Run integration tests (unless unit-only mode)
    if not args.unit_only:
        print("\n⚙️ Step 3: Integration Tests (Tier 2)")
        cmd = ["uv", "run", "python", "scripts/run_tests.py", "--integration"]

        exit_code, _ = run_command(cmd, "Integration Tests")
        exit_codes.append(exit_code)

        if exit_code != 0 and args.fail_fast:
            print("❌ Integration tests failed. Stopping CI execution.")
            sys.exit(1)

    # Summary
    total_failures = sum(1 for code in exit_codes if code != 0)

    print("\n" + "=" * 60)
    print("📋 CI TEST SUMMARY")
    print("=" * 60)

    if total_failures == 0:
        print("✅ All CI tests passed! Ready for merge.")
        print("💡 Next: Run staging tests before deployment")
        sys.exit(0)
    else:
        print(f"❌ {total_failures} test suite(s) failed")
        print("🔧 Fix issues before merging")

        # Show which steps failed
        step_names = ["Import Validation", "Unit Tests", "Integration Tests"]
        for i, code in enumerate(exit_codes):
            if i < len(step_names):
                status = "✅" if code == 0 else "❌"
                print(f"   {status} {step_names[i]}")

        sys.exit(1)


if __name__ == "__main__":
    main()
