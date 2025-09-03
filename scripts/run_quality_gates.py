#!/usr/bin/env python3
"""Comprehensive Quality Gates Runner for DocMind AI.

This script provides a unified interface to run all quality gate checks:
- Coverage threshold validation
- Performance regression detection
- Test suite health monitoring
- Pre-commit hook validation

Usage:
    python scripts/run_quality_gates.py --all
    python scripts/run_quality_gates.py --coverage --performance
    python scripts/run_quality_gates.py --quick
    python scripts/run_quality_gates.py --ci

Exit codes:
    0: All quality gates passed
    1: One or more quality gates failed
    2: Error running quality gates
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Quality gate scripts
QUALITY_SCRIPTS = {
    "coverage": {
        "script": "scripts/check_coverage.py",
        "args": ["--threshold", "80", "--fail-under"],
        "description": "Coverage threshold validation",
        "timeout": 180,
    },
    "performance": {
        "script": "scripts/performance_monitor.py",
        "args": ["--collection-only", "--check-regressions"],
        "description": "Performance regression detection",
        "timeout": 300,
    },
    "health": {
        "script": "scripts/test_health.py",
        "args": ["--patterns", "--stability"],
        "description": "Test suite health monitoring",
        "timeout": 120,
    },
}

QUALITY_SUITES = {
    "quick": ["coverage"],
    "standard": ["coverage", "performance"],
    "comprehensive": ["coverage", "performance", "health"],
    "ci": ["coverage", "performance"],
}


class QualityGateRunner:
    """Orchestrates quality gate execution and reporting."""

    def __init__(self, verbose: bool = False):
        """Initialize quality gate runner.

        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.results: dict[str, dict] = {}
        self.failures: list[str] = []

        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
        )

    def run_quality_gate(
        self, gate_name: str, additional_args: list[str] = None
    ) -> bool:
        """Run a single quality gate.

        Args:
            gate_name: Name of the quality gate to run
            additional_args: Additional arguments for the script

        Returns:
            True if quality gate passed, False otherwise
        """
        if gate_name not in QUALITY_SCRIPTS:
            self.failures.append(f"Unknown quality gate: {gate_name}")
            return False

        gate_config = QUALITY_SCRIPTS[gate_name]
        script_path = Path(gate_config["script"])

        if not script_path.exists():
            self.failures.append(f"Quality gate script not found: {script_path}")
            return False

        # Build command
        cmd = ["python", str(script_path)]
        cmd.extend(gate_config["args"])
        if additional_args:
            cmd.extend(additional_args)

        logger.info(f"Running {gate_config['description']}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=gate_config["timeout"],
            )

            # Store result
            self.results[gate_name] = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "description": gate_config["description"],
                "passed": result.returncode == 0,
            }

            if result.returncode == 0:
                logger.info(f"✅ {gate_config['description']} - PASSED")
                if self.verbose and result.stdout:
                    print(result.stdout)
            else:
                logger.error(f"❌ {gate_config['description']} - FAILED")
                self.failures.append(f"{gate_name}: {gate_config['description']}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            error_msg = f"{gate_name} exceeded timeout of {gate_config['timeout']}s"
            self.failures.append(error_msg)
            logger.error(f"⏰ {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Error running {gate_name}: {e}"
            self.failures.append(error_msg)
            logger.error(f"💥 {error_msg}")
            return False

    def run_quality_suite(
        self, suite_name: str, additional_args: list[str] = None
    ) -> bool:
        """Run a predefined quality gate suite.

        Args:
            suite_name: Name of the quality suite
            additional_args: Additional arguments for all gates

        Returns:
            True if all gates in suite passed, False otherwise
        """
        if suite_name not in QUALITY_SUITES:
            self.failures.append(f"Unknown quality suite: {suite_name}")
            return False

        gates = QUALITY_SUITES[suite_name]
        logger.info(f"Running {suite_name} quality suite ({len(gates)} gates)...")

        all_passed = True
        for gate_name in gates:
            passed = self.run_quality_gate(gate_name, additional_args)
            if not passed:
                all_passed = False

        return all_passed

    def run_pre_commit_hooks(self) -> bool:
        """Run pre-commit hooks as quality gates.

        Returns:
            True if pre-commit hooks passed, False otherwise
        """
        if not Path(".pre-commit-config.yaml").exists():
            self.failures.append("Pre-commit configuration not found")
            return False

        logger.info("Running pre-commit hooks...")

        try:
            # Install hooks if needed
            subprocess.run(
                ["pre-commit", "install"],
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Run hooks on all files
            result = subprocess.run(
                ["pre-commit", "run", "--all-files"],
                capture_output=True,
                text=True,
                timeout=600,
            )

            self.results["pre-commit"] = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "description": "Pre-commit hooks validation",
                "passed": result.returncode == 0,
            }

            if result.returncode == 0:
                logger.info("✅ Pre-commit hooks - PASSED")
            else:
                logger.error("❌ Pre-commit hooks - FAILED")
                self.failures.append("pre-commit: Pre-commit hooks validation")
                if result.stdout:
                    print("STDOUT:", result.stdout)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            error_msg = "Pre-commit hooks exceeded timeout of 600s"
            self.failures.append(error_msg)
            logger.error(f"⏰ {error_msg}")
            return False
        except FileNotFoundError:
            error_msg = (
                "pre-commit command not found - install with: pip install pre-commit"
            )
            self.failures.append(error_msg)
            logger.error(f"💥 {error_msg}")
            return False
        except Exception as e:
            error_msg = f"Error running pre-commit hooks: {e}"
            self.failures.append(error_msg)
            logger.error(f"💥 {error_msg}")
            return False

    def generate_report(self) -> str:
        """Generate comprehensive quality gates report.

        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 70,
            "DOCMIND AI QUALITY GATES REPORT",
            "=" * 70,
            "",
        ]

        # Summary
        total_gates = len(self.results)
        passed_gates = sum(1 for result in self.results.values() if result["passed"])
        failed_gates = total_gates - passed_gates

        report_lines.extend(
            [
                "SUMMARY:",
                f"  Total Gates:     {total_gates}",
                f"  Passed:          {passed_gates} ✅",
                f"  Failed:          {failed_gates} ❌",
                f"  Overall Status:  {'PASS' if failed_gates == 0 else 'FAIL'}",
                "",
            ]
        )

        # Individual results
        if self.results:
            report_lines.extend(["GATE RESULTS:", ""])

            for gate_name, result in self.results.items():
                status = "✅ PASS" if result["passed"] else "❌ FAIL"
                report_lines.append(f"  {gate_name.upper()}: {status}")
                report_lines.append(f"    {result['description']}")
                report_lines.append(f"    Exit Code: {result['exit_code']}")
                report_lines.append("")

        # Failures
        if self.failures:
            report_lines.extend(
                [
                    "FAILURES:",
                    *[f"  • {failure}" for failure in self.failures],
                    "",
                ]
            )

        # Recommendations
        recommendations = self._generate_recommendations()
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

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on results.

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Coverage recommendations
        if "coverage" in self.results and not self.results["coverage"]["passed"]:
            recommendations.append("Increase test coverage to meet 80% threshold")

        # Performance recommendations
        if "performance" in self.results and not self.results["performance"]["passed"]:
            recommendations.append("Address performance regressions detected")

        # Health recommendations
        if "health" in self.results and not self.results["health"]["passed"]:
            recommendations.append(
                "Fix test suite health issues (flaky tests, anti-patterns)"
            )

        # Pre-commit recommendations
        if "pre-commit" in self.results and not self.results["pre-commit"]["passed"]:
            recommendations.append("Fix pre-commit hook violations before committing")

        return recommendations


def main() -> int:
    """Main entry point for quality gates runner."""
    parser = argparse.ArgumentParser(
        description="Run DocMind AI quality gates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Quality gate selection
    parser.add_argument(
        "--all",
        "--comprehensive",
        action="store_true",
        help="Run all quality gates (coverage, performance, health)",
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick quality gates (coverage only)"
    )

    parser.add_argument(
        "--ci", action="store_true", help="Run CI quality gates (coverage, performance)"
    )

    # Individual gates
    parser.add_argument(
        "--coverage", action="store_true", help="Run coverage threshold validation"
    )

    parser.add_argument(
        "--performance",
        action="store_true",
        help="Run performance regression detection",
    )

    parser.add_argument(
        "--health", action="store_true", help="Run test suite health monitoring"
    )

    parser.add_argument(
        "--pre-commit", action="store_true", help="Run pre-commit hooks"
    )

    # Configuration
    parser.add_argument(
        "--report", action="store_true", help="Generate detailed report"
    )

    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running gates even if one fails",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Initialize runner
    runner = QualityGateRunner(verbose=args.verbose)
    overall_success = True

    try:
        # Determine which gates to run
        gates_to_run = []

        if args.all:
            gates_to_run.extend(QUALITY_SUITES["comprehensive"])
        elif args.quick:
            gates_to_run.extend(QUALITY_SUITES["quick"])
        elif args.ci:
            gates_to_run.extend(QUALITY_SUITES["ci"])
        else:
            # Individual gate selection
            if args.coverage:
                gates_to_run.append("coverage")
            if args.performance:
                gates_to_run.append("performance")
            if args.health:
                gates_to_run.append("health")

        # Default to quick if nothing specified
        if not gates_to_run and not args.pre_commit:
            gates_to_run.extend(QUALITY_SUITES["quick"])

        # Run quality gates
        for gate_name in gates_to_run:
            success = runner.run_quality_gate(gate_name)
            if not success:
                overall_success = False
                if not args.continue_on_failure:
                    break

        # Run pre-commit if requested
        if args.pre_commit:
            success = runner.run_pre_commit_hooks()
            if not success:
                overall_success = False

        # Generate and display report
        if args.report or not overall_success:
            report = runner.generate_report()
            print("\n" + report)

        # Final status
        if overall_success:
            print("\n🎉 All quality gates PASSED!")
        else:
            print(f"\n💥 Quality gates FAILED ({len(runner.failures)} issues)")
            for failure in runner.failures:
                print(f"  • {failure}")

    except Exception as e:
        logger.exception("Unexpected error during quality gate execution")
        print(f"\n💥 Unexpected error: {e}")
        return 2

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
