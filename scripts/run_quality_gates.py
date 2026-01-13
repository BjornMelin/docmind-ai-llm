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
import os
import subprocess
import sys
from pathlib import Path

try:  # Python 3.11+
    import tomllib  # type: ignore[attr-defined]

    TOML_PARSER = tomllib
except ModuleNotFoundError:  # pragma: no cover - fallback for older Python
    try:
        import tomli

        TOML_PARSER = tomli
    except ModuleNotFoundError:  # pragma: no cover - last resort
        TOML_PARSER = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _read_thresholds_from_pyproject(project_root: Path) -> tuple[float, float]:
    """Read coverage thresholds from pyproject.

    Prefers [tool.pytest-quality] if available; falls back to
    [tool.coverage.report].fail_under for line coverage.
    Branch threshold falls back to 0.0 if not specified.
    """
    line = 0.0
    branch = 0.0
    pyproject = project_root / "pyproject.toml"
    if TOML_PARSER and pyproject.exists():
        try:
            with pyproject.open("rb") as f:
                data = TOML_PARSER.load(f)
            q = data.get("tool", {}).get("pytest-quality", {})
            line = float(q.get("min_line_coverage_percent", line))
            branch = float(q.get("min_branch_coverage_percent", branch))
            # Fallbacks
            if not line:
                cov = data.get("tool", {}).get("coverage", {})
                rep = cov.get("report", {})
                if isinstance(rep, dict):
                    line = float(rep.get("fail_under", line or 0.0))
        except (OSError, ValueError) as exc:  # pragma: no cover - non-fatal
            logging.debug("Failed to read thresholds from pyproject: %s", exc)
    # Env overrides win
    line = float(os.getenv("COVERAGE_LINE_THRESHOLD", line or 0.0))
    branch = float(os.getenv("COVERAGE_BRANCH_THRESHOLD", branch or 0.0))
    return line, branch


def _build_quality_scripts(project_root: Path) -> dict:
    """Build QUALITY_SCRIPTS dynamically using project thresholds/env."""
    line, branch = _read_thresholds_from_pyproject(project_root)
    cov_args = [
        "--collect" if not (project_root / "coverage.json").exists() else None,
        "--threshold",
        str(line or 0.0),
        "--branch-threshold",
        str(branch or 0.0),
        "--fail-under",
    ]
    cov_args = [a for a in cov_args if a is not None]

    return {
        "coverage": {
            "script": "scripts/check_coverage.py",
            # Assume coverage already collected by test runner; just validate thresholds
            "args": cov_args,
            "description": "Coverage threshold validation",
            "timeout": 900,
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
    "ci": ["coverage"],
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
        self, gate_name: str, additional_args: list[str] | None = None
    ) -> bool:
        """Run a single quality gate.

        Args:
            gate_name: Name of the quality gate to run
            additional_args: Additional arguments for the script

        Returns:
            True if quality gate passed, False otherwise
        """
        scripts = _build_quality_scripts(Path.cwd())
        if gate_name not in scripts:
            self.failures.append(f"Unknown quality gate: {gate_name}")
            return False

        gate_config = scripts[gate_name]
        script_path = Path(gate_config["script"])

        if not script_path.exists():
            self.failures.append(f"Quality gate script not found: {script_path}")
            return False

        # Build command (use uv to ensure project environment and deps)
        cmd = ["uv", "run", "python", str(script_path)]
        cmd.extend(gate_config["args"])
        if additional_args:
            cmd.extend(additional_args)

        logger.info("Running %s...", gate_config["description"])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
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
                logger.info("%s - PASSED", gate_config["description"])
                if self.verbose and result.stdout:
                    print(result.stdout)
            else:
                logger.error("%s - FAILED", gate_config["description"])
                self.failures.append(f"{gate_name}: {gate_config['description']}")
                if result.stdout:
                    print("STDOUT:", result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            error_msg = f"{gate_name} exceeded timeout of {gate_config['timeout']}s"
            self.failures.append(error_msg)
            logger.error("TIMEOUT: %s", error_msg)
            return False
        except (OSError, ValueError) as e:
            error_msg = f"Error running {gate_name}: {e}"
            self.failures.append(error_msg)
            logger.error("ERROR: %s", error_msg)
            return False

    def run_quality_suite(
        self, suite_name: str, additional_args: list[str] | None = None
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
        logger.info("Running %s quality suite (%d gates)...", suite_name, len(gates))

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
                check=False,
                timeout=60,
            )

            # Run hooks on all files
            result = subprocess.run(
                ["pre-commit", "run", "--all-files"],
                capture_output=True,
                text=True,
                check=False,
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
                logger.info("Pre-commit hooks - PASSED")
            else:
                logger.error("Pre-commit hooks - FAILED")
                self.failures.append("pre-commit: Pre-commit hooks validation")
                if result.stdout:
                    print("STDOUT:", result.stdout)

            return result.returncode == 0

        except subprocess.TimeoutExpired:
            error_msg = "Pre-commit hooks exceeded timeout of 600s"
            self.failures.append(error_msg)
            logger.error("TIMEOUT: %s", error_msg)
            return False
        except FileNotFoundError:
            error_msg = (
                "pre-commit command not found - install with: uv pip install pre-commit"
            )
            self.failures.append(error_msg)
            logger.error("ERROR: %s", error_msg)
            return False
        except (OSError, ValueError) as e:
            error_msg = f"Error running pre-commit hooks: {e}"
            self.failures.append(error_msg)
            logger.error("ERROR: %s", error_msg)
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
                f"  Passed:          {passed_gates}",
                f"  Failed:          {failed_gates}",
                f"  Overall Status:  {'PASS' if failed_gates == 0 else 'FAIL'}",
                "",
            ]
        )

        # Individual results
        if self.results:
            report_lines.extend(["GATE RESULTS:", ""])

            for gate_name, result in self.results.items():
                status = "PASS" if result["passed"] else "FAIL"
                report_lines.append(f"  {gate_name.upper()}: {status}")
                report_lines.append(f"    {result['description']}")
                report_lines.append(f"    Exit Code: {result['exit_code']}")
                report_lines.append("")

        # Failures
        if self.failures:
            report_lines.extend(
                [
                    "FAILURES:",
                    *[f"  - {failure}" for failure in self.failures],
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


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for quality gates."""
    parser = argparse.ArgumentParser(
        description="Run DocMind AI quality gates",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
    return parser


def _select_gates(args: argparse.Namespace) -> list[str]:
    """Resolve which quality gates to run based on CLI args."""
    gates: list[str] = []
    if args.all:
        gates.extend(QUALITY_SUITES["comprehensive"])
    elif args.quick:
        gates.extend(QUALITY_SUITES["quick"])
    elif args.ci:
        gates.extend(QUALITY_SUITES["ci"])
    else:
        if args.coverage:
            gates.append("coverage")
        if args.performance:
            gates.append("performance")
        if args.health:
            gates.append("health")
    if not gates and not args.pre_commit:
        gates.extend(QUALITY_SUITES["quick"])
    return gates


def _run_gates(
    runner: QualityGateRunner, gates: list[str], continue_on_failure: bool
) -> bool:
    """Run selected gates and return overall success."""
    overall_success = True
    for gate_name in gates:
        success = runner.run_quality_gate(gate_name)
        if success:
            continue
        overall_success = False
        if not continue_on_failure:
            break
    return overall_success


def _run_pre_commit(runner: QualityGateRunner, enabled: bool) -> bool:
    """Run pre-commit hooks when enabled."""
    if not enabled:
        return True
    return runner.run_pre_commit_hooks()


def _print_report(runner: QualityGateRunner) -> None:
    """Print the aggregate quality gate report."""
    report = runner.generate_report()
    print("\n" + report)


def _print_final_status(runner: QualityGateRunner, overall_success: bool) -> None:
    """Print the final summary banner for quality gates."""
    if overall_success:
        print("\nOK: All quality gates PASSED!")
        return
    print(f"\nFAIL: Quality gates FAILED ({len(runner.failures)} issues)")
    for failure in runner.failures:
        print(f"  - {failure}")


def main() -> int:
    """Main entry point for quality gates runner."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    runner = QualityGateRunner(verbose=args.verbose)

    try:
        gates = _select_gates(args)
        overall_success = _run_gates(runner, gates, args.continue_on_failure)
        if not _run_pre_commit(runner, args.pre_commit):
            overall_success = False
        if args.report or not overall_success:
            _print_report(runner)
        _print_final_status(runner, overall_success)
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Unexpected error during quality gate execution")
        print(f"\nERROR: Unexpected error: {e}")
        return 2

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
