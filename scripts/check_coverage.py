#!/usr/bin/env python3
"""Coverage Threshold Checker for DocMind AI.

This module provides comprehensive coverage analysis and threshold enforcement:
- Coverage threshold validation (line and branch)
- Detailed coverage reports with failure analysis
- New code coverage tracking
- Integration with CI/CD pipelines
- HTML report generation with failure details

Usage:
    python scripts/check_coverage.py --threshold 80 --branch-threshold 75
    python scripts/check_coverage.py --report --html
    python scripts/check_coverage.py --new-code-only --diff-from main

Exit codes:
    0: Coverage meets thresholds
    1: Coverage below threshold or errors occurred
    2: Missing coverage data files
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default thresholds
DEFAULT_LINE_THRESHOLD = 80.0
DEFAULT_BRANCH_THRESHOLD = 75.0

# Coverage file locations
COVERAGE_JSON = Path("coverage.json")
COVERAGE_XML = Path("coverage.xml")
COVERAGE_HTML_DIR = Path("htmlcov")


class CoverageAnalyzer:
    """Comprehensive coverage analysis and threshold checking."""

    def __init__(
        self,
        line_threshold: float = DEFAULT_LINE_THRESHOLD,
        branch_threshold: float = DEFAULT_BRANCH_THRESHOLD,
    ):
        """Initialize coverage analyzer.

        Args:
            line_threshold: Minimum line coverage percentage required
            branch_threshold: Minimum branch coverage percentage required
        """
        self.line_threshold = line_threshold
        self.branch_threshold = branch_threshold
        self.coverage_data: dict[str, Any] = {}
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def load_coverage_data(self) -> bool:
        """Load coverage data from JSON file.

        Returns:
            True if coverage data loaded successfully, False otherwise
        """
        if not COVERAGE_JSON.exists():
            self.failures.append(f"Coverage JSON file not found: {COVERAGE_JSON}")
            return False

        try:
            with open(COVERAGE_JSON, encoding="utf-8") as f:
                self.coverage_data = json.load(f)
            logger.info("Coverage data loaded successfully")
            return True
        except (OSError, json.JSONDecodeError) as e:
            self.failures.append(f"Failed to load coverage data: {e}")
            return False

    def check_overall_coverage(self) -> dict[str, Any]:
        """Check overall coverage against thresholds.

        Returns:
            Dictionary with coverage analysis results
        """
        if not self.coverage_data:
            return {"status": "error", "message": "No coverage data available"}

        totals = self.coverage_data.get("totals", {})

        # Line coverage check
        line_percent = totals.get("percent_covered", 0.0)
        line_covered = totals.get("covered_lines", 0)
        line_total = totals.get("num_statements", 0)

        line_meets_threshold = line_percent >= self.line_threshold

        # Branch coverage check (if available)
        branch_percent = totals.get("percent_covered_branches", 0.0)
        branch_covered = totals.get("covered_branches", 0)
        branch_total = totals.get("num_branches", 0)

        branch_meets_threshold = True  # Default for when branch coverage not available
        if branch_total > 0:
            branch_meets_threshold = branch_percent >= self.branch_threshold

        # Overall status
        overall_pass = line_meets_threshold and branch_meets_threshold

        result = {
            "status": "pass" if overall_pass else "fail",
            "line_coverage": {
                "percent": line_percent,
                "covered": line_covered,
                "total": line_total,
                "threshold": self.line_threshold,
                "meets_threshold": line_meets_threshold,
            },
            "branch_coverage": {
                "percent": branch_percent,
                "covered": branch_covered,
                "total": branch_total,
                "threshold": self.branch_threshold,
                "meets_threshold": branch_meets_threshold,
            },
            "overall_pass": overall_pass,
        }

        # Record failures
        if not line_meets_threshold:
            self.failures.append(
                f"Line coverage {line_percent:.1f}% below threshold "
                f"{self.line_threshold}%"
            )

        if branch_total > 0 and not branch_meets_threshold:
            self.failures.append(
                f"Branch coverage {branch_percent:.1f}% below threshold "
                f"{self.branch_threshold}%"
            )

        return result

    def analyze_file_coverage(self, min_file_threshold: float = 70.0) -> dict[str, Any]:
        """Analyze coverage for individual files.

        Args:
            min_file_threshold: Minimum coverage threshold for individual files

        Returns:
            Dictionary with file coverage analysis
        """
        if not self.coverage_data:
            return {"status": "error", "message": "No coverage data available"}

        files = self.coverage_data.get("files", {})

        low_coverage_files = []
        high_coverage_files = []
        uncovered_files = []

        for filename, file_data in files.items():
            summary = file_data.get("summary", {})
            percent_covered = summary.get("percent_covered", 0.0)
            covered_lines = summary.get("covered_lines", 0)
            total_lines = summary.get("num_statements", 0)

            file_info = {
                "filename": filename,
                "percent_covered": percent_covered,
                "covered_lines": covered_lines,
                "total_lines": total_lines,
            }

            if percent_covered == 0.0 and total_lines > 0:
                uncovered_files.append(file_info)
            elif percent_covered < min_file_threshold:
                low_coverage_files.append(file_info)
            else:
                high_coverage_files.append(file_info)

        # Sort by coverage percentage
        low_coverage_files.sort(key=lambda x: x["percent_covered"])
        high_coverage_files.sort(key=lambda x: x["percent_covered"], reverse=True)

        # Record warnings for low coverage files
        for file_info in low_coverage_files:
            self.warnings.append(
                f"Low coverage in {file_info['filename']}: "
                f"{file_info['percent_covered']:.1f}% "
                f"({file_info['covered_lines']}/{file_info['total_lines']} lines)"
            )

        return {
            "total_files": len(files),
            "low_coverage_files": low_coverage_files,
            "high_coverage_files": high_coverage_files,
            "uncovered_files": uncovered_files,
            "files_below_threshold": len(low_coverage_files),
            "threshold_used": min_file_threshold,
        }

    def get_uncovered_lines_report(self) -> dict[str, list[int]]:
        """Get report of uncovered lines per file.

        Returns:
            Dictionary mapping filenames to lists of uncovered line numbers
        """
        if not self.coverage_data:
            return {}

        uncovered_lines = {}
        files = self.coverage_data.get("files", {})

        for filename, file_data in files.items():
            missing_lines = file_data.get("missing_lines", [])
            if missing_lines:
                uncovered_lines[filename] = missing_lines

        return uncovered_lines

    def generate_coverage_report(self, detailed: bool = False) -> str:
        """Generate a comprehensive coverage report.

        Args:
            detailed: Include detailed file-by-file analysis

        Returns:
            Formatted coverage report string
        """
        if not self.coverage_data:
            return "Error: No coverage data available"

        overall = self.check_overall_coverage()
        file_analysis = self.analyze_file_coverage()

        report_lines = [
            "=" * 60,
            "DOCMIND AI COVERAGE REPORT",
            "=" * 60,
            "",
            "OVERALL COVERAGE:",
            f"  Line Coverage:   {overall['line_coverage']['percent']:.1f}% "
            f"({overall['line_coverage']['covered']}/"
            f"{overall['line_coverage']['total']} lines)",
            f"  Threshold:       {overall['line_coverage']['threshold']}%",
            f"  Status:          "
            f"{'✓ PASS' if overall['line_coverage']['meets_threshold'] else '✗ FAIL'}",
            "",
        ]

        # Branch coverage (if available)
        if overall["branch_coverage"]["total"] > 0:
            status = (
                "✓ PASS" if overall["branch_coverage"]["meets_threshold"] else "✗ FAIL"
            )
            report_lines.extend(
                [
                    (
                        f"  Branch Coverage: "
                        f"{overall['branch_coverage']['percent']:.1f}% "
                        f"({overall['branch_coverage']['covered']}/"
                        f"{overall['branch_coverage']['total']} branches)"
                    ),
                    f"  Threshold:       {overall['branch_coverage']['threshold']}%",
                    f"  Status:          {status}",
                    "",
                ]
            )

        # Overall status
        report_lines.extend(
            [
                f"OVERALL STATUS: {'✓ PASS' if overall['overall_pass'] else '✗ FAIL'}",
                "",
            ]
        )

        # File analysis summary
        report_lines.extend(
            [
                "FILE ANALYSIS:",
                f"  Total Files:     {file_analysis['total_files']}",
                f"  Low Coverage:    {file_analysis['files_below_threshold']} files",
                f"  Uncovered Files: {len(file_analysis['uncovered_files'])} files",
                "",
            ]
        )

        # Failures
        if self.failures:
            report_lines.extend(
                [
                    "FAILURES:",
                    *[f"  ✗ {failure}" for failure in self.failures],
                    "",
                ]
            )

        # Warnings
        if self.warnings:
            report_lines.extend(
                [
                    "WARNINGS:",
                    *[f"  ! {warning}" for warning in self.warnings],
                    "",
                ]
            )

        # Detailed file analysis
        if detailed:
            if file_analysis["low_coverage_files"]:
                report_lines.extend(
                    [
                        "LOW COVERAGE FILES:",
                        *(
                            f"  {file_info['filename']}: "
                            f"{file_info['percent_covered']:.1f}% "
                            f"({file_info['covered_lines']}/"
                            f"{file_info['total_lines']})"
                            for file_info in file_analysis["low_coverage_files"][
                                :10
                            ]  # Top 10
                        ),
                        "",
                    ]
                )

            if file_analysis["uncovered_files"]:
                report_lines.extend(
                    [
                        "COMPLETELY UNCOVERED FILES:",
                        *(
                            f"  {file_info['filename']}: 0% "
                            f"(0/{file_info['total_lines']})"
                            for file_info in file_analysis["uncovered_files"][
                                :10
                            ]  # Top 10
                        ),
                        "",
                    ]
                )

        report_lines.append("=" * 60)
        return "\n".join(report_lines)

    def run_coverage_collection(self) -> bool:
        """Run pytest with coverage collection.

        Returns:
            True if coverage collection succeeded, False otherwise
        """
        try:
            logger.info("Running pytest with coverage collection...")

            cmd = [
                "uv",
                "run",
                "python",
                "-m",
                "pytest",
                "--cov=src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
                "--cov-report=json:coverage.json",
                "--cov-branch",
                "-q",  # Quiet mode for coverage collection
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                self.warnings.append(
                    f"Some tests failed during coverage collection: {result.stdout}"
                )
                # Continue anyway - we still want coverage data

            logger.info("Coverage collection completed")
            return True

        except subprocess.SubprocessError as e:
            self.failures.append(f"Failed to run coverage collection: {e}")
            return False

    def check_new_code_coverage(self, base_branch: str = "main") -> dict[str, Any]:
        """Check coverage for new/modified code only.

        Args:
            base_branch: Base branch to compare against

        Returns:
            Dictionary with new code coverage analysis
        """
        try:
            # Get list of modified files
            result = subprocess.run(
                ["git", "diff", "--name-only", f"{base_branch}...HEAD", "--", "*.py"],
                capture_output=True,
                text=True,
                check=True,
            )

            modified_files = [
                f
                for f in result.stdout.strip().split("\n")
                if f and not f.startswith("tests/")
            ]

            if not modified_files:
                return {
                    "status": "no_changes",
                    "message": "No Python files modified",
                    "files": [],
                }

            # Filter coverage data to only include modified files
            new_code_coverage = {}
            files = self.coverage_data.get("files", {})

            total_covered = 0
            total_lines = 0

            for file_path in modified_files:
                # Try different path variations
                for potential_path in [
                    file_path,
                    f"./{file_path}",
                    file_path.replace("/", "\\"),
                ]:
                    if potential_path in files:
                        file_data = files[potential_path]
                        summary = file_data.get("summary", {})

                        covered = summary.get("covered_lines", 0)
                        lines = summary.get("num_statements", 0)
                        percent = summary.get("percent_covered", 0.0)

                        new_code_coverage[potential_path] = {
                            "covered_lines": covered,
                            "total_lines": lines,
                            "percent_covered": percent,
                        }

                        total_covered += covered
                        total_lines += lines
                        break

            overall_percent = (
                (total_covered / total_lines * 100) if total_lines > 0 else 0.0
            )
            meets_threshold = overall_percent >= self.line_threshold

            return {
                "status": "pass" if meets_threshold else "fail",
                "overall_percent": overall_percent,
                "total_covered_lines": total_covered,
                "total_lines": total_lines,
                "threshold": self.line_threshold,
                "meets_threshold": meets_threshold,
                "files": new_code_coverage,
                "modified_files": modified_files,
            }

        except subprocess.CalledProcessError as e:
            self.failures.append(f"Failed to get modified files: {e}")
            return {"status": "error", "message": str(e)}
        except (OSError, ValueError, json.JSONDecodeError) as e:
            self.failures.append(f"Error checking new code coverage: {e}")
            return {"status": "error", "message": str(e)}


def main() -> int:
    """Main entry point for coverage threshold checking."""
    parser = argparse.ArgumentParser(
        description="Check coverage thresholds for DocMind AI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_LINE_THRESHOLD,
        help="Minimum line coverage threshold percentage",
    )

    parser.add_argument(
        "--branch-threshold",
        type=float,
        default=DEFAULT_BRANCH_THRESHOLD,
        help="Minimum branch coverage threshold percentage",
    )

    parser.add_argument(
        "--file-threshold",
        type=float,
        default=70.0,
        help="Minimum coverage threshold for individual files",
    )

    parser.add_argument(
        "--collect",
        action="store_true",
        help="Run coverage collection before checking thresholds",
    )

    parser.add_argument(
        "--report", action="store_true", help="Generate detailed coverage report"
    )

    parser.add_argument(
        "--html", action="store_true", help="Generate HTML coverage report"
    )

    parser.add_argument(
        "--new-code-only",
        action="store_true",
        help="Check coverage for new/modified code only",
    )

    parser.add_argument(
        "--diff-from", default="main", help="Base branch for new code comparison"
    )

    parser.add_argument(
        "--fail-under",
        action="store_true",
        help="Exit with error code if coverage below threshold",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    analyzer = CoverageAnalyzer(
        line_threshold=args.threshold, branch_threshold=args.branch_threshold
    )

    exit_code = 0

    try:
        # Collect coverage if requested
        if args.collect and not analyzer.run_coverage_collection():
            print("❌ Coverage collection failed")
            return 2

        # Load coverage data
        if not analyzer.load_coverage_data():
            print("❌ Failed to load coverage data")
            return 2

        # Check coverage thresholds
        if args.new_code_only:
            # Check new code coverage only
            new_code_result = analyzer.check_new_code_coverage(args.diff_from)

            if new_code_result["status"] == "error":
                print(
                    f"❌ Error checking new code coverage: {new_code_result['message']}"
                )
                exit_code = 1
            elif new_code_result["status"] == "no_changes":
                print(f"ℹ️  {new_code_result['message']}")
            else:
                meets_threshold = new_code_result["meets_threshold"]
                percent = new_code_result["overall_percent"]
                threshold = new_code_result["threshold"]

                status_icon = "✅" if meets_threshold else "❌"
                print(
                    f"{status_icon} New code coverage: {percent:.1f}% "
                    f"(threshold: {threshold}%)"
                )

                if not meets_threshold and args.fail_under:
                    exit_code = 1
        else:
            # Check overall coverage
            overall_result = analyzer.check_overall_coverage()

            if overall_result["status"] == "error":
                print(f"❌ {overall_result['message']}")
                exit_code = 2
            else:
                line_coverage = overall_result["line_coverage"]
                meets_threshold = overall_result["overall_pass"]

                status_icon = "✅" if meets_threshold else "❌"
                print(
                    f"{status_icon} Overall line coverage: "
                    f"{line_coverage['percent']:.1f}% "
                    f"(threshold: {line_coverage['threshold']}%)"
                )

                # Branch coverage output
                if overall_result["branch_coverage"]["total"] > 0:
                    branch_coverage = overall_result["branch_coverage"]
                    branch_icon = "✅" if branch_coverage["meets_threshold"] else "❌"
                    print(
                        f"{branch_icon} Branch coverage: "
                        f"{branch_coverage['percent']:.1f}% "
                        f"(threshold: {branch_coverage['threshold']}%)"
                    )

                if not meets_threshold and args.fail_under:
                    exit_code = 1

        # Generate detailed report if requested
        if args.report:
            report = analyzer.generate_coverage_report(detailed=True)
            print("\n" + report)

        # Generate HTML report if requested
        if args.html:
            if COVERAGE_HTML_DIR.exists():
                print(f"📊 HTML coverage report: {COVERAGE_HTML_DIR}/index.html")
            else:
                print("HTML coverage report not found. Run --collect to generate.")

        # Print any failures or warnings
        if analyzer.failures:
            print("\n❌ FAILURES:")
            for failure in analyzer.failures:
                print(f"  • {failure}")
            exit_code = 1

        if analyzer.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in analyzer.warnings:
                print(f"  • {warning}")

    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        logger.exception("Unexpected error during coverage checking")
        print(f"❌ Unexpected error: {e}")
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
