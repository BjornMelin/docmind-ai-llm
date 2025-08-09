#!/usr/bin/env python
"""Coverage analysis and gap identification for DocMind AI.

This script provides detailed analysis of test coverage data, identifying
coverage gaps, low-coverage files, and providing actionable recommendations
for improving test coverage.

Usage:
    python analyze_coverage.py
    python analyze_coverage.py --detailed
    python analyze_coverage.py --critical-only
    python analyze_coverage.py --export-report coverage_analysis.json
"""

import argparse
import json
from pathlib import Path


class CoverageAnalyzer:
    """Analyze test coverage data and provide detailed insights."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.coverage_data = None

    def load_coverage_data(self) -> bool:
        """Load coverage data from coverage.json."""
        coverage_file = self.project_root / "coverage.json"

        if not coverage_file.exists():
            print("❌ coverage.json not found. Run tests with coverage first:")
            print("   python run_tests.py --coverage")
            print("   uv run pytest --cov=. --cov-report=json")
            return False

        try:
            with coverage_file.open() as f:
                self.coverage_data = json.load(f)
            print(f"✅ Loaded coverage data from {coverage_file}")
            return True
        except Exception as e:
            print(f"❌ Error loading coverage data: {e}")
            return False

    def get_overall_stats(self) -> dict[str, float]:
        """Calculate overall coverage statistics."""
        if not self.coverage_data:
            return {}

        files = self.coverage_data.get("files", {})
        if not files:
            return {}

        total_statements = sum(f["summary"]["num_statements"] for f in files.values())
        covered_statements = sum(f["summary"]["covered_lines"] for f in files.values())
        missing_statements = sum(f["summary"]["missing_lines"] for f in files.values())

        overall_coverage = (
            (covered_statements / total_statements) * 100 if total_statements > 0 else 0
        )

        return {
            "total_statements": total_statements,
            "covered_statements": covered_statements,
            "missing_statements": missing_statements,
            "overall_coverage": overall_coverage,
            "total_files": len(files),
        }

    def get_file_coverage_analysis(
        self, min_coverage: float = 0.0
    ) -> list[tuple[str, dict]]:
        """Get detailed coverage analysis for each file."""
        if not self.coverage_data:
            return []

        files = self.coverage_data.get("files", {})
        analysis = []

        for filename, file_data in files.items():
            # Skip test files and cache directories
            if any(
                skip in filename
                for skip in [
                    "test_",
                    "__pycache__",
                    ".pyc",
                    "/tests/",
                    "/.pytest_cache/",
                ]
            ):
                continue

            summary = file_data["summary"]
            coverage_pct = summary["percent_covered"]

            if coverage_pct >= min_coverage:
                file_analysis = {
                    "filename": filename,
                    "coverage_percent": coverage_pct,
                    "num_statements": summary["num_statements"],
                    "covered_lines": summary["covered_lines"],
                    "missing_lines": summary["missing_lines"],
                    "excluded_lines": summary.get("excluded_lines", 0),
                    "missing_branches": summary.get("missing_branches", 0),
                    "covered_branches": summary.get("covered_branches", 0),
                }

                # Add missing line details if available
                if "missing_lines" in file_data:
                    file_analysis["missing_line_numbers"] = file_data["missing_lines"]

                analysis.append((filename, file_analysis))

        # Sort by coverage percentage (lowest first)
        return sorted(analysis, key=lambda x: x[1]["coverage_percent"])

    def identify_critical_files(self) -> dict[str, dict]:
        """Identify and analyze critical files that need high coverage."""
        critical_patterns = [
            "models.py",
            "utils/utils.py",
            "utils/document_loader.py",
            "utils/index_builder.py",
            "utils/model_manager.py",
            "utils/qdrant_utils.py",
            "agent_factory.py",
            "agents/agent_utils.py",
            "app.py",
        ]

        critical_files = {}

        if not self.coverage_data:
            return critical_files

        files = self.coverage_data.get("files", {})

        for pattern in critical_patterns:
            found = False
            for filename, file_data in files.items():
                if pattern in filename:
                    summary = file_data["summary"]
                    critical_files[filename] = {
                        "pattern": pattern,
                        "coverage_percent": summary["percent_covered"],
                        "num_statements": summary["num_statements"],
                        "missing_lines": summary["missing_lines"],
                        "priority": "HIGH"
                        if summary["percent_covered"] < 80
                        else "MEDIUM"
                        if summary["percent_covered"] < 90
                        else "LOW",
                    }
                    found = True
                    break

            if not found:
                critical_files[pattern] = {
                    "pattern": pattern,
                    "coverage_percent": 0,
                    "num_statements": 0,
                    "missing_lines": 0,
                    "priority": "MISSING",
                    "status": "File not found in coverage data",
                }

        return critical_files

    def get_coverage_gaps(self, threshold: float = 80.0) -> list[dict]:
        """Identify significant coverage gaps."""
        gaps = []

        file_analysis = self.get_file_coverage_analysis()

        for filename, analysis in file_analysis:
            if analysis["coverage_percent"] < threshold:
                gap_severity = (
                    "CRITICAL"
                    if analysis["coverage_percent"] < 50
                    else "HIGH"
                    if analysis["coverage_percent"] < 70
                    else "MEDIUM"
                )

                gaps.append(
                    {
                        "filename": filename,
                        "coverage_percent": analysis["coverage_percent"],
                        "gap_percent": threshold - analysis["coverage_percent"],
                        "missing_lines": analysis["missing_lines"],
                        "severity": gap_severity,
                        "estimated_tests_needed": max(
                            1, analysis["missing_lines"] // 10
                        ),  # Rough estimate
                    }
                )

        return sorted(gaps, key=lambda x: x["gap_percent"], reverse=True)

    def generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations for improving coverage."""
        recommendations = []

        overall_stats = self.get_overall_stats()
        if not overall_stats:
            return ["Run tests with coverage to generate recommendations"]

        overall_coverage = overall_stats["overall_coverage"]

        # Overall coverage recommendations
        if overall_coverage < 60:
            recommendations.append(
                "🚨 URGENT: Overall coverage is critically low. Focus on unit tests for core modules."
            )
        elif overall_coverage < 80:
            recommendations.append(
                "⚠️  Overall coverage needs improvement. Target 80%+ for production readiness."
            )
        elif overall_coverage < 90:
            recommendations.append(
                "📈 Good coverage base. Focus on edge cases and integration scenarios."
            )
        else:
            recommendations.append(
                "✅ Excellent coverage! Maintain quality and add tests for new features."
            )

        # Critical file recommendations
        critical_files = self.identify_critical_files()
        high_priority_files = [
            (f, data)
            for f, data in critical_files.items()
            if data["priority"] in ["HIGH", "MISSING"]
        ]

        if high_priority_files:
            recommendations.append(
                f"🎯 Critical files needing attention: {len(high_priority_files)} files"
            )
            for filename, data in high_priority_files[:3]:  # Top 3
                if data["priority"] == "MISSING":
                    recommendations.append(
                        f"   📝 {data['pattern']}: File not covered by tests"
                    )
                else:
                    recommendations.append(
                        f"   📝 {filename}: {data['coverage_percent']:.1f}% coverage"
                    )

        # Coverage gap recommendations
        gaps = self.get_coverage_gaps(80.0)
        critical_gaps = [g for g in gaps if g["severity"] == "CRITICAL"]

        if critical_gaps:
            recommendations.append(
                f"🔥 {len(critical_gaps)} files have critical coverage gaps (<50%)"
            )
            for gap in critical_gaps[:2]:  # Top 2 critical gaps
                recommendations.append(
                    f"   🚨 {gap['filename']}: {gap['coverage_percent']:.1f}% - needs {gap['estimated_tests_needed']} more tests"
                )

        # Specific improvement areas
        if overall_stats["missing_statements"] > 100:
            recommendations.append(
                "📚 Focus on testing utility functions and error handling paths"
            )

        recommendations.append(
            "🧪 Consider adding property-based tests with hypothesis library"
        )
        recommendations.append("🔄 Add integration tests for end-to-end workflows")
        recommendations.append("⚡ Add performance tests for critical paths")

        return recommendations

    def print_summary(self, detailed: bool = False) -> None:
        """Print comprehensive coverage summary."""
        print("\n" + "=" * 80)
        print("📊 COVERAGE ANALYSIS REPORT")
        print("=" * 80)

        overall_stats = self.get_overall_stats()
        if not overall_stats:
            print("❌ No coverage data available")
            return

        # Overall statistics
        print(f"📈 OVERALL COVERAGE: {overall_stats['overall_coverage']:.1f}%")
        print(f"   📄 Total files analyzed: {overall_stats['total_files']}")
        print(f"   📝 Total statements: {overall_stats['total_statements']}")
        print(f"   ✅ Covered statements: {overall_stats['covered_statements']}")
        print(f"   ❌ Missing statements: {overall_stats['missing_statements']}")

        # Coverage status indicator
        coverage = overall_stats["overall_coverage"]
        if coverage >= 90:
            status = "🟢 EXCELLENT"
        elif coverage >= 80:
            status = "🟡 GOOD"
        elif coverage >= 70:
            status = "🟠 FAIR"
        elif coverage >= 60:
            status = "🔴 POOR"
        else:
            status = "💀 CRITICAL"

        print(f"   📊 Status: {status}")

        # Critical files analysis
        print("\n🎯 CRITICAL FILES COVERAGE:")
        print("-" * 60)

        critical_files = self.identify_critical_files()
        for filename, data in critical_files.items():
            if data["priority"] == "MISSING":
                print(f"   ❓ {data['pattern']}: Not found in coverage")
            else:
                priority_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "✅"}
                icon = priority_icon.get(data["priority"], "❓")
                print(f"   {icon} {filename}: {data['coverage_percent']:.1f}%")

        # Coverage gaps
        gaps = self.get_coverage_gaps(80.0)
        if gaps:
            print("\n🔍 COVERAGE GAPS (<80%):")
            print("-" * 60)

            for gap in gaps[:10]:  # Top 10 gaps
                severity_icon = {"CRITICAL": "🚨", "HIGH": "🔴", "MEDIUM": "🟠"}
                icon = severity_icon.get(gap["severity"], "❓")
                print(
                    f"   {icon} {gap['filename']}: {gap['coverage_percent']:.1f}% "
                    f"(missing {gap['missing_lines']} lines)"
                )

        # Detailed file analysis
        if detailed:
            print("\n📋 DETAILED FILE ANALYSIS:")
            print("-" * 60)

            file_analysis = self.get_file_coverage_analysis()
            for filename, analysis in file_analysis:
                if analysis["num_statements"] > 5:  # Skip tiny files
                    print(f"   📄 {filename}")
                    print(f"      Coverage: {analysis['coverage_percent']:.1f}%")
                    print(
                        f"      Lines: {analysis['covered_lines']}/{analysis['num_statements']}"
                    )
                    if analysis["missing_lines"] > 0:
                        print(f"      Missing: {analysis['missing_lines']} lines")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 60)

        recommendations = self.generate_recommendations()
        for rec in recommendations:
            print(f"   {rec}")

        # Next steps
        print("\n🚀 NEXT STEPS:")
        print("-" * 60)
        print("   1. Run: python run_tests.py --unit  # Focus on unit test coverage")
        print("   2. Add tests for files with <80% coverage")
        print("   3. Run: python run_tests.py --integration  # Test workflows")
        print("   4. Review missing lines and add targeted tests")
        print("   5. Run: python analyze_coverage.py --detailed  # Monitor progress")

        print("\n📄 Coverage report: htmlcov/index.html")
        print("📊 Coverage data: coverage.json")

    def export_report(self, output_file: Path) -> bool:
        """Export detailed coverage report to JSON."""
        try:
            overall_stats = self.get_overall_stats()
            critical_files = self.identify_critical_files()
            gaps = self.get_coverage_gaps(80.0)
            recommendations = self.generate_recommendations()
            file_analysis = self.get_file_coverage_analysis()

            report = {
                "timestamp": "2025-01-28",  # Current date
                "overall_stats": overall_stats,
                "critical_files": critical_files,
                "coverage_gaps": gaps,
                "recommendations": recommendations,
                "file_analysis": {
                    filename: analysis for filename, analysis in file_analysis
                },
                "summary": {
                    "total_files_analyzed": len(file_analysis),
                    "critical_gaps": len(
                        [g for g in gaps if g["severity"] == "CRITICAL"]
                    ),
                    "files_below_80_percent": len(
                        [g for g in gaps if g["coverage_percent"] < 80]
                    ),
                    "average_coverage": sum(
                        a["coverage_percent"] for _, a in file_analysis
                    )
                    / len(file_analysis)
                    if file_analysis
                    else 0,
                },
            }

            with output_file.open("w") as f:
                json.dump(report, f, indent=2)

            print(f"✅ Coverage report exported to {output_file}")
            return True

        except Exception as e:
            print(f"❌ Error exporting report: {e}")
            return False


def main():
    """Main entry point for coverage analyzer."""
    parser = argparse.ArgumentParser(description="DocMind AI Coverage Analyzer")
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed file analysis"
    )
    parser.add_argument(
        "--critical-only", action="store_true", help="Show only critical files"
    )
    parser.add_argument(
        "--export-report", type=str, help="Export detailed report to JSON file"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="Coverage threshold for gaps (default: 80%)",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent
    analyzer = CoverageAnalyzer(project_root)

    if not analyzer.load_coverage_data():
        return 1

    if args.critical_only:
        print("🎯 CRITICAL FILES COVERAGE ANALYSIS")
        print("=" * 50)
        critical_files = analyzer.identify_critical_files()
        for filename, data in critical_files.items():
            if data["priority"] in ["HIGH", "MISSING"]:
                status = (
                    "❌"
                    if data["priority"] == "MISSING"
                    else f"{data['coverage_percent']:.1f}%"
                )
                print(f"   {filename}: {status}")

    elif args.export_report:
        output_file = Path(args.export_report)
        analyzer.export_report(output_file)

    else:
        analyzer.print_summary(detailed=args.detailed)

    return 0


if __name__ == "__main__":
    exit(main())
