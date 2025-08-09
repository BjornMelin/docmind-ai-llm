#!/usr/bin/env python
"""Optimize test markers and categorize tests for DocMind AI.

This script analyzes test files and adds appropriate pytest markers to categorize
tests by speed, resource requirements, and test type. This enables selective
test execution and better CI/CD pipeline organization.

Usage:
    python optimize_test_markers.py            # Analyze and suggest markers
    python optimize_test_markers.py --apply    # Apply marker suggestions
    python optimize_test_markers.py --check    # Check existing markers
"""

import argparse
import ast
from pathlib import Path


class TestMarkerOptimizer:
    """Analyze and optimize pytest markers for test categorization."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"

        # Marker categories and their criteria
        self.marker_criteria = {
            "slow": [
                "download",
                "model",
                "embedding",
                "gpu",
                "network",
                "splade",
                "bge",
                "multimodal",
                "whisper",
                "time.sleep",
                "requests.",
                "httpx",
                "subprocess.run",
                "torch.cuda",
                "load_model",
            ],
            "integration": [
                "workflow",
                "end_to_end",
                "pipeline",
                "agent_system",
                "load_documents",
                "build_index",
                "create_agent",
                "process_query",
                "setup_hybrid",
            ],
            "performance": [
                "benchmark",
                "timing",
                "memory",
                "cpu",
                "speed",
                "latency",
                "throughput",
                "concurrent",
                "batch",
                "profile",
            ],
            "requires_gpu": [
                "cuda",
                "gpu",
                "torch.compile",
                "CUDAExecutionProvider",
                "gpu_acceleration",
                "torch.cuda",
                "nvidia",
            ],
            "requires_network": [
                "requests",
                "httpx",
                "download",
                "http://",
                "https://",
                "api",
                "network",
                "internet",
                "remote",
            ],
            "requires_models": [
                "load_model",
                "download_model",
                "huggingface",
                "transformers",
                "sentence_transformers",
                "embed_model",
                "fastembed",
            ],
            "unit": [
                "mock",
                "MagicMock",
                "patch",
                "unittest.mock",
                "@patch",
                "test_.*_creation",
                "test_.*_validation",
            ],
            "smoke": ["health_check", "import", "basic", "simple", "smoke"],
        }

    def analyze_test_file(self, test_file: Path) -> dict[str, set[str]]:
        """Analyze a test file and suggest appropriate markers."""
        try:
            content = test_file.read_text()
            tree = ast.parse(content)
        except Exception as e:
            print(f"⚠️  Could not parse {test_file}: {e}")
            return {}

        file_markers = {}

        # Find all test functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                markers = self._analyze_test_function(node, content)
                if markers:
                    file_markers[node.name] = markers

            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                class_markers = self._analyze_test_class(node, content)
                if class_markers:
                    file_markers[node.name] = class_markers

        return file_markers

    def _analyze_test_function(self, node: ast.FunctionDef, content: str) -> set[str]:
        """Analyze a test function and suggest markers."""
        markers = set()

        # Get function source
        try:
            func_start = node.lineno
            func_end = (
                node.end_lineno if hasattr(node, "end_lineno") else func_start + 20
            )
            func_lines = content.split("\n")[func_start - 1 : func_end]
            func_content = "\n".join(func_lines)
        except:
            func_content = ""

        # Check for existing markers
        existing_markers = self._extract_existing_markers(node)

        # Analyze function content for marker criteria
        for marker, keywords in self.marker_criteria.items():
            if marker in existing_markers:
                continue  # Skip if already marked

            for keyword in keywords:
                if keyword in func_content.lower() or keyword in node.name.lower():
                    markers.add(marker)
                    break

        # Special logic for specific patterns
        if "patch(" in func_content or "@patch" in func_content:
            markers.add("unit")

        if "time.sleep" in func_content or "asyncio.sleep" in func_content:
            markers.add("slow")

        if "subprocess" in func_content or "os.system" in func_content:
            markers.add("slow")

        if "memory" in node.name.lower() or "performance" in node.name.lower():
            markers.add("performance")

        if "integration" in node.name.lower() or "workflow" in node.name.lower():
            markers.add("integration")

        return markers

    def _analyze_test_class(self, node: ast.ClassDef, content: str) -> set[str]:
        """Analyze a test class and suggest markers."""
        markers = set()

        # Get class docstring and name
        class_name = node.name.lower()

        # Check class name for patterns
        if "integration" in class_name:
            markers.add("integration")
        if "performance" in class_name:
            markers.add("performance")
        if "gpu" in class_name:
            markers.add("requires_gpu")
        if "slow" in class_name:
            markers.add("slow")
        if "multimodal" in class_name:
            markers.add("slow")
        if "embedding" in class_name:
            markers.add("slow")

        # Check docstring
        if node.body and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Str):
                docstring = node.body[0].value.s.lower()

                for marker, keywords in self.marker_criteria.items():
                    for keyword in keywords:
                        if keyword in docstring:
                            markers.add(marker)
                            break

        return markers

    def _extract_existing_markers(self, node: ast.FunctionDef) -> set[str]:
        """Extract existing pytest markers from a test function."""
        markers = set()

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Attribute):
                if (
                    isinstance(decorator.value, ast.Name)
                    and decorator.value.id == "pytest"
                    and decorator.attr == "mark"
                ):
                    # This is pytest.mark.something
                    continue

            elif isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute) and isinstance(
                    decorator.func.value, ast.Attribute
                ):
                    if (
                        decorator.func.value.value.id == "pytest"
                        and decorator.func.value.attr == "mark"
                    ):
                        markers.add(decorator.func.attr)

        return markers

    def generate_marker_suggestions(self) -> dict[str, dict[str, set[str]]]:
        """Generate marker suggestions for all test files."""
        suggestions = {}

        test_files = list(self.tests_dir.glob("test_*.py"))

        for test_file in test_files:
            file_suggestions = self.analyze_test_file(test_file)
            if file_suggestions:
                suggestions[str(test_file)] = file_suggestions

        return suggestions

    def print_analysis_report(self) -> None:
        """Print comprehensive analysis report."""
        print("🔍 TEST MARKER ANALYSIS REPORT")
        print("=" * 60)

        suggestions = self.generate_marker_suggestions()

        # Count markers across all tests
        marker_counts = {}
        test_counts = {}

        for file_path, file_suggestions in suggestions.items():
            filename = Path(file_path).name
            test_counts[filename] = len(file_suggestions)

            for test_name, markers in file_suggestions.items():
                for marker in markers:
                    marker_counts[marker] = marker_counts.get(marker, 0) + 1

        # Print summary
        total_tests = sum(test_counts.values())
        print("📊 ANALYSIS SUMMARY:")
        print(f"   📄 Test files analyzed: {len(suggestions)}")
        print(f"   🧪 Total tests found: {total_tests}")
        print(
            f"   🏷️  Marker suggestions generated: {sum(len(m) for m in marker_counts.values() if isinstance(m, int))}"
        )

        # Print marker distribution
        if marker_counts:
            print("\n🏷️  SUGGESTED MARKERS:")
            print("-" * 40)

            for marker, count in sorted(
                marker_counts.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_tests) * 100
                print(f"   {marker}: {count} tests ({percentage:.1f}%)")

        # Print file-by-file analysis
        print("\n📋 DETAILED FILE ANALYSIS:")
        print("-" * 60)

        for file_path, file_suggestions in suggestions.items():
            filename = Path(file_path).name
            print(f"\n📄 {filename}: {len(file_suggestions)} tests")

            # Group by markers
            markers_in_file = {}
            for test_name, markers in file_suggestions.items():
                for marker in markers:
                    if marker not in markers_in_file:
                        markers_in_file[marker] = []
                    markers_in_file[marker].append(test_name)

            for marker, tests in sorted(markers_in_file.items()):
                print(f"   🏷️  {marker}: {len(tests)} tests")
                if len(tests) <= 3:
                    for test in tests:
                        print(f"      - {test}")
                else:
                    for test in tests[:2]:
                        print(f"      - {test}")
                    print(f"      ... and {len(tests) - 2} more")

    def apply_markers(self, dry_run: bool = True) -> None:
        """Apply marker suggestions to test files."""
        suggestions = self.generate_marker_suggestions()

        print(
            f"{'🔍 DRY RUN: ' if dry_run else '✏️  APPLYING: '}Test Marker Optimization"
        )
        print("=" * 60)

        for file_path, file_suggestions in suggestions.items():
            filename = Path(file_path).name

            if dry_run:
                print(f"\n📄 Would modify {filename}:")
                for test_name, markers in file_suggestions.items():
                    markers_str = ", ".join(sorted(markers))
                    print(f"   {test_name}: +{markers_str}")
            else:
                self._apply_markers_to_file(Path(file_path), file_suggestions)
                print(f"✅ Modified {filename}")

    def _apply_markers_to_file(
        self, file_path: Path, suggestions: dict[str, set[str]]
    ) -> None:
        """Apply markers to a specific test file."""
        content = file_path.read_text()
        lines = content.split("\n")

        try:
            tree = ast.parse(content)
        except:
            print(f"⚠️  Could not parse {file_path}, skipping")
            return

        # Find test functions and their line numbers
        test_locations = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                if node.name in suggestions:
                    test_locations[node.lineno] = (node.name, suggestions[node.name])

        # Apply markers in reverse order to preserve line numbers
        for line_no in sorted(test_locations.keys(), reverse=True):
            test_name, markers = test_locations[line_no]

            # Create marker decorators
            marker_lines = []
            for marker in sorted(markers):
                if marker in [
                    "slow",
                    "integration",
                    "performance",
                    "unit",
                    "smoke",
                ] or marker.startswith("requires_"):
                    marker_lines.append(f"    @pytest.mark.{marker}")

            if marker_lines:
                # Find the function definition line
                func_line = line_no - 1

                # Insert markers before the function
                for i, marker_line in enumerate(reversed(marker_lines)):
                    lines.insert(func_line, marker_line)

        # Write back the modified content
        modified_content = "\n".join(lines)
        file_path.write_text(modified_content)

    def check_existing_markers(self) -> None:
        """Check and report existing markers in test files."""
        print("🔍 EXISTING MARKER ANALYSIS")
        print("=" * 60)

        test_files = list(self.tests_dir.glob("test_*.py"))
        marker_usage = {}
        total_tests = 0
        marked_tests = 0

        for test_file in test_files:
            try:
                content = test_file.read_text()
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith(
                        "test_"
                    ):
                        total_tests += 1
                        existing_markers = self._extract_existing_markers(node)

                        if existing_markers:
                            marked_tests += 1
                            for marker in existing_markers:
                                marker_usage[marker] = marker_usage.get(marker, 0) + 1

            except Exception as e:
                print(f"⚠️  Error analyzing {test_file}: {e}")

        print("📊 CURRENT MARKER USAGE:")
        print(f"   🧪 Total tests: {total_tests}")
        print(
            f"   🏷️  Tests with markers: {marked_tests} ({marked_tests / total_tests * 100:.1f}%)"
        )
        print(f"   📋 Unmarked tests: {total_tests - marked_tests}")

        if marker_usage:
            print("\n🏷️  MARKERS IN USE:")
            print("-" * 30)
            for marker, count in sorted(
                marker_usage.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_tests) * 100
                print(f"   {marker}: {count} tests ({percentage:.1f}%)")

        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 30)

        unmarked_percentage = ((total_tests - marked_tests) / total_tests) * 100

        if unmarked_percentage > 80:
            print(
                "   🚨 URGENT: Most tests lack markers. Run with --apply to add markers."
            )
        elif unmarked_percentage > 50:
            print(
                "   ⚠️  Many tests need markers. Consider running marker optimization."
            )
        elif unmarked_percentage > 20:
            print("   📈 Good marker coverage. Review and add missing markers.")
        else:
            print("   ✅ Excellent marker coverage!")

        print("   📝 Run: python optimize_test_markers.py --apply")
        print("   🧪 Fast tests: pytest -m 'not slow'")
        print("   ⚡ Unit tests: pytest -m unit")
        print("   🔄 Integration: pytest -m integration")


def main():
    """Main entry point for test marker optimizer."""
    parser = argparse.ArgumentParser(description="DocMind AI Test Marker Optimizer")
    parser.add_argument(
        "--apply", action="store_true", help="Apply marker suggestions to test files"
    )
    parser.add_argument("--check", action="store_true", help="Check existing markers")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without applying",
    )

    args = parser.parse_args()

    project_root = Path(__file__).parent
    optimizer = TestMarkerOptimizer(project_root)

    if args.check:
        optimizer.check_existing_markers()
    elif args.apply:
        optimizer.apply_markers(dry_run=False)
    elif args.dry_run:
        optimizer.apply_markers(dry_run=True)
    else:
        optimizer.print_analysis_report()


if __name__ == "__main__":
    main()
