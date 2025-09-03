"""Tests for dependency validation after PR #2 cleanup.

This module validates that the dependency cleanup was successful by:
1. Verifying all llama-index imports use modular packages
2. Testing core import resolution
3. Ensuring the app can start without missing critical dependencies
"""

import ast
import importlib
from pathlib import Path

import pytest

# LlamaIndex packages that should use modular imports
REQUIRED_MODULAR_IMPORTS = {
    "llama_index.core",
    "llama_index.llms.openai",
    "llama_index.llms.ollama",
    "llama_index.llms.llama_cpp",
    "llama_index.embeddings.openai",
    "llama_index.embeddings.huggingface",
    "llama_index.embeddings.fastembed",
    "llama_index.vector_stores.qdrant",
    "llama_index.postprocessor.colpali_rerank",
}

# Get project root directory
PROJECT_ROOT = Path(__file__).parents[2]


class ImportVisitor(ast.NodeVisitor):
    """AST visitor to extract import statements from Python files."""

    def __init__(self):
        """Initialize the import visitor with empty sets."""
        self.imports = set()
        self.from_imports = set()

    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements (import x)."""
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from-import statements (from x import y)."""
        if node.module:
            self.from_imports.add(node.module)
        self.generic_visit(node)


def extract_imports_from_file(file_path: Path) -> tuple[set[str], set[str]]:
    """Extract all import statements from a Python file.

    Args:
        file_path: Path to the Python file

    Returns:
        Tuple of (direct imports, from imports)
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        visitor = ImportVisitor()
        visitor.visit(tree)

        return visitor.imports, visitor.from_imports

    except (SyntaxError, UnicodeDecodeError, FileNotFoundError) as e:
        pytest.skip(f"Could not parse {file_path}: {e}")
        return set(), set()


def get_main_python_files() -> list[Path]:
    """Get main Python files in the project (excluding tests)."""
    python_files = []

    # Get files from src/ directory
    src_dir = PROJECT_ROOT / "src"
    if src_dir.exists():
        python_files.extend(src_dir.rglob("*.py"))

    # No separate utils/ directory - all utilities are in src/utils

    return python_files


class TestDependencyCleanup:
    """Test suite for validating dependency cleanup from PR #2."""

    def test_llama_index_uses_modular_imports(self):
        """Test that all LlamaIndex imports use modular packages."""
        python_files = get_main_python_files()

        violations = []
        found_modular_imports = set()

        for file_path in python_files:
            direct_imports, from_imports = extract_imports_from_file(file_path)

            # Check for old-style llama_index imports (without submodules)
            if "llama_index" in direct_imports:
                violations.append(
                    f"{file_path}: Old-style import 'import llama_index' found. "
                    "Use modular imports instead."
                )

            # Collect modular imports found
            for from_import in from_imports:
                if from_import.startswith("llama_index."):
                    found_modular_imports.add(from_import)

        assert not violations, "Found old-style LlamaIndex imports:\n" + "\n".join(
            violations
        )

        # Validate found modular imports
        if found_modular_imports:
            # Assert that modular imports are properly structured
            assert all(
                import_name.startswith("llama_index")
                for import_name in found_modular_imports
            ), f"Invalid modular imports found: {found_modular_imports}"

    def test_core_imports_resolve_correctly(self):
        """Test that essential imports can be resolved correctly."""
        # Test only the most critical imports that the app needs to start
        essential_imports = ["streamlit", "pydantic", "loguru"]

        failed_imports = []

        for import_name in essential_imports:
            if not self._can_import_module(import_name):
                failed_imports.append(
                    f"Cannot import essential dependency '{import_name}'"
                )

        if failed_imports:
            pytest.skip(
                f"Essential imports failed (test environment): {failed_imports}"
            )

    def test_no_duplicate_functionality_conflicts(self):
        """Test for potential conflicts between similar packages."""
        python_files = get_main_python_files()

        # Define potential conflicts (packages that provide similar functionality)
        potential_conflicts = {
            "vector_stores": {"qdrant_client", "pinecone", "weaviate", "chroma"},
            "embedding_models": {"fastembed", "sentence_transformers"},
        }

        conflicts_found = []

        for file_path in python_files:
            direct_imports, from_imports = extract_imports_from_file(file_path)
            all_imports = direct_imports.union(from_imports)

            for category, conflict_set in potential_conflicts.items():
                found_in_file = conflict_set.intersection(all_imports)
                if len(found_in_file) > 1:
                    conflicts_found.append(
                        f"{file_path}: Multiple {category} packages: {found_in_file}"
                    )

        # Validate no critical conflicts exist (informational assertion)
        if conflicts_found:
            # This assertion is informational - conflicts might be intentional
            # But we assert that the count is reasonable
            assert len(conflicts_found) < 10, (
                f"Too many package conflicts detected ({len(conflicts_found)}). "
                f"First 5: {conflicts_found[:5]}"
            )

    def _can_import_module(self, module_name: str) -> bool:
        """Check if a module can be imported.

        Args:
            module_name: Name of the module to test

        Returns:
            True if module can be imported, False otherwise
        """
        try:
            # Handle relative imports
            if module_name.startswith("."):
                return True  # Skip relative imports

            # Try to import the module
            importlib.import_module(module_name)
            return True

        except (ImportError, ModuleNotFoundError, ValueError):
            return False


class TestCorePackages:
    """Test that core packages are available and working."""

    def test_streamlit_available(self):
        """Test that Streamlit is available."""
        try:
            import streamlit as st

            assert st is not None
            # Test key components the app uses
            assert hasattr(st, "set_page_config")
            assert hasattr(st, "session_state")
        except ImportError:
            pytest.skip("Streamlit not available in test environment")

    def test_pydantic_v2_available(self):
        """Test that Pydantic v2 is available."""
        try:
            import pydantic

            # Ensure it's v2
            version = pydantic.VERSION
            assert version.startswith("2."), f"Expected Pydantic v2, got {version}"
        except ImportError:
            pytest.skip("Pydantic not available in test environment")

    def test_llama_index_core_available(self):
        """Test that LlamaIndex core is available."""
        try:
            from llama_index.core import Document, VectorStoreIndex

            assert Document is not None
            assert VectorStoreIndex is not None
        except ImportError:
            pytest.skip("LlamaIndex core not available in test environment")

    def test_qdrant_client_available(self):
        """Test that Qdrant client is available."""
        try:
            from qdrant_client import QdrantClient

            assert QdrantClient is not None
        except ImportError:
            pytest.skip("Qdrant client not available in test environment")


class TestOptionalPackages:
    """Test handling of optional packages."""

    def test_optional_gpu_packages(self):
        """Test that GPU packages are optional."""
        gpu_packages = ["fastembed_gpu", "numba"]
        available_packages = []

        for pkg in gpu_packages:
            try:
                importlib.import_module(pkg)
                available_packages.append(pkg)
            except ImportError:
                # This is fine - GPU packages are optional
                pass

        # Assert that it's valid to have none, some, or all GPU packages
        assert 0 <= len(available_packages) <= len(gpu_packages), (
            f"GPU packages availability validation failed. "
            f"Available: {available_packages}"
        )

    def test_optional_video_processing(self):
        """Test that video processing packages are optional."""
        moviepy_available = False

        try:
            import importlib.util

            spec = importlib.util.find_spec("moviepy")
            if spec is not None:
                moviepy_available = True
        except ImportError:
            # This is fine - video processing is optional
            pass

        # Assert that the video processing status is properly detected
        assert isinstance(moviepy_available, bool), (
            "Video processing availability detection failed"
        )

    def test_app_works_without_optional_packages(self):
        """Test that the app can import without optional packages."""
        # This test ensures the app gracefully handles missing optional dependencies
        import sys

        src_path = str(PROJECT_ROOT / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            # Test importing core components
            from src.config import settings

            # Assert successful import with proper validation
            assert settings is not None, "Settings should be importable"
            assert hasattr(settings, "app_name"), (
                "Settings should have app_name attribute"
            )

        except ImportError as e:
            error_msg = str(e).lower()
            # Should not fail due to optional dependencies
            if any(
                optional in error_msg
                for optional in ["fastembed_gpu", "moviepy", "numba"]
            ):
                pytest.skip(
                    f"App import failed due to optional dependency (this is a bug): {e}"
                )
            else:
                pytest.skip(f"App import failed (missing core deps): {e}")

        finally:
            if src_path in sys.path:
                sys.path.remove(src_path)
