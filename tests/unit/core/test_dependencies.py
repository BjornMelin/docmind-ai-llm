"""Tests for the canonical application dependency contract."""

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PROJECT_ROOT = Path(__file__).parents[3]
REQUIRED_PROJECT_DEPENDENCIES = {
    "docling",
    "fastembed",
    "llama-index-core",
    "llama-index-embeddings-huggingface",
    "llama-index-llms-ollama",
    "llama-index-llms-openai",
    "llama-index-llms-openai-like",
    "onnxruntime",
    "pypdfium2",
    "rapidocr",
    "torch",
}
FORBIDDEN_DIRECT_DEPENDENCIES = {
    "fastembed-gpu",
    "kuzu",
    "llama-index",
    "llama-index-embeddings-clip",
    "llama-index-embeddings-fastembed",
    "llama-index-embeddings-openai",
    "llama-index-graph-stores-kuzu",
    "onnxruntime-gpu",
    "openai",
    "openai-whisper",
    "tenacity",
}

pytestmark = pytest.mark.unit


def _project_dependencies() -> set[str]:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    return {
        canonicalize_name(Requirement(value).name) for value in project["dependencies"]
    }


def _source_imports(path: Path) -> tuple[set[str], set[str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    direct: set[str] = set()
    from_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            direct.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            from_modules.add(node.module)
    return direct, from_modules


def test_project_dependency_contract() -> None:
    """Required integrations have one owner and removed paths stay removed."""
    dependencies = _project_dependencies()

    assert dependencies >= REQUIRED_PROJECT_DEPENDENCIES
    assert FORBIDDEN_DIRECT_DEPENDENCIES.isdisjoint(dependencies)

    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    assert "llama" not in project["optional-dependencies"]


def test_llama_index_imports_are_modular() -> None:
    """Production code imports concrete LlamaIndex modules, not the meta-package."""
    violations: list[Path] = []
    modular_imports: set[str] = set()

    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        direct, from_modules = _source_imports(path)
        if "llama_index" in direct:
            violations.append(path.relative_to(PROJECT_ROOT))
        modular_imports.update(
            module for module in from_modules if module.startswith("llama_index.")
        )

    assert not violations
    assert modular_imports


@pytest.mark.parametrize(
    "module_name",
    ["cryptography", "llama_index.core", "loguru", "pydantic", "streamlit"],
)
def test_required_runtime_imports_resolve(module_name: str) -> None:
    """The locked base environment contains every required runtime package."""
    assert importlib.import_module(module_name)
