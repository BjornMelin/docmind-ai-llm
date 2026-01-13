"""Shared helpers for E2E test mocking and assertions."""

from __future__ import annotations

import sys
import types as types_
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


class _StubCoordinator:
    """Lightweight stub coordinator used for isolation in tests."""

    def __init__(self, *_: object, **__: object) -> None:
        return

    def process_query(self, *_a: object, **_kw: object) -> SimpleNamespace:
        return SimpleNamespace(content="stub")


class _StubToolFactory:
    """Stubbed ToolFactory with no-op tool creation."""

    @staticmethod
    def create_basic_tools(_settings: object) -> list[object]:
        return []


def install_mock_torch(
    monkeypatch: Any,
    *,
    include_cuda_props: bool = True,
    include_device: bool = False,
    include_tensor: bool = False,
    include_nn: bool = False,
) -> MagicMock:
    """Install a mocked torch module with optional CUDA/device helpers."""
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.7.1+cu126"
    mock_torch.__spec__ = MagicMock()
    mock_torch.__spec__.name = "torch"
    mock_torch.cuda.is_available.return_value = True
    if include_cuda_props:
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            name="RTX 4090",
            total_memory=17179869184,
        )
    if include_device:
        mock_torch.device = MagicMock()
    if include_tensor:
        mock_torch.tensor = MagicMock()
    if include_nn:
        mock_torch.nn = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    return mock_torch


def install_heavy_dependencies(monkeypatch: Any, dependencies: Iterable[str]) -> None:
    """Mock heavy optional dependencies if they are not already loaded."""
    for module in dependencies:
        if module not in sys.modules:
            monkeypatch.setitem(sys.modules, module, MagicMock())


def install_mock_ollama(monkeypatch: Any) -> MagicMock:
    """Mock ollama client boundary calls."""
    mock_ollama = MagicMock()
    mock_ollama.list.return_value = {
        "models": [{"name": "qwen3-4b-instruct-2507:latest"}]
    }
    mock_ollama.pull.return_value = {"status": "success"}
    mock_ollama.chat.return_value = {"message": {"content": "Test response"}}
    monkeypatch.setitem(sys.modules, "ollama", mock_ollama)
    return mock_ollama


def install_dependency_injector(monkeypatch: Any) -> None:
    """Mock dependency_injector for import resolution."""
    mock_dependency_injector = MagicMock()
    mock_dependency_injector.wiring = MagicMock()
    mock_dependency_injector.wiring.Provide = MagicMock()
    mock_dependency_injector.wiring.inject = MagicMock()
    mock_dependency_injector.containers = MagicMock()
    mock_dependency_injector.providers = MagicMock()
    monkeypatch.setitem(sys.modules, "dependency_injector", mock_dependency_injector)
    monkeypatch.setitem(
        sys.modules,
        "dependency_injector.containers",
        mock_dependency_injector.containers,
    )
    monkeypatch.setitem(
        sys.modules, "dependency_injector.providers", mock_dependency_injector.providers
    )
    monkeypatch.setitem(
        sys.modules, "dependency_injector.wiring", mock_dependency_injector.wiring
    )


def install_llama_index_core(monkeypatch: Any) -> None:
    """Mock minimal LlamaIndex core objects used by E2E tests."""
    li_pkg = ModuleType("llama_index")
    li_pkg.__path__ = []  # mark as package

    li_core = ModuleType("llama_index.core")
    li_core.__path__ = []  # mark as package

    class _DummySettings:
        llm = None
        embed_model = None
        context_window = 4096
        num_output = 512

    li_core.Settings = _DummySettings

    class _DummyDocument:
        def __init__(
            self, text: str = "", metadata: dict[str, Any] | None = None, **_: Any
        ):
            self.text = text
            self.metadata = metadata or {}

    li_core.Document = _DummyDocument

    class _DummyStorageContext:
        """Minimal storage context shim used in tests."""

        def __init__(
            self, *, vector_store: Any = None, image_store: Any = None
        ) -> None:
            self.vector_store = vector_store
            self.image_store = image_store

        @classmethod
        def from_defaults(
            cls, *, vector_store: Any = None, image_store: Any = None
        ) -> _DummyStorageContext:
            return cls(vector_store=vector_store, image_store=image_store)

    li_core.StorageContext = _DummyStorageContext

    class _DummyPGI:
        """Placeholder for PropertyGraphIndex import compatibility in tests."""

        pass

    li_core.PropertyGraphIndex = _DummyPGI

    li_llms = ModuleType("llama_index.core.llms")

    class _ChatMessage:
        """Chat message shim exposing role and content fields."""

        def __init__(self, role: str, content: str):
            self.role = role
            self.content = content

    li_llms.ChatMessage = _ChatMessage

    li_indices = ModuleType("llama_index.core.indices")

    class _DummyMMIndex:
        """Minimal multi-modal index shim with factory constructor."""

        @classmethod
        def from_documents(cls, *_args: Any, **_kwargs: Any) -> _DummyMMIndex:
            return cls()

    li_indices.MultiModalVectorStoreIndex = _DummyMMIndex

    class _DummyVSI:
        pass

    li_core.VectorStoreIndex = _DummyVSI

    li_retrievers = ModuleType("llama_index.core.retrievers")

    class _DummyBaseRetriever:
        pass

    li_retrievers.BaseRetriever = _DummyBaseRetriever

    li_pkg.core = li_core

    monkeypatch.setitem(sys.modules, "llama_index", li_pkg)
    monkeypatch.setitem(sys.modules, "llama_index.core", li_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.llms", li_llms)
    monkeypatch.setitem(sys.modules, "llama_index.core.indices", li_indices)
    monkeypatch.setitem(sys.modules, "llama_index.core.retrievers", li_retrievers)
    monkeypatch.setitem(sys.modules, "llama_index.core.memory", MagicMock())
    monkeypatch.setitem(sys.modules, "llama_index.core.vector_stores", MagicMock())


def install_agent_stubs(monkeypatch: Any) -> None:
    """Provide lightweight stubs for src.agents imports."""
    agents_coord = ModuleType("src.agents.coordinator")
    agents_coord.MultiAgentCoordinator = _StubCoordinator

    agents_factory = ModuleType("src.agents.tool_factory")
    agents_factory.ToolFactory = _StubToolFactory

    monkeypatch.setitem(sys.modules, "src.agents.coordinator", agents_coord)
    monkeypatch.setitem(sys.modules, "src.agents.tool_factory", agents_factory)


def build_isolated_modules() -> dict[str, types_.ModuleType]:
    """Build isolated module stubs for workflow testing."""
    src_pkg = types_.ModuleType("src")
    agents_pkg = types_.ModuleType("src.agents")
    utils_pkg = types_.ModuleType("src.utils")

    coord_mod = types_.ModuleType("src.agents.coordinator")
    coord_mod.MultiAgentCoordinator = _StubCoordinator

    tf_mod = types_.ModuleType("src.agents.tool_factory")
    tf_mod.ToolFactory = _StubToolFactory

    src_pkg.agents = agents_pkg
    src_pkg.utils = utils_pkg

    return {
        "src": src_pkg,
        "src.agents": agents_pkg,
        "src.utils": utils_pkg,
        "src.utils.core": types_.ModuleType("src.utils.core"),
        "src.utils.document": types_.ModuleType("src.utils.document"),
        "src.agents.coordinator": coord_mod,
        "src.agents.tool_factory": tf_mod,
    }


def assert_documents_have_required_metadata(
    documents: Iterable[object],
    *,
    expected_source: str | None = None,
) -> None:
    """Assert that documents include text and required metadata."""
    for doc in documents:
        text = getattr(doc, "text", None)
        assert text is not None
        assert len(text) > 0
        metadata = getattr(doc, "metadata", None)
        assert metadata is not None
        assert "source" in metadata
        assert "page" in metadata
        assert "chunk_id" in metadata
        if expected_source is not None:
            assert metadata["source"] == expected_source


def configure_hardware_mocks(mock_detect: MagicMock, mock_validate: MagicMock) -> None:
    """Configure standard hardware detection mocks."""
    mock_detect.return_value = {
        "gpu_name": "RTX 4090",
        "vram_total_gb": 24,
        "cuda_available": True,
    }
    mock_validate.return_value = True


@contextmanager
def patch_async_workflow_dependencies() -> Generator[tuple[AsyncMock, Any], None, None]:
    """Patch async workflow dependencies used across E2E tests."""
    with (
        patch(
            "src.utils.document.load_documents_unstructured",
            new_callable=AsyncMock,
            create=True,
        ) as mock_load,
        patch(
            "src.agents.coordinator.MultiAgentCoordinator",
            create=True,
        ) as mock_coordinator_class,
    ):
        yield mock_load, mock_coordinator_class


@contextmanager
def patch_document_loader() -> Generator[AsyncMock, None, None]:
    """Patch the async document loader for E2E tests."""
    with patch(
        "src.utils.document.load_documents_unstructured",
        new_callable=AsyncMock,
        create=True,
    ) as mock_load_docs:
        yield mock_load_docs


@contextmanager
def patch_hardware_validation() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Patch hardware detection and startup validation."""
    with (
        patch("src.utils.core.detect_hardware", create=True) as mock_detect,
        patch(
            "src.utils.core.validate_startup_configuration", create=True
        ) as mock_validate,
    ):
        yield mock_detect, mock_validate
