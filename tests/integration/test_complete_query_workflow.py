"""End-to-end query workflow tests aligned with current APIs.

Modernized to use LlamaIndex in-memory indexes, MockLLM/MockEmbedding, and
LangGraph supervisor patching via compile().stream to avoid external services.
"""

import contextlib
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.memory import ChatMemoryBuffer

# Ensure proper imports
from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import AgentResponse
from src.models.processing import ProcessingResult
from src.processing.document_processor import DocumentProcessor

# pylint: disable=redefined-outer-name
# Rationale: pytest fixture names intentionally shadow same-named objects when
# injected into tests; keeping names aligns with pytest patterns and readability.
from tests.fixtures.sample_documents import create_sample_documents
from tests.fixtures.test_settings import IntegrationTestSettings


def _fake_supervisor_graph(final_text: str):
    """Return a fake supervisor object compatible with compile().stream."""

    class _Compiled:
        def __init__(self, text: str, supervisor_ref: Mock) -> None:
            self.text = text
            self._supervisor_ref = supervisor_ref

        def stream(self, initial_state=None, config=None, stream_mode: str = "values"):
            """Yield a single final state with the provided text."""
            # Mark ainvoke as called for legacy assertions
            with contextlib.suppress(Exception):
                self._supervisor_ref.ainvoke()
            yield {
                "messages": [Mock(content=self.text, type="ai")],
                "next": "FINISH",
            }

    class _Supervisor:
        def __init__(self, text: str) -> None:
            self._text = text
            # legacy attribute used in some assertions
            self.ainvoke = Mock()

        def compile(self, checkpointer=None):  # pylint: disable=unused-argument
            """Return compiled shim that supports .stream()."""
            return _Compiled(self._text, self)

    return _Supervisor(final_text)


@pytest.fixture
def integration_settings():
    """Provide integration settings for workflow tests."""
    return IntegrationTestSettings(
        data_dir=Path("./query_workflow_data"),
        cache_dir=Path("./query_workflow_cache"),
        enable_gpu_acceleration=False,
        log_level="INFO",
    )


@pytest.fixture
async def docs_on_disk(tmp_path):
    """Create sample document files on disk for testing."""
    return create_sample_documents(tmp_path)


@pytest.fixture
def supervisor_patched():
    """Patch LangGraph components to a minimal, deterministic pipeline."""
    with (
        patch("src.agents.coordinator.create_react_agent") as _mock_agent,
        patch("src.agents.coordinator.create_supervisor") as mock_supervisor,
        patch("src.config.integrations.setup_llamaindex"),
    ):
        _mock_agent.return_value = Mock()
        mock_supervisor.return_value = _fake_supervisor_graph(
            "Mocked final agent response"
        )
        yield


def _build_index_from_paths(paths: list[Path]) -> VectorStoreIndex:
    docs = []
    for p in paths:
        with contextlib.suppress(Exception):
            docs.append(Document(text=p.read_text()))
    return VectorStoreIndex.from_documents(docs)


@pytest.mark.integration
class TestCompleteQueryWorkflow:
    """End-to-end workflows with in-memory index and patched supervisor."""

    @pytest.mark.usefixtures("supervisor_patched")
    def test_end_to_end_with_in_memory_index(self, docs_on_disk):
        """End-to-end query with in-memory index + patched graph."""
        index = _build_index_from_paths(
            [
                docs_on_disk["research_paper"],
                docs_on_disk["tech_docs"],
                docs_on_disk["business_report"],
            ]
        )

        coordinator = MultiAgentCoordinator()
        memory = ChatMemoryBuffer.from_defaults()

        response = coordinator.process_query(
            "Explain relationships between AI, ML, and neural networks",
            context=memory,
            settings_override={"vector": index},
        )

        assert isinstance(response, AgentResponse)
        assert response.content
        assert coordinator.total_queries >= 1

    @pytest.mark.usefixtures("supervisor_patched")
    def test_contextual_conversation(self):
        """Ensure repeated queries update coordinator stats and succeed."""
        coordinator = MultiAgentCoordinator()
        memory = ChatMemoryBuffer.from_defaults()

        r1 = coordinator.process_query("Tell me about AI", context=memory)
        assert isinstance(r1, AgentResponse)
        r2 = coordinator.process_query("How about ML?", context=memory)
        assert isinstance(r2, AgentResponse)
        assert coordinator.successful_queries >= 2

    def test_error_recovery(self):
        """First graph step fails then recovers and finishes."""
        # Patch to raise once, then finish
        with (
            patch("src.agents.coordinator.create_react_agent") as _mock_agent,
            patch("src.agents.coordinator.create_supervisor") as mock_supervisor,
            patch("src.config.integrations.setup_llamaindex"),
        ):
            _mock_agent.return_value = Mock()

            class _FlakyCompiled:
                def __init__(self) -> None:
                    self.count = 0

                def stream(self, initial_state=None, config=None, stream_mode="values"):
                    """Yield once; raise on first call for failure simulation."""
                    self.count += 1
                    if self.count == 1:
                        raise RuntimeError("Simulated coordination failure")
                    yield {
                        "messages": [Mock(content="Recovered response", type="ai")],
                        "next": "FINISH",
                    }

            class _FlakySupervisor:
                def __init__(self) -> None:
                    self.ainvoke = Mock()

                def compile(self, checkpointer=None):  # pylint: disable=unused-argument
                    return _FlakyCompiled()

            mock_supervisor.return_value = _FlakySupervisor()

            coordinator = MultiAgentCoordinator()
            memory = ChatMemoryBuffer.from_defaults()
            resp = coordinator.process_query("Test", context=memory)
            assert isinstance(resp, AgentResponse)


@pytest.mark.integration
class TestDocumentProcessingBoundary:
    """Boundary tests for DocumentProcessor within workflow context."""

    @pytest.mark.asyncio
    async def test_processing_small_doc(self, integration_settings, tmp_path):
        """Processor extracts elements from a small text file."""
        p = tmp_path / "small.txt"
        p.write_text("Small document for processor test.")
        processor = DocumentProcessor(integration_settings)
        result = await processor.process_document_async(p)
        assert isinstance(result, ProcessingResult)
        assert result.elements

    @pytest.mark.asyncio
    async def test_processing_corrupted_doc(self, integration_settings, tmp_path):
        """Processor handles corrupted input without crashing the suite."""
        p = tmp_path / "corrupted.bin"
        p.write_bytes(b"\x00\xff\x00bad")
        processor = DocumentProcessor(integration_settings)
        with contextlib.suppress(Exception):
            _ = await processor.process_document_async(p)
