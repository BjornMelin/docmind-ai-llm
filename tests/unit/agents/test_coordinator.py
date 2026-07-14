"""Unit tests for MultiAgentCoordinator and ContextManager.

Covers configuration, state management, workflow setup, and error handling.
"""

# NOTE: We intentionally exercise a few hook helpers via private methods
# (e.g., _create_post_model_hook) because no equivalent public seam exists.
# Keep private access minimal and well-documented.

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agents.coordinator import (
    COORDINATION_OVERHEAD_THRESHOLD,
    ContextManager,
    MultiAgentCoordinator,
)
from src.agents.models import AgentResponse
from tests.fixtures.test_settings import MockDocMindSettings, MockLLMRequestConfig


def _make_llm_mock() -> Mock:
    """Create a fully mocked LLM with sync/async surfaces."""
    llm = Mock()
    llm.metadata = Mock()
    llm.metadata.model_name = "test-model"
    llm.invoke = Mock(return_value="ok")
    llm.ainvoke = AsyncMock(return_value="ok")
    llm.predict = Mock(return_value="ok")
    llm.apredict = AsyncMock(return_value="ok")
    llm.stream = Mock(return_value=iter(()))
    llm.astream = AsyncMock(return_value=iter(()))
    return llm


@pytest.fixture
def test_settings():
    """Create test settings for coordinator tests."""
    return MockDocMindSettings()


@pytest.fixture
def sample_messages():
    """Sample messages for testing."""
    return [
        {"content": "First test message with some content"},
        {"content": "Second longer message with more detailed content for testing"},
        {"content": "Third message"},
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM for testing."""
    return _make_llm_mock()


@pytest.mark.unit
class TestContextManager:
    """Comprehensive tests for ContextManager class."""

    def test_context_manager_initialization_default(self):
        """Test ContextManager initialization with default parameters."""
        context_manager = ContextManager()

        assert context_manager.max_context_tokens == 131072  # 128K default
        assert context_manager.trim_threshold == int(131072 * 0.9)  # 90% threshold

    def test_context_manager_initialization_custom(self):
        """Test ContextManager initialization with custom parameters."""
        custom_tokens = 50000
        context_manager = ContextManager(max_context_tokens=custom_tokens)

        assert context_manager.max_context_tokens == custom_tokens
        assert context_manager.trim_threshold == int(custom_tokens * 0.9)

    def test_estimate_tokens_empty_messages(self):
        """Test token estimation with empty message list."""
        context_manager = ContextManager()

        tokens = context_manager.estimate_tokens([])
        assert tokens == 0

    def test_estimate_tokens_dict_messages(self, sample_messages):
        """Test token estimation with dictionary messages."""
        context_manager = ContextManager()

        tokens = context_manager.estimate_tokens(sample_messages)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_object_messages(self):
        """Test token estimation with object messages."""
        context_manager = ContextManager()

        # Create mock messages with content attribute
        messages = [
            Mock(content="Test message one"),
            Mock(content="Test message two with more content"),
        ]

        tokens = context_manager.estimate_tokens(messages)
        assert tokens > 0
        assert isinstance(tokens, int)

    def test_estimate_tokens_mixed_messages(self):
        """Test token estimation with mixed message types."""
        context_manager = ContextManager()

        messages = [
            {"content": "Dict message"},
            Mock(content="Object message"),
            "String message",
        ]

        tokens = context_manager.estimate_tokens(messages)
        assert tokens > 0
        assert isinstance(tokens, int)


@pytest.mark.unit
class TestMultiAgentCoordinator:
    """Comprehensive tests for MultiAgentCoordinator class."""

    def test_coordinator_initialization_defaults(self):
        """Coordinator derives context and timeout defaults from settings."""
        coordinator = MultiAgentCoordinator()

        from src.config import settings as app_settings

        assert (
            coordinator.context_manager.max_context_tokens
            == app_settings.effective_context_window
        )
        assert coordinator.max_agent_timeout == app_settings.agents.decision_timeout
        assert coordinator._setup_complete is False

    def test_injected_persistence_is_preserved(self, tmp_path):
        """Injected test savers/stores remain valid and exclude path ownership."""
        checkpointer = object()
        store = object()

        coordinator = MultiAgentCoordinator(
            checkpointer=checkpointer,
            store=store,
        )

        assert coordinator.checkpointer is checkpointer
        assert coordinator.store is store
        with pytest.raises(ValueError, match="mutually exclusive"):
            MultiAgentCoordinator(
                checkpointer=checkpointer,
                checkpointer_path=tmp_path / "chat.db",
            )

    def test_coordinator_defaults_resolve_settings_at_call_time(self):
        """Read effective model and context after module import, not in defaults."""
        runtime_settings = MockDocMindSettings(
            llm_request=MockLLMRequestConfig(
                model="runtime/model",
                context_window=8192,
            ),
        )
        settings_globals = MultiAgentCoordinator.__init__.__globals__

        with patch.dict(
            settings_globals,
            {"settings": runtime_settings},
            clear=False,
        ):
            coordinator = MultiAgentCoordinator()
            assert coordinator.context_manager.max_context_tokens == 8192
            metrics = coordinator._build_base_optimization_metrics_from_state({})
            assert metrics["model_path"] == "runtime/model"

    @patch("src.config.setup_llamaindex")
    @patch("llama_index.core.Settings")
    def test_ensure_setup_success(self, mock_settings, mock_setup):
        """Test successful setup of coordinator components."""
        # Mock LLM in Settings
        mock_llm = _make_llm_mock()
        mock_settings._llm = mock_llm

        mock_agent_factory = Mock(return_value=Mock())
        mock_supervisor = Mock()

        # Configure supervisor to return a mock graph with compile()
        mock_graph = Mock()
        mock_compiled = Mock()
        mock_graph.compile.return_value = mock_compiled
        mock_supervisor.return_value = mock_graph

        graph_globals = MultiAgentCoordinator._setup_agent_graph.__globals__

        with patch.dict(
            graph_globals,
            {
                "create_agent": mock_agent_factory,
                "build_agent_tool_sets": lambda _settings: {
                    "planner_agent": [],
                    "retrieval_agent": [],
                    "synthesis_agent": [],
                    "validation_agent": [],
                },
                "build_multi_agent_supervisor_graph": mock_supervisor,
            },
            clear=False,
        ):
            coordinator = MultiAgentCoordinator()
            result = coordinator._ensure_setup()

            assert result is True
            assert coordinator._setup_complete is True
            assert hasattr(coordinator.llm, "invoke")
            assert coordinator.compiled_graph == mock_compiled
            mock_setup.assert_called_once()
            assert mock_agent_factory.call_count == 4

    @patch("src.config.setup_llamaindex")
    @patch("llama_index.core.Settings")
    def test_ensure_setup_no_llm_failure(self, mock_settings, mock_setup):
        """Test setup failure when LLM not configured."""
        mock_settings._llm = None

        coordinator = MultiAgentCoordinator()
        result = coordinator._ensure_setup()

        assert result is False
        assert coordinator._setup_complete is False

    @patch("src.config.setup_llamaindex", side_effect=RuntimeError("Setup failed"))
    def test_ensure_setup_runtime_error(self, mock_setup):
        """Test setup failure with runtime error."""
        coordinator = MultiAgentCoordinator()
        result = coordinator._ensure_setup()

        assert result is False
        assert coordinator._setup_complete is False

    def test_ensure_setup_already_complete(self):
        """Test that setup is skipped when already complete."""
        coordinator = MultiAgentCoordinator()
        coordinator._setup_complete = True

        result = coordinator._ensure_setup()

        assert result is True

    def test_setup_agent_graph_success(self, mock_llm):
        """Test successful agent graph setup."""
        # Setup mocks
        mock_agent_factory = Mock(return_value=Mock())
        mock_graph = Mock()
        mock_supervisor = Mock(return_value=mock_graph)
        mock_compiled_graph = Mock()
        mock_graph.compile.return_value = mock_compiled_graph

        graph_globals = MultiAgentCoordinator._setup_agent_graph.__globals__

        with patch.dict(
            graph_globals,
            {
                "create_agent": mock_agent_factory,
                "build_multi_agent_supervisor_graph": mock_supervisor,
            },
            clear=False,
        ):
            coordinator = MultiAgentCoordinator()
            coordinator.llm = mock_llm

            coordinator._setup_agent_graph()

            # Verify all agents were created
            assert mock_agent_factory.call_count == 4

            # Verify supervisor was created
            mock_supervisor.assert_called_once()

            # Verify graph was compiled
            mock_graph.compile.assert_called_once()
            assert coordinator.compiled_graph == mock_compiled_graph

            # Verify agents dictionary was populated
            assert len(coordinator.agents) == 4
            expected_agents = [
                "planner_agent",
                "retrieval_agent",
                "synthesis_agent",
                "validation_agent",
            ]
            for agent_name in expected_agents:
                assert agent_name in coordinator.agents

    def test_supervisor_parameters_adr_011(self, mock_llm):
        """Supervisor is created with modern ADR-011 parameters."""
        mock_graph = Mock()
        mock_graph.compile.return_value = Mock()
        mock_build_supervisor = Mock(return_value=mock_graph)
        mock_agent_factory = Mock(return_value=Mock())

        graph_globals = MultiAgentCoordinator._setup_agent_graph.__globals__

        with patch.dict(
            graph_globals,
            {
                "create_agent": mock_agent_factory,
                "build_agent_tool_sets": lambda _settings: {
                    "planner_agent": [],
                    "retrieval_agent": [],
                    "synthesis_agent": [],
                    "validation_agent": [],
                },
                "build_multi_agent_supervisor_graph": mock_build_supervisor,
            },
            clear=False,
        ):
            coordinator = MultiAgentCoordinator()
            coordinator.llm = mock_llm
            coordinator._setup_agent_graph()

            # Validate graph builder is wired with ADR-011 flags.
            _, kwargs = mock_build_supervisor.call_args
            params = kwargs.get("params")
            assert params is not None
            assert params.output_mode == "last_message"
            assert params.add_handoff_messages is True
            assert params.add_handoff_back_messages is True
            assert kwargs.get("state_schema") is not None
            assert kwargs.get("middleware")

    def test_setup_agent_graph_failure(self, mock_llm):
        """Test agent graph setup failure handling."""
        coordinator = MultiAgentCoordinator()
        coordinator.llm = mock_llm

        graph_globals = MultiAgentCoordinator._setup_agent_graph.__globals__
        failing_agent_factory = Mock(side_effect=RuntimeError("Agent creation failed"))

        with (
            patch.dict(
                graph_globals, {"create_agent": failing_agent_factory}, clear=False
            ),
            pytest.raises(RuntimeError, match="Agent graph initialization failed"),
        ):
            coordinator._setup_agent_graph()

    def test_create_supervisor_prompt(self):
        """Test supervisor prompt creation."""
        coordinator = MultiAgentCoordinator()
        prompt = coordinator._create_supervisor_prompt()

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "supervisor" in prompt.lower()
        assert "200ms" in prompt  # Performance target
        assert "parallel" in prompt.lower()
        assert "RouterQueryEngine" in prompt
        assert "validation_agent" in prompt

    def test_create_pre_model_hook(self):
        """Test pre-model hook creation and execution."""
        coordinator = MultiAgentCoordinator()
        hook = coordinator._create_pre_model_hook()

        assert callable(hook)

        # Test hook execution with normal state
        test_state = {"messages": [{"content": "Short message"}]}

        result = hook(test_state)
        assert isinstance(result, dict)
        assert "messages" in result

    def test_create_pre_model_hook_with_trimming(self):
        """Test pre-model hook with context trimming."""
        coordinator = MultiAgentCoordinator()
        coordinator.context_manager.trim_threshold = 1  # Force trimming

        hook = coordinator._create_pre_model_hook()

        # Create state that exceeds threshold
        test_state = {"messages": [{"content": "Very long message " * 100}]}

        with patch(
            "langchain_core.messages.utils.trim_messages",
            return_value=[{"content": "trimmed"}],
        ) as mock_trim:
            result = hook(test_state)

            mock_trim.assert_called_once()
            assert result.get("context_trimmed") is True
            assert "tokens_trimmed" in result

    def test_create_post_model_hook(self):
        """Test post-model hook creation and execution."""
        coordinator = MultiAgentCoordinator()
        hook = coordinator._create_post_model_hook()

        assert callable(hook)

        # Test hook execution
        test_state = {
            "output_mode": "structured",
            "messages": [{"content": "Test"}],
            "response": "Test response",
        }

        result = hook(test_state)
        assert isinstance(result, dict)
        assert "optimization_metrics" in result

    def test_error_response_via_public_process_query(self):
        """Exercise error path via public API instead of private helper."""
        coordinator = MultiAgentCoordinator()
        # Force setup to succeed then make workflow raise to hit error path
        with (
            patch.object(coordinator, "_ensure_setup", return_value=True),
            patch.object(
                coordinator,
                "_run_agent_workflow",
                side_effect=RuntimeError("secret=/tmp/private/token.txt"),
            ),
        ):
            response = coordinator.process_query("Test error")

        assert isinstance(response, AgentResponse)
        assert response.content == "Unable to process the query."
        assert response.validation_score == 0.0
        assert response.processing_time >= 0
        assert response.metadata == {"reason": "execution_failed"}
        assert "secret" not in str(response.model_dump())

    @patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=False)
    def test_process_query_setup_failure(self, mock_setup):
        """Test process_query with setup failure."""
        coordinator = MultiAgentCoordinator()

        response = coordinator.process_query("test query")

        assert isinstance(response, AgentResponse)
        assert response.content == "Unable to initialize the coordinator."
        assert response.metadata == {"reason": "initialization_failed"}
        assert response.validation_score == 0.0

    @patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True)
    @patch.object(MultiAgentCoordinator, "_run_agent_workflow")
    @patch.object(MultiAgentCoordinator, "_extract_response")
    def test_process_query_success(self, mock_extract, mock_workflow, mock_setup):
        """Test successful query processing."""
        coordinator = MultiAgentCoordinator()

        # Mock workflow result
        mock_workflow.return_value = {"messages": []}

        # Mock extracted response
        mock_response = AgentResponse(
            content="Test response",
            sources=[],
            metadata={},
            validation_score=0.9,
            processing_time=1.0,
            optimization_metrics={},
        )
        mock_extract.return_value = mock_response

        result = coordinator.process_query("test query")

        assert result == mock_response
        mock_setup.assert_called_once()
        mock_workflow.assert_called_once()
        mock_extract.assert_called_once()


@pytest.mark.unit
class TestConstants:
    """Test module-level constants and configuration."""

    def test_coordination_overhead_threshold(self):
        """Test coordination overhead threshold constant."""
        assert COORDINATION_OVERHEAD_THRESHOLD > 0
        assert (
            COORDINATION_OVERHEAD_THRESHOLD <= 1.0
        )  # Should be reasonable (≤ 1 second)

    def test_import_constants(self):
        """Test that all expected constants can be imported."""
        from src.agents.coordinator import CONTEXT_TRIM_STRATEGY

        assert isinstance(CONTEXT_TRIM_STRATEGY, str)

        # Test specific value
        assert CONTEXT_TRIM_STRATEGY == "last"
