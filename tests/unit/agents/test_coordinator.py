"""Unit tests for MultiAgentCoordinator and ContextManager.

Covers configuration, state management, workflow setup, and error handling.
"""
# pylint: disable=too-many-public-methods

# NOTE: We intentionally exercise a few hook helpers via private methods
# (e.g., _create_post_model_hook) because no equivalent public seam exists.
# Keep private access minimal and well-documented.
# pylint: disable=protected-access,redefined-outer-name,unused-argument

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.agents.coordinator import (
    COORDINATION_OVERHEAD_THRESHOLD,
    ContextManager,
    MultiAgentCoordinator,
    create_multi_agent_coordinator,
)
from src.agents.models import AgentResponse
from tests.fixtures.test_settings import MockDocMindSettings


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
def mock_dspy_retriever():
    """Mock DSPy retriever."""
    return Mock()


@pytest.mark.unit
class TestContextManager:
    """Comprehensive tests for ContextManager class."""

    def test_context_manager_initialization_default(self):
        """Test ContextManager initialization with default parameters."""
        context_manager = ContextManager()

        assert context_manager.max_context_tokens == 131072  # 128K default
        assert context_manager.trim_threshold == int(131072 * 0.9)  # 90% threshold
        assert context_manager.kv_cache_memory_per_token == 1024

    def test_context_manager_initialization_custom(self):
        """Test ContextManager initialization with custom parameters."""
        custom_tokens = 50000
        context_manager = ContextManager(max_context_tokens=custom_tokens)

        assert context_manager.max_context_tokens == custom_tokens
        assert context_manager.trim_threshold == int(custom_tokens * 0.9)
        assert context_manager.kv_cache_memory_per_token == 1024

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

    def test_calculate_kv_cache_usage(self, sample_messages):
        """Test KV cache usage calculation."""
        context_manager = ContextManager()

        state = {"messages": sample_messages}
        usage_gb = context_manager.calculate_kv_cache_usage(state)

        assert isinstance(usage_gb, float)
        assert usage_gb >= 0

    def test_calculate_kv_cache_usage_empty_state(self):
        """Test KV cache usage with empty state."""
        context_manager = ContextManager()

        usage_gb = context_manager.calculate_kv_cache_usage({})
        assert usage_gb == 0.0

    def test_structure_response(self):
        """Test response structuring."""
        context_manager = ContextManager()

        response = "Test response content"
        structured = context_manager.structure_response(response)

        assert isinstance(structured, dict)
        assert structured["content"] == response
        assert structured["structured"] is True
        assert "generated_at" in structured
        assert structured["context_optimized"] is True


@pytest.mark.unit
class TestMultiAgentCoordinator:
    """Comprehensive tests for MultiAgentCoordinator class."""

    @pytest.mark.usefixtures("test_settings")
    def test_coordinator_initialization_defaults(self, mock_llm):
        """Test coordinator initialization with default parameters."""
        with (
            patch("src.config.setup_llamaindex"),
            patch("llama_index.core.Settings") as mock_settings,
        ):
            mock_settings.llm = mock_llm

            coordinator = MultiAgentCoordinator()

            from src.config import settings as app_settings

            assert coordinator.model_path == "Qwen/Qwen3-4B-Instruct-2507-FP8"
            # Defaults derive from application settings, not test fixture
            assert coordinator.max_context_length == app_settings.vllm.context_window
            assert coordinator.backend == "vllm"
            assert coordinator.enable_fallback is True
            # Defaults derive from application settings
            assert coordinator.max_agent_timeout == app_settings.agents.decision_timeout
            assert coordinator._setup_complete is False

    @pytest.mark.usefixtures("test_settings")
    def test_coordinator_initialization_custom_params(self):
        """Test coordinator initialization with custom parameters."""
        custom_model = "custom/model"
        custom_context = 64000
        custom_timeout = 15.0

        coordinator = MultiAgentCoordinator(
            model_path=custom_model,
            max_context_length=custom_context,
            backend="custom",
            enable_fallback=False,
            max_agent_timeout=custom_timeout,
        )

        assert coordinator.model_path == custom_model
        assert coordinator.max_context_length == custom_context
        assert coordinator.backend == "custom"
        assert coordinator.enable_fallback is False
        assert coordinator.max_agent_timeout == custom_timeout

    def test_coordinator_vllm_config_generation(self, test_settings):
        """Test vLLM configuration generation.

        Patch the coordinator's imported settings object to avoid mutating the
        pydantic BaseSettings instance directly (which forbids setattr on
        non-fields). Provide a subclass overriding get_vllm_env_vars.
        """

        class _OverrideSettings(MockDocMindSettings):
            def get_vllm_env_vars(self):  # type: ignore[override]
                return {"test_var": "test_value"}

        with patch("src.agents.coordinator.settings", new=_OverrideSettings()):
            coordinator = MultiAgentCoordinator()

            assert "model" in coordinator.vllm_config
            assert "max_model_len" in coordinator.vllm_config
            assert "test_var" in coordinator.vllm_config
            assert coordinator.vllm_config["test_var"] == "test_value"

    def test_coordinator_performance_metrics_initialization(self):
        """Test performance metrics are initialized correctly."""
        coordinator = MultiAgentCoordinator()

        assert coordinator.total_queries == 0
        assert coordinator.successful_queries == 0
        assert coordinator.fallback_queries == 0
        assert coordinator.avg_processing_time == 0.0
        assert coordinator.avg_coordination_overhead == 0.0

    @patch("src.config.setup_llamaindex")
    @patch("llama_index.core.Settings")
    def test_ensure_setup_success(self, mock_settings, mock_setup):
        """Test successful setup of coordinator components."""
        # Mock LLM in Settings
        mock_llm = Mock()
        mock_llm.invoke = Mock(return_value="ok")
        mock_llm.ainvoke = AsyncMock(return_value="ok")
        mock_llm.predict = Mock(return_value="ok")
        mock_llm.apredict = AsyncMock(return_value="ok")
        mock_llm.stream = Mock(return_value=iter(()))
        mock_llm.astream = AsyncMock(return_value=iter(()))
        mock_settings.llm = mock_llm

        # Mock DSPy availability
        with (
            patch("src.agents.coordinator.is_dspy_available", return_value=True),
            patch("src.agents.coordinator.DSPyLlamaIndexRetriever") as mock_dspy,
            patch("src.agents.coordinator.create_react_agent") as mock_react,
            patch("src.agents.coordinator.create_supervisor") as mock_supervisor,
            patch(
                "src.agents.coordinator.create_forward_message_tool",
                return_value=Mock(),
            ),
        ):
            # Configure supervisor to return a mock graph with compile()
            mock_graph = Mock()
            mock_compiled = Mock()
            mock_graph.compile.return_value = mock_compiled
            mock_supervisor.return_value = mock_graph

            coordinator = MultiAgentCoordinator()
            result = coordinator._ensure_setup()

            assert result is True
            assert coordinator._setup_complete is True
            assert hasattr(coordinator.llm, "invoke")
            assert coordinator._shared_llm_wrapper is not None
            assert coordinator._shared_llm_wrapper.inner == mock_llm
            assert coordinator.compiled_graph == mock_compiled
            mock_setup.assert_called_once()
            mock_dspy.assert_called_once_with(llm=coordinator.llm)
            assert mock_react.call_count == 5

    @patch("src.config.setup_llamaindex")
    @patch("llama_index.core.Settings")
    def test_ensure_setup_no_llm_failure(self, mock_settings, mock_setup):
        """Test setup failure when LLM not configured."""
        mock_settings.llm = None

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

    @patch("src.agents.coordinator.create_react_agent")
    @patch("src.agents.coordinator.create_supervisor")
    @patch("src.agents.coordinator.create_forward_message_tool")
    def test_setup_agent_graph_success(
        self, mock_forward_tool, mock_supervisor, mock_react_agent, mock_llm
    ):
        """Test successful agent graph setup."""
        # Setup mocks
        mock_agent = Mock()
        mock_react_agent.return_value = mock_agent

        mock_graph = Mock()
        mock_supervisor.return_value = mock_graph
        mock_compiled_graph = Mock()
        mock_graph.compile.return_value = mock_compiled_graph

        mock_forward_tool.return_value = Mock()

        coordinator = MultiAgentCoordinator()
        coordinator.llm = mock_llm

        coordinator._setup_agent_graph()

        # Verify all agents were created
        assert mock_react_agent.call_count == 5  # 5 agents

        # Verify supervisor was created
        mock_supervisor.assert_called_once()

        # Verify graph was compiled
        mock_graph.compile.assert_called_once()
        assert coordinator.compiled_graph == mock_compiled_graph

        # Verify agents dictionary was populated
        assert len(coordinator.agents) == 5
        expected_agents = [
            "router_agent",
            "planner_agent",
            "retrieval_agent",
            "synthesis_agent",
            "validation_agent",
        ]
        for agent_name in expected_agents:
            assert agent_name in coordinator.agents

    @patch("src.agents.coordinator.create_supervisor")
    @patch("src.agents.coordinator.create_forward_message_tool")
    @patch("src.agents.coordinator.create_react_agent")
    def test_supervisor_parameters_adr_011(
        self, mock_react_agent, mock_forward_tool, mock_create_supervisor, mock_llm
    ):
        """Supervisor is created with modern ADR-011 parameters."""
        mock_graph = Mock()
        mock_graph.compile.return_value = Mock()
        mock_create_supervisor.return_value = mock_graph

        coordinator = MultiAgentCoordinator()
        coordinator.llm = mock_llm
        coordinator._setup_agent_graph()

        # Validate create_supervisor was called with ADR-011 flags
        _, kwargs = mock_create_supervisor.call_args
        assert kwargs.get("parallel_tool_calls") is True
        assert kwargs.get("output_mode") == "last_message"
        assert kwargs.get("add_handoff_messages") is True
        assert "pre_model_hook" in kwargs
        assert "post_model_hook" in kwargs

    @patch(
        "src.agents.coordinator.create_react_agent",
        side_effect=RuntimeError("Agent creation failed"),
    )
    def test_setup_agent_graph_failure(self, mock_react_agent, mock_llm):
        """Test agent graph setup failure handling."""
        coordinator = MultiAgentCoordinator()
        coordinator.llm = mock_llm

        with pytest.raises(RuntimeError, match="Agent graph initialization failed"):
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
        assert "router_agent" in prompt
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

    def test_create_pre_model_hook_error_handling(self):
        """Test pre-model hook error handling."""
        coordinator = MultiAgentCoordinator()
        hook = coordinator._create_pre_model_hook()

        # Test with invalid state
        invalid_state = None
        result = hook(invalid_state)

        assert result is None  # Should return original state even if None

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

    def test_create_post_model_hook_error_handling(self):
        """Test post-model hook error handling."""
        coordinator = MultiAgentCoordinator()
        hook = coordinator._create_post_model_hook()

        # Test with invalid state
        invalid_state = None
        result = hook(invalid_state)

        assert result is None

    def test_error_response_via_public_process_query(self):
        """Exercise error path via public API instead of private helper."""
        coordinator = MultiAgentCoordinator(enable_fallback=False)
        # Force setup to succeed then make workflow raise to hit error path
        with (
            patch.object(coordinator, "_ensure_setup", return_value=True),
            patch.object(
                coordinator, "_run_agent_workflow", side_effect=RuntimeError("boom")
            ),
        ):
            response = coordinator.process_query("Test error")

        assert isinstance(response, AgentResponse)
        assert "Error processing query" in response.content
        assert response.validation_score == 0.0
        assert response.processing_time >= 0
        assert response.metadata.get("error")

    def test_fallback_basic_rag_via_public_process_query(self):
        """Fallback behavior exercised via public API when workflow fails."""
        coordinator = MultiAgentCoordinator(enable_fallback=True)
        with (
            patch.object(coordinator, "_ensure_setup", return_value=True),
            patch.object(
                coordinator, "_run_agent_workflow", side_effect=RuntimeError("fail")
            ),
        ):
            response = coordinator.process_query("Test query")

        assert isinstance(response, AgentResponse)
        assert response.metadata.get("fallback_used") is True
        assert response.validation_score == 0.3
        assert coordinator.fallback_queries == 1

    def test_fallback_error_path_via_public_process_query(self):
        """If fallback path itself fails, public API still returns an error response."""
        coordinator = MultiAgentCoordinator(enable_fallback=True)
        with (
            patch.object(coordinator, "_ensure_setup", return_value=True),
            patch.object(
                coordinator, "_run_agent_workflow", side_effect=RuntimeError("wf")
            ),
            patch("src.agents.coordinator.logger.info", side_effect=RuntimeError("x")),
        ):
            response = coordinator.process_query("Test query")

        assert isinstance(response, AgentResponse)
        assert response.metadata.get("fallback_failed") is True

    def test_update_performance_metrics_first_query(self):
        """Test performance metrics update for first query."""
        coordinator = MultiAgentCoordinator()
        processing_time = 1.5
        coordination_time = 0.1

        coordinator.total_queries = 1
        coordinator._update_performance_metrics(processing_time, coordination_time)

        assert coordinator.avg_processing_time == processing_time
        assert coordinator.avg_coordination_overhead == coordination_time

    def test_update_performance_metrics_multiple_queries(self):
        """Test performance metrics update for multiple queries."""
        coordinator = MultiAgentCoordinator()

        # Simulate previous metrics
        coordinator.total_queries = 2
        coordinator.avg_processing_time = 1.0
        coordinator.avg_coordination_overhead = 0.05

        # Add new metrics
        processing_time = 2.0
        coordination_time = 0.1

        coordinator._update_performance_metrics(processing_time, coordination_time)

        # Check that averages are updated correctly
        expected_avg_processing = (1.0 * 1 + processing_time) / 2
        expected_avg_coordination = (0.05 * 1 + coordination_time) / 2

        assert coordinator.avg_processing_time == expected_avg_processing
        assert coordinator.avg_coordination_overhead == expected_avg_coordination

    def test_get_performance_stats(self):
        """Test performance statistics retrieval."""
        coordinator = MultiAgentCoordinator()
        coordinator.total_queries = 10
        coordinator.successful_queries = 8
        coordinator.fallback_queries = 2
        coordinator.avg_processing_time = 1.5
        coordinator.avg_coordination_overhead = 0.1

        stats = coordinator.get_performance_stats()

        assert isinstance(stats, dict)
        assert stats["total_queries"] == 10
        assert stats["successful_queries"] == 8
        assert stats["fallback_queries"] == 2
        assert stats["success_rate"] == 0.8
        assert stats["fallback_rate"] == 0.2
        assert stats["avg_processing_time"] == 1.5
        assert "adr_compliance" in stats
        assert "model_config" in stats

    def test_get_performance_stats_no_queries(self):
        """Test performance statistics with no queries."""
        coordinator = MultiAgentCoordinator()

        stats = coordinator.get_performance_stats()

        assert stats["success_rate"] == 0.0
        assert stats["fallback_rate"] == 0.0

    def test_validate_adr_compliance(self):
        """Test ADR compliance validation."""
        coordinator = MultiAgentCoordinator()
        coordinator._setup_complete = True
        coordinator.compiled_graph = Mock()
        coordinator.avg_coordination_overhead = 0.1  # Under 200ms

        compliance = coordinator.validate_adr_compliance()

        assert isinstance(compliance, dict)
        assert compliance["adr_001_supervisor_pattern"] is True
        assert compliance["adr_004_fp8_model"] is True  # Model path ends with FP8
        assert compliance["coordination_under_200ms"] is True
        assert compliance["context_128k_support"] is True

    def test_validate_adr_compliance_failures(self):
        """Test ADR compliance with some failures."""
        coordinator = MultiAgentCoordinator(
            model_path="non-fp8-model", max_context_length=50000
        )
        coordinator._setup_complete = False
        coordinator.avg_coordination_overhead = 0.3  # Over 200ms

        compliance = coordinator.validate_adr_compliance()

        assert compliance["adr_001_supervisor_pattern"] is False
        assert compliance["adr_004_fp8_model"] is False
        assert compliance["coordination_under_200ms"] is False
        assert compliance["context_128k_support"] is False

    def test_reset_performance_stats(self):
        """Test performance statistics reset."""
        coordinator = MultiAgentCoordinator()
        coordinator.total_queries = 10
        coordinator.successful_queries = 8
        coordinator.fallback_queries = 2
        coordinator.avg_processing_time = 1.5
        coordinator.avg_coordination_overhead = 0.1

        coordinator.reset_performance_stats()

        assert coordinator.total_queries == 0
        assert coordinator.successful_queries == 0
        assert coordinator.fallback_queries == 0
        assert coordinator.avg_processing_time == 0.0
        assert coordinator.avg_coordination_overhead == 0.0

    @patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=False)
    def test_process_query_setup_failure(self, mock_setup):
        """Test process_query with setup failure."""
        coordinator = MultiAgentCoordinator()

        response = coordinator.process_query("test query")

        assert isinstance(response, AgentResponse)
        assert "Failed to initialize coordinator" in response.content
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
        assert coordinator.total_queries == 1
        assert coordinator.successful_queries == 1
        mock_setup.assert_called_once()
        mock_workflow.assert_called_once()
        mock_extract.assert_called_once()

    @patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True)
    @patch.object(
        MultiAgentCoordinator,
        "_run_agent_workflow",
        side_effect=RuntimeError("Workflow error"),
    )
    @patch.object(MultiAgentCoordinator, "_fallback_basic_rag")
    def test_process_query_with_fallback(
        self, mock_fallback, mock_workflow, mock_setup
    ):
        """Test query processing with fallback on error."""
        coordinator = MultiAgentCoordinator(enable_fallback=True)

        # Mock fallback response
        mock_response = AgentResponse(
            content="Fallback response",
            sources=[],
            metadata={"fallback_used": True},
            validation_score=0.3,
            processing_time=0.5,
            optimization_metrics={},
        )
        mock_fallback.return_value = mock_response

        result = coordinator.process_query("test query")

        assert result == mock_response
        mock_fallback.assert_called_once()

    @patch.object(MultiAgentCoordinator, "_ensure_setup", return_value=True)
    @patch.object(
        MultiAgentCoordinator,
        "_run_agent_workflow",
        side_effect=RuntimeError("Workflow error"),
    )
    def test_process_query_no_fallback(self, mock_workflow, mock_setup):
        """Test query processing without fallback on error."""
        coordinator = MultiAgentCoordinator(enable_fallback=False)

        result = coordinator.process_query("test query")

        assert isinstance(result, AgentResponse)
        assert "Workflow error" in result.content
        assert result.validation_score == 0.0


@pytest.mark.unit
class TestFactoryFunction:
    """Test the factory function for creating coordinators."""

    def test_create_multi_agent_coordinator_defaults(self):
        """Test factory function with default parameters."""
        with patch("src.agents.coordinator.MultiAgentCoordinator") as mock_coordinator:
            mock_instance = Mock()
            mock_coordinator.return_value = mock_instance

            result = create_multi_agent_coordinator()

            mock_coordinator.assert_called_once_with(
                model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
                max_context_length=131072,  # From MockDocMindSettings
                enable_fallback=True,
                tool_registry=None,
                use_shared_llm_client=None,
            )
            assert result == mock_instance

    def test_create_multi_agent_coordinator_custom_params(self):
        """Test factory function with custom parameters."""
        with patch("src.agents.coordinator.MultiAgentCoordinator") as mock_coordinator:
            mock_instance = Mock()
            mock_coordinator.return_value = mock_instance

            result = create_multi_agent_coordinator(
                model_path="custom/model",
                max_context_length=64000,
                enable_fallback=False,
            )

            mock_coordinator.assert_called_once_with(
                model_path="custom/model",
                max_context_length=64000,
                enable_fallback=False,
                tool_registry=None,
                use_shared_llm_client=None,
            )
            assert result == mock_instance


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
        from src.agents.coordinator import (
            CONTEXT_TRIM_STRATEGY,
            PARALLEL_TOOL_CALLS_ENABLED,
        )

        assert isinstance(CONTEXT_TRIM_STRATEGY, str)
        assert isinstance(PARALLEL_TOOL_CALLS_ENABLED, bool)

        # Test specific value
        assert CONTEXT_TRIM_STRATEGY == "last"
