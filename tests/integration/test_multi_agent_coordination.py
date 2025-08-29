"""Comprehensive integration tests for multi-agent coordination system.

Tests focus on agent coordination workflows, communication patterns,
and system integration scenarios. Targets 40% coverage for coordinator.py.

Key test areas:
- Agent system initialization and setup
- Query routing between agents
- Response generation and synthesis
- Fallback mechanisms and error handling
- Agent communication coordination
- Performance and timeout handling

Library-First Approach:
- LangGraph testing patterns
- pytest-asyncio for async coordination
- Mock agent calls but test coordination logic
- Real agent communication patterns
"""

import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import HumanMessage
from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer

# Ensure src is in Python path
PROJECT_ROOT = Path(__file__).parents[2]
src_path = str(PROJECT_ROOT / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import with graceful fallback
try:
    from src.agents.coordinator import MultiAgentCoordinator
    from src.agents.models import AgentResponse, MultiAgentState

    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    IMPORT_ERROR = str(e)


@pytest.fixture
def mock_settings():
    """Create mock settings that match the actual structure."""
    from src.config.settings import DocMindSettings

    # Create a copy of actual settings with overrides for testing
    settings = DocMindSettings()

    # Override vLLM configuration for testing
    settings.vllm.model = "test-model"
    settings.vllm.context_window = 131072
    settings.vllm.kv_cache_dtype = "fp8_e5m2"
    settings.vllm.attention_backend = "FLASHINFER"
    settings.vllm.backend = "ollama"

    # Override agent configuration
    settings.agents.enable_multi_agent = True
    settings.agents.decision_timeout = 200
    settings.agents.max_retries = 2

    return settings


@pytest.fixture
def mock_llamaindex_llm():
    """Create mock LlamaIndex LLM that works with Settings."""
    llm = Mock()
    llm.complete = Mock(return_value=Mock(text="Mock LLM response"))
    llm.acomplete = AsyncMock(return_value=Mock(text="Mock LLM response"))
    return llm


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorInitialization:
    """Test multi-agent coordinator initialization workflows."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_coordinator_initialization_basic(self, mock_setup, mock_settings):
        """Test basic coordinator initialization."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()

            # Test initialization attributes
            assert coordinator.model_path == "Qwen/Qwen3-4B-Instruct-2507-FP8"
            assert coordinator.max_context_length == mock_settings.vllm.context_window
            assert coordinator.enable_fallback is True
            assert (
                coordinator.max_agent_timeout == mock_settings.agents.decision_timeout
            )

            # Test performance tracking initialization
            assert coordinator.total_queries == 0
            assert coordinator.successful_queries == 0
            assert coordinator.fallback_queries == 0
            assert coordinator.avg_processing_time == 0.0
            assert coordinator.avg_coordination_overhead == 0.0

    @patch("src.config.integrations.setup_llamaindex")
    def test_coordinator_initialization_with_custom_params(self, mock_setup):
        """Test coordinator initialization with custom parameters."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator(
                model_path="custom/model",
                max_context_length=65536,
                enable_fallback=False,
                max_agent_timeout=100,
            )

            assert coordinator.model_path == "custom/model"
            assert coordinator.max_context_length == 65536
            assert coordinator.enable_fallback is False
            assert coordinator.max_agent_timeout == 100

    @patch("src.config.integrations.setup_llamaindex")
    def test_coordinator_vllm_configuration(self, mock_setup, mock_settings):
        """Test vLLM configuration setup."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()

            # Test vLLM config structure
            assert coordinator.vllm_config["model"] == coordinator.model_path
            assert (
                coordinator.vllm_config["max_model_len"]
                == coordinator.max_context_length
            )
            assert "VLLM_ATTENTION_BACKEND" in coordinator.vllm_config
            assert "VLLM_KV_CACHE_DTYPE" in coordinator.vllm_config


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorSetup:
    """Test agent system setup and configuration."""

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.agents.coordinator.create_react_agent")
    @patch("src.agents.coordinator.create_supervisor")
    def test_setup_agent_graph_success(
        self, mock_create_supervisor, mock_create_react_agent, mock_setup
    ):
        """Test successful agent graph setup."""
        # Setup mocks
        mock_agent = Mock()
        mock_create_react_agent.return_value = mock_agent
        mock_graph = Mock()
        mock_create_supervisor.return_value = mock_graph

        with patch.object(Settings, "llm", mock_llamaindex_llm()) as mock_llm:
            coordinator = MultiAgentCoordinator()
            result = coordinator._ensure_setup()

            assert result is True
            assert coordinator._setup_complete is True
            assert coordinator.llm == mock_llm

            # Verify agents were created
            mock_create_react_agent.assert_called()
            assert mock_create_react_agent.call_count == 5  # 5 agents

            # Verify supervisor was created
            mock_create_supervisor.assert_called_once()

    @patch("src.config.integrations.setup_llamaindex")
    def test_setup_failure_handling(self, mock_setup):
        """Test setup failure handling."""
        # Mock Settings.llm to be None to trigger failure
        with patch.object(Settings, "llm", None):
            coordinator = MultiAgentCoordinator()
            result = coordinator._ensure_setup()

            assert result is False
            assert coordinator._setup_complete is False

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.agents.coordinator.create_supervisor")
    def test_supervisor_creation_with_modern_parameters(
        self, mock_create_supervisor, mock_setup
    ):
        """Test supervisor creation with ADR-011 modern parameters."""
        mock_graph = Mock()
        mock_create_supervisor.return_value = mock_graph

        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_agent_graph()

            # Verify supervisor was called with modern parameters
            call_kwargs = mock_create_supervisor.call_args[1]
            assert call_kwargs["parallel_tool_calls"] is True
            assert call_kwargs["output_mode"] == "structured"
            assert call_kwargs["create_forward_message_tool"] is True
            assert call_kwargs["add_handoff_back_messages"] is True
            assert "pre_model_hook" in call_kwargs
            assert "post_model_hook" in call_kwargs


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
@pytest.mark.asyncio
class TestMultiAgentCoordinatorQueryProcessing:
    """Test query processing workflows and agent coordination."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_process_query_simple_workflow(self, mock_setup):
        """Test simple query processing workflow."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock compiled graph with simple workflow
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                return_value=[
                    {
                        "messages": [
                            HumanMessage(content="Paris is the capital of France")
                        ],
                        "routing_decision": {
                            "strategy": "vector",
                            "complexity": "simple",
                        },
                        "validation_result": {"confidence": 0.95},
                    }
                ]
            )

            response = coordinator.process_query("What is the capital of France?")

            assert isinstance(response, AgentResponse)
            assert "Paris" in response.content
            assert response.validation_score == 0.95
            assert response.processing_time > 0
            assert "coordination_overhead_ms" in response.optimization_metrics

    @patch("src.config.integrations.setup_llamaindex")
    def test_process_query_complex_workflow(self, mock_setup):
        """Test complex query processing with planning."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock compiled graph with complex workflow
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                return_value=[
                    {
                        "messages": [HumanMessage(content="Complex analysis complete")],
                        "routing_decision": {
                            "strategy": "hybrid",
                            "complexity": "complex",
                            "needs_planning": True,
                        },
                        "planning_output": {
                            "sub_tasks": ["Analyze A", "Analyze B", "Compare results"]
                        },
                        "synthesis_result": {
                            "documents": [
                                {"content": "Synthesized analysis", "score": 0.9}
                            ]
                        },
                        "validation_result": {"confidence": 0.85},
                        "agent_timings": {
                            "router_agent": 0.05,
                            "planner_agent": 0.08,
                            "retrieval_agent": 0.15,
                            "synthesis_agent": 0.12,
                            "validation_agent": 0.06,
                        },
                    }
                ]
            )

            query = "Compare AI vs ML techniques in depth"
            response = coordinator.process_query(query)

            assert isinstance(response, AgentResponse)
            assert response.metadata["routing_decision"]["complexity"] == "complex"
            assert response.metadata["routing_decision"]["needs_planning"] is True
            assert len(response.metadata["planning_output"]["sub_tasks"]) == 3
            assert len(response.sources) > 0
            assert response.validation_score == 0.85

    @patch("src.config.integrations.setup_llamaindex")
    def test_process_query_with_context(self, mock_setup):
        """Test query processing with conversation context."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock compiled graph
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                return_value=[
                    {
                        "messages": [HumanMessage(content="Context-aware response")],
                        "validation_result": {"confidence": 0.9},
                    }
                ]
            )

            # Create conversation context
            context = ChatMemoryBuffer.from_defaults()
            context.put(HumanMessage(content="What is machine learning?"))

            response = coordinator.process_query(
                "Can you explain more about that?", context=context
            )

            assert isinstance(response, AgentResponse)
            assert response.optimization_metrics["context_window_used"] == 131072


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorFallback:
    """Test fallback mechanisms and error handling."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_fallback_on_setup_failure(self, mock_setup):
        """Test fallback when coordinator setup fails."""
        coordinator = MultiAgentCoordinator(enable_fallback=True)
        coordinator._setup_complete = False  # Force setup failure

        response = coordinator.process_query("Test query")

        assert isinstance(response, AgentResponse)
        assert "error" in response.content.lower()
        assert response.metadata["fallback_available"] is True
        assert "initialization_failed" in response.optimization_metrics

    @patch("src.config.integrations.setup_llamaindex")
    def test_fallback_on_workflow_failure(self, mock_setup):
        """Test fallback when agent workflow fails."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator(enable_fallback=True)
            coordinator._setup_complete = True

            # Mock workflow failure
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                side_effect=Exception("Workflow failed")
            )

            response = coordinator.process_query("Test query")

            assert isinstance(response, AgentResponse)
            assert response.metadata["fallback_used"] is True
            assert "fallback_mode" in response.optimization_metrics
            assert coordinator.fallback_queries == 1

    @patch("src.config.integrations.setup_llamaindex")
    def test_no_fallback_error_handling(self, mock_setup):
        """Test error handling when fallback is disabled."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator(enable_fallback=False)
            coordinator._setup_complete = True

            # Mock workflow failure
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                side_effect=Exception("Workflow failed")
            )

            response = coordinator.process_query("Test query")

            assert isinstance(response, AgentResponse)
            assert "Error processing query" in response.content
            assert response.validation_score == 0.0


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorPerformance:
    """Test performance tracking and coordination overhead."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_coordination_overhead_tracking(self, mock_setup):
        """Test coordination overhead measurement and 200ms target."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock fast workflow
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                return_value=[
                    {
                        "messages": [HumanMessage(content="Fast response")],
                        "validation_result": {"confidence": 0.9},
                    }
                ]
            )

            start_time = time.perf_counter()
            response = coordinator.process_query("Test query")
            elapsed = time.perf_counter() - start_time

            # Verify coordination overhead is tracked
            assert "coordination_overhead_ms" in response.optimization_metrics
            coordination_ms = response.optimization_metrics["coordination_overhead_ms"]
            assert coordination_ms >= 0

            # Verify meets 200ms target tracking
            assert "meets_200ms_target" in response.optimization_metrics

    @patch("src.config.integrations.setup_llamaindex")
    def test_performance_metrics_accumulation(self, mock_setup):
        """Test performance metrics accumulate over multiple queries."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock workflow
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                return_value=[
                    {
                        "messages": [HumanMessage(content="Response")],
                        "validation_result": {"confidence": 0.9},
                    }
                ]
            )

            # Process multiple queries
            queries = ["Query 1", "Query 2", "Query 3"]
            for query in queries:
                coordinator.process_query(query)

            # Verify metrics accumulation
            stats = coordinator.get_performance_stats()
            assert stats["total_queries"] == 3
            assert stats["successful_queries"] == 3
            assert stats["success_rate"] == 1.0
            assert stats["avg_processing_time"] > 0
            assert stats["avg_coordination_overhead_ms"] >= 0

    @patch("src.config.integrations.setup_llamaindex")
    def test_timeout_handling(self, mock_setup):
        """Test agent timeout handling."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator(max_agent_timeout=0.1)
            coordinator._setup_complete = True

            # Mock slow workflow
            def slow_stream(*args, **kwargs):
                time.sleep(0.2)  # Exceed timeout
                return [
                    {
                        "messages": [HumanMessage(content="Slow response")],
                        "validation_result": {"confidence": 0.9},
                    }
                ]

            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = slow_stream

            start_time = time.perf_counter()
            response = coordinator.process_query("Test query")
            elapsed = time.perf_counter() - start_time

            # Should not wait indefinitely
            assert elapsed < 1.0
            assert isinstance(response, AgentResponse)


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorContextManagement:
    """Test context management and token handling."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_context_estimation(self, mock_setup):
        """Test token estimation for context management."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()

            # Test token estimation
            messages = [
                {"content": "A" * 400},  # ~100 tokens
                {"content": "B" * 800},  # ~200 tokens
            ]

            tokens = coordinator.context_manager.estimate_tokens(messages)
            assert tokens == 300  # 100 + 200 tokens

    @patch("src.config.integrations.setup_llamaindex")
    def test_context_hooks_creation(self, mock_setup):
        """Test pre/post model hook creation."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()

            # Test pre-model hook
            pre_hook = coordinator._create_pre_model_hook()
            assert callable(pre_hook)

            # Test post-model hook
            post_hook = coordinator._create_post_model_hook()
            assert callable(post_hook)

            # Test hook execution
            test_state = {
                "messages": [{"content": "Test message"}],
                "output_mode": "structured",
            }

            result = pre_hook(test_state)
            assert isinstance(result, dict)

            result = post_hook(result)
            assert isinstance(result, dict)


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorCompliance:
    """Test ADR compliance validation."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_adr_compliance_validation(self, mock_setup):
        """Test comprehensive ADR compliance validation."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            compliance = coordinator.validate_adr_compliance()

            # Test ADR-001 (supervisor pattern)
            assert "adr_001_supervisor_pattern" in compliance

            # Test ADR-004 (FP8 model)
            assert "adr_004_fp8_model" in compliance
            assert compliance["adr_004_fp8_model"] is True  # Model path contains FP8

            # Test ADR-010 (performance optimization)
            assert "adr_010_performance_optimization" in compliance
            assert compliance["adr_010_performance_optimization"] is True

            # Test ADR-011 (modern parameters)
            assert "adr_011_modern_parameters" in compliance
            assert compliance["adr_011_modern_parameters"] is True

            # Test context support
            assert "context_128k_support" in compliance
            assert compliance["context_128k_support"] is True

    @patch("src.config.integrations.setup_llamaindex")
    def test_performance_stats_adr_reporting(self, mock_setup):
        """Test ADR compliance reporting in performance stats."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()

            stats = coordinator.get_performance_stats()

            # Test ADR compliance section
            assert "adr_compliance" in stats
            adr_compliance = stats["adr_compliance"]

            assert "adr_001" in adr_compliance
            assert "adr_004" in adr_compliance
            assert "adr_010" in adr_compliance
            assert "adr_011" in adr_compliance
            assert "adr_018" in adr_compliance

    @patch("src.config.integrations.setup_llamaindex")
    def test_reset_performance_stats(self, mock_setup):
        """Test performance statistics reset functionality."""
        with patch.object(Settings, "llm", mock_llamaindex_llm()):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock and run a query to accumulate stats
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                return_value=[
                    {
                        "messages": [HumanMessage(content="Response")],
                        "validation_result": {"confidence": 0.9},
                    }
                ]
            )

            coordinator.process_query("Test query")
            assert coordinator.total_queries == 1

            # Reset stats
            coordinator.reset_performance_stats()

            assert coordinator.total_queries == 0
            assert coordinator.successful_queries == 0
            assert coordinator.fallback_queries == 0
            assert coordinator.avg_processing_time == 0.0
            assert coordinator.avg_coordination_overhead == 0.0


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorFactoryFunction:
    """Test factory function for coordinator creation."""

    @patch("src.agents.coordinator.MultiAgentCoordinator")
    def test_factory_function_default_params(self, mock_coordinator_class):
        """Test factory function with default parameters."""
        from src.agents.coordinator import create_multi_agent_coordinator

        mock_instance = Mock()
        mock_coordinator_class.return_value = mock_instance

        result = create_multi_agent_coordinator()

        mock_coordinator_class.assert_called_once_with(
            model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
            max_context_length=131072,
            enable_fallback=True,
        )
        assert result == mock_instance

    @patch("src.agents.coordinator.MultiAgentCoordinator")
    def test_factory_function_custom_params(self, mock_coordinator_class):
        """Test factory function with custom parameters."""
        from src.agents.coordinator import create_multi_agent_coordinator

        mock_instance = Mock()
        mock_coordinator_class.return_value = mock_instance

        result = create_multi_agent_coordinator(
            model_path="custom/model", max_context_length=65536, enable_fallback=False
        )

        mock_coordinator_class.assert_called_once_with(
            model_path="custom/model", max_context_length=65536, enable_fallback=False
        )
        assert result == mock_instance


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentState:
    """Test MultiAgentState data model integration."""

    def test_state_initialization_with_defaults(self):
        """Test MultiAgentState initialization with proper defaults."""
        state = MultiAgentState(messages=[HumanMessage(content="Test message")])

        # Test that state can be created and accessed
        assert state is not None
        assert len(state["messages"]) == 1

        # Test default values (access as dict due to LangGraph MessagesState)
        assert state.get("tools_data", {}) == {}
        assert state.get("context") is None
        assert state.get("routing_decision", {}) == {}
        assert state.get("planning_output", {}) == {}
        assert state.get("retrieval_results", []) == []
        assert state.get("synthesis_result", {}) == {}
        assert state.get("validation_result", {}) == {}
        assert state.get("agent_timings", {}) == {}
        assert state.get("parallel_execution_active", False) is False
        assert state.get("token_reduction_achieved", 0.0) == 0.0

    def test_state_with_performance_data(self):
        """Test MultiAgentState with performance tracking data."""
        state = MultiAgentState(
            messages=[HumanMessage(content="Test")],
            agent_timings={"router_agent": 0.05, "retrieval_agent": 0.08},
            parallel_execution_active=True,
            token_reduction_achieved=0.65,
            context_trimmed=True,
            tokens_trimmed=1500,
            kv_cache_usage_gb=8.2,
        )

        # Verify performance data
        assert state["agent_timings"]["router_agent"] == 0.05
        assert state["agent_timings"]["retrieval_agent"] == 0.08
        assert state["parallel_execution_active"] is True
        assert state["token_reduction_achieved"] == 0.65
        assert state["context_trimmed"] is True
        assert state["tokens_trimmed"] == 1500
        assert state["kv_cache_usage_gb"] == 8.2
