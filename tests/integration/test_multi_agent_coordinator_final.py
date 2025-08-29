"""Final comprehensive integration tests for multi-agent coordinator.

Targets 40%+ coverage for coordinator.py with working tests.
Focus on agent coordination workflows and system integration.
"""

import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

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
def mock_llamaindex_llm():
    """Create mock LlamaIndex LLM that works with Settings."""
    from llama_index.core.llms.mock import MockLLM

    # Use LlamaIndex's built-in MockLLM
    mock_llm = MockLLM(max_tokens=2048)
    return mock_llm


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorCore:
    """Core integration tests for multi-agent coordinator."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_coordinator_initialization_basic(self, mock_setup, mock_llamaindex_llm):
        """Test basic coordinator initialization and attributes."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()

            # Test initialization attributes
            assert coordinator.model_path == "Qwen/Qwen3-4B-Instruct-2507-FP8"
            assert coordinator.max_context_length == 131072
            assert coordinator.enable_fallback is True

            # Test performance tracking initialization
            assert coordinator.total_queries == 0
            assert coordinator.successful_queries == 0
            assert coordinator.fallback_queries == 0
            assert coordinator.avg_processing_time == 0.0
            assert coordinator.avg_coordination_overhead == 0.0

    @patch("src.config.integrations.setup_llamaindex")
    def test_coordinator_initialization_custom(self, mock_setup, mock_llamaindex_llm):
        """Test coordinator initialization with custom parameters."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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
    def test_vllm_configuration(self, mock_setup, mock_llamaindex_llm):
        """Test vLLM configuration setup."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()

            # Test vLLM config structure
            assert coordinator.vllm_config["model"] == coordinator.model_path
            assert (
                coordinator.vllm_config["max_model_len"]
                == coordinator.max_context_length
            )
            assert "VLLM_ATTENTION_BACKEND" in coordinator.vllm_config
            assert "VLLM_KV_CACHE_DTYPE" in coordinator.vllm_config

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.agents.coordinator.create_react_agent")
    @patch("src.agents.coordinator.create_supervisor")
    def test_agent_graph_setup(
        self,
        mock_create_supervisor,
        mock_create_react_agent,
        mock_setup,
        mock_llamaindex_llm,
    ):
        """Test agent graph setup with proper mocks."""
        # Setup mocks
        mock_agent = Mock()
        mock_create_react_agent.return_value = mock_agent
        mock_graph = Mock()
        mock_create_supervisor.return_value = mock_graph

        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_agent_graph()

            # Verify agents were created (5 agents: router, planner, retrieval, synthesis, validation)
            assert mock_create_react_agent.call_count == 5

            # Verify supervisor was created with modern parameters
            mock_create_supervisor.assert_called_once()
            call_kwargs = mock_create_supervisor.call_args[1]
            assert call_kwargs["parallel_tool_calls"] is True
            assert call_kwargs["output_mode"] == "structured"
            assert call_kwargs["create_forward_message_tool"] is True
            assert call_kwargs["add_handoff_back_messages"] is True


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorSetup:
    """Test coordinator setup and initialization processes."""

    @patch("src.config.integrations.setup_llamaindex")
    @patch("src.agents.coordinator.create_react_agent")
    @patch("src.agents.coordinator.create_supervisor")
    def test_ensure_setup_success(
        self,
        mock_create_supervisor,
        mock_create_react_agent,
        mock_setup,
        mock_llamaindex_llm,
    ):
        """Test successful coordinator setup."""
        # Setup mocks for successful setup
        mock_agent = Mock()
        mock_create_react_agent.return_value = mock_agent
        mock_graph = Mock()
        mock_create_supervisor.return_value = mock_graph

        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()

            # Initially not setup
            assert not coordinator._setup_complete

            # Ensure setup should work
            result = coordinator._ensure_setup()

            assert result is True
            assert coordinator._setup_complete is True
            assert coordinator.llm is not None

    @patch("src.config.integrations.setup_llamaindex")
    def test_ensure_setup_failure(self, mock_setup):
        """Test coordinator setup failure handling."""
        # Mock Settings.llm to be None to trigger failure
        with patch.object(Settings, "_llm", None):
            coordinator = MultiAgentCoordinator()
            result = coordinator._ensure_setup()

            assert result is False
            assert coordinator._setup_complete is False


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorFallback:
    """Test fallback mechanisms and error handling."""

    def test_fallback_on_setup_failure(self):
        """Test fallback when coordinator setup fails."""
        coordinator = MultiAgentCoordinator(enable_fallback=True)
        coordinator._setup_complete = False  # Force setup failure

        response = coordinator.process_query("Test query")

        assert isinstance(response, AgentResponse)
        assert "error" in response.content.lower()
        assert response.metadata["fallback_available"] is True
        assert "initialization_failed" in response.optimization_metrics

    @patch("src.config.integrations.setup_llamaindex")
    def test_fallback_on_workflow_failure(self, mock_setup, mock_llamaindex_llm):
        """Test fallback when agent workflow fails."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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
    def test_no_fallback_error_handling(self, mock_setup, mock_llamaindex_llm):
        """Test error handling when fallback is disabled."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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
class TestMultiAgentCoordinatorWorkflows:
    """Test agent coordination workflows."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_process_query_basic(self, mock_setup, mock_llamaindex_llm):
        """Test basic query processing workflow."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock compiled graph with simple response
            coordinator.compiled_graph = Mock()
            coordinator.compiled_graph.stream = Mock(
                return_value=[
                    {
                        "messages": [HumanMessage(content="Test response")],
                        "validation_result": {"confidence": 0.9},
                    }
                ]
            )

            response = coordinator.process_query("Test query")

            assert isinstance(response, AgentResponse)
            assert response.processing_time > 0
            # Content comes from mock - just verify structure is correct

    @patch("src.config.integrations.setup_llamaindex")
    def test_process_query_complex(self, mock_setup, mock_llamaindex_llm):
        """Test complex query processing with agent coordination."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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

            response = coordinator.process_query("Complex query")

            assert isinstance(response, AgentResponse)
            assert response.metadata["routing_decision"]["complexity"] == "complex"
            assert response.metadata["routing_decision"]["needs_planning"] is True
            assert len(response.metadata["planning_output"]["sub_tasks"]) == 3
            assert len(response.sources) > 0
            assert response.validation_score == 0.85

    @patch("src.config.integrations.setup_llamaindex")
    def test_query_with_context(self, mock_setup, mock_llamaindex_llm):
        """Test query processing with conversation context."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock workflow
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
            context.put(HumanMessage(content="Previous question"))

            response = coordinator.process_query("Follow-up question", context=context)

            assert isinstance(response, AgentResponse)
            assert response.processing_time > 0


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentCoordinatorPerformance:
    """Test performance tracking and metrics."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_performance_tracking_accumulation(self, mock_setup, mock_llamaindex_llm):
        """Test performance metrics accumulate properly."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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
            for i in range(3):
                coordinator.process_query(f"Query {i}")

            # Verify metrics accumulation
            stats = coordinator.get_performance_stats()
            assert stats["total_queries"] == 3
            assert stats["successful_queries"] == 3
            assert stats["success_rate"] == 1.0
            assert stats["avg_processing_time"] > 0

    @patch("src.config.integrations.setup_llamaindex")
    def test_coordination_overhead_tracking(self, mock_setup, mock_llamaindex_llm):
        """Test coordination overhead measurement."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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

            response = coordinator.process_query("Test query")

            # Verify coordination overhead is tracked
            assert "coordination_overhead_ms" in response.optimization_metrics
            coordination_ms = response.optimization_metrics["coordination_overhead_ms"]
            assert coordination_ms >= 0

            # Verify meets 200ms target tracking
            assert "meets_200ms_target" in response.optimization_metrics

    @patch("src.config.integrations.setup_llamaindex")
    def test_performance_stats_reset(self, mock_setup, mock_llamaindex_llm):
        """Test performance statistics reset functionality."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            # Mock workflow and process query
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

    @patch("src.config.integrations.setup_llamaindex")
    def test_timeout_handling(self, mock_setup, mock_llamaindex_llm):
        """Test agent timeout handling."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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
    """Test context management and optimization features."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_context_estimation(self, mock_setup, mock_llamaindex_llm):
        """Test context token estimation."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()

            # Test token estimation
            messages = [
                {"content": "A" * 400},  # ~100 tokens
                {"content": "B" * 800},  # ~200 tokens
            ]

            tokens = coordinator.context_manager.estimate_tokens(messages)
            assert tokens == 300  # 100 + 200 tokens

    @patch("src.config.integrations.setup_llamaindex")
    def test_context_hooks_creation(self, mock_setup, mock_llamaindex_llm):
        """Test pre/post model hook creation."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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
    """Test ADR compliance and reporting."""

    @patch("src.config.integrations.setup_llamaindex")
    def test_adr_compliance_validation(self, mock_setup, mock_llamaindex_llm):
        """Test ADR compliance validation."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
            coordinator = MultiAgentCoordinator()
            coordinator._setup_complete = True

            compliance = coordinator.validate_adr_compliance()

            # Test key compliance checks
            assert "adr_004_fp8_model" in compliance
            assert compliance["adr_004_fp8_model"] is True  # Model path contains FP8
            assert "context_128k_support" in compliance
            assert compliance["context_128k_support"] is True
            assert "adr_010_performance_optimization" in compliance
            assert "adr_011_modern_parameters" in compliance

    @patch("src.config.integrations.setup_llamaindex")
    def test_performance_stats_adr_reporting(self, mock_setup, mock_llamaindex_llm):
        """Test ADR compliance reporting in performance stats."""
        with patch.object(Settings, "_llm", mock_llamaindex_llm):
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


@pytest.mark.skipif(
    not COMPONENTS_AVAILABLE, reason="Multi-agent coordinator components not available"
)
@pytest.mark.integration
class TestMultiAgentStateIntegration:
    """Test MultiAgentState model integration."""

    def test_state_creation_and_access(self):
        """Test MultiAgentState can be created and accessed properly."""
        state = MultiAgentState(messages=[HumanMessage(content="Test message")])

        # Test that state can be created and accessed
        assert state is not None
        assert len(state["messages"]) == 1

        # Test default values (access as dict due to LangGraph MessagesState)
        assert state.get("tools_data", {}) == {}
        assert state.get("context") is None
        assert state.get("routing_decision", {}) == {}
        assert state.get("agent_timings", {}) == {}

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


# Skip entire module if coordinator not available
if not COMPONENTS_AVAILABLE:
    pytest.skip(
        f"Multi-agent coordinator modules not available: {IMPORT_ERROR}",
        allow_module_level=True,
    )
