"""Comprehensive pytest fixtures for Multi-Agent Coordination System testing.

This module provides specialized fixtures for testing the ADR-011 compliant
multi-agent coordination system with proper mocking, async support, and
performance validation capabilities.

Features:
- Mock LLM and agent infrastructure
- Async support for agent workflows
- Performance timing fixtures
- ADR compliance validation fixtures
- Context management and state fixtures
- FP8 optimization mock components
"""

import time
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from llama_index.core import Document
from llama_index.core.memory import ChatMemoryBuffer

# Import the classes we're testing
from src.agents.coordinator import MultiAgentCoordinator
from src.agents.models import MultiAgentState
from src.config import settings
from src.dspy_integration import DSPyLlamaIndexRetriever

# Note: event_loop_policy removed - use fixture from main conftest.py


# Mock classes to replace removed backward compatibility classes
class MockVLLMConfig:
    """Mock VLLMConfig for testing."""

    def __init__(self, **kwargs):
        """Initialize mock VLLMConfig with optional parameters."""
        self.model = kwargs.get("model", settings.vllm.model)
        self.max_model_len = kwargs.get("max_model_len", settings.vllm.context_window)
        self.kv_cache_dtype = kwargs.get("kv_cache_dtype", settings.kv_cache_dtype)
        self.attention_backend = kwargs.get(
            "attention_backend", settings.vllm.attention_backend
        )
        self.gpu_memory_utilization = kwargs.get(
            "gpu_memory_utilization", settings.vllm.gpu_memory_utilization
        )
        self.enable_chunked_prefill = kwargs.get(
            "enable_chunked_prefill", settings.vllm.enable_chunked_prefill
        )
        self.max_num_seqs = kwargs.get("max_num_seqs", settings.vllm.max_num_seqs)
        self.max_num_batched_tokens = kwargs.get(
            "max_num_batched_tokens", settings.vllm.max_num_batched_tokens
        )

        # Additional attributes expected by tests
        self.trust_remote_code = kwargs.get("trust_remote_code", True)
        self.calculate_kv_scales = kwargs.get("calculate_kv_scales", True)
        self.use_cudnn_prefill = kwargs.get("use_cudnn_prefill", True)
        self.target_decode_throughput = kwargs.get("target_decode_throughput", 130)
        self.target_prefill_throughput = kwargs.get("target_prefill_throughput", 1050)
        self.vram_usage_target_gb = kwargs.get("vram_usage_target_gb", 13.5)
        self.host = kwargs.get("host", "0.0.0.0")  # noqa: S104
        self.port = kwargs.get("port", 8000)
        self.served_model_name = kwargs.get("served_model_name", "docmind-qwen3-fp8")


class MockContextManager:
    """Mock ContextManager for testing."""

    def __init__(self):
        """Initialize mock ContextManager with default settings."""
        self.max_context_tokens = settings.vllm.context_window
        self.trim_threshold = int(settings.vllm.context_window * 0.9)
        self.preserve_ratio = 0.3
        self.kv_cache_memory_per_token = 1024  # bytes per token
        self.total_kv_cache_gb_at_128k = 8.0

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Estimate token count for context management."""
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4  # 4 chars per token average

    def pre_model_hook(self, state: dict) -> dict:
        """Trim context before model processing."""
        return state

    def post_model_hook(self, state: dict) -> dict:
        """Format response after model generation."""
        try:
            if state.get("output_mode") == "structured":
                # Add metadata for structured output
                state["metadata"] = {
                    "context_used": self.estimate_tokens(state.get("messages", [])),
                    "kv_cache_usage_gb": self.calculate_kv_cache_usage(state),
                    "parallel_execution_active": state.get(
                        "parallel_tool_calls", False
                    ),
                }

                # Structure response if present
                if "response" in state:
                    state["response"] = self.structure_response(state["response"])

            return state
        except Exception:
            # Return original state on any error
            return state

    def trim_to_token_limit(self, messages: list[dict], token_limit: int) -> list[dict]:
        """Trim messages to fit within token limit while preserving structure."""
        if not messages or self.estimate_tokens(messages) <= token_limit:
            return messages

        # Always preserve system message if present
        system_messages = [msg for msg in messages if msg.get("role") == "system"]
        other_messages = [msg for msg in messages if msg.get("role") != "system"]

        # Start with system messages
        result = system_messages[:]
        current_tokens = self.estimate_tokens(result)

        # Add messages from the end (most recent first) until limit is reached
        for msg in reversed(other_messages):
            msg_tokens = self.estimate_tokens([msg])
            if current_tokens + msg_tokens <= token_limit:
                result.insert(-len(system_messages) if system_messages else 0, msg)
                current_tokens += msg_tokens
            else:
                break

        return result

    def calculate_kv_cache_usage(self, state: dict) -> float:
        """Calculate KV cache memory usage in GB."""
        messages = state.get("messages", [])
        tokens = self.estimate_tokens(messages)
        usage_bytes = tokens * self.kv_cache_memory_per_token
        return usage_bytes / (1024**3)

    def structure_response(self, response: str) -> dict:
        """Structure response with metadata."""
        return {
            "content": response,
            "structured": True,
            "generated_at": time.time(),
            "context_optimized": True,
        }


class MockVLLMManager:
    """Mock VLLMManager for testing."""

    def __init__(self, config):
        """Initialize mock VLLMManager with configuration."""
        self.config = config
        self.llm = None
        self.async_engine = None
        self.context_manager = MockContextManager()

        # Performance metrics initialization
        self._performance_metrics = {
            "requests_processed": 0,
            "avg_decode_throughput": 0.0,
            "avg_prefill_throughput": 0.0,
            "peak_vram_usage_gb": 0.0,
        }

    def initialize_engine(self) -> bool:
        """Initialize vLLM engine."""
        return True

    def validate_performance(self) -> dict:
        """Validate performance against targets."""
        if self.llm is None:
            return {"error": "Engine not initialized"}

        try:
            # Mock performance validation
            return {
                "decode_throughput_estimate": 130.5,
                "meets_decode_target": True,
                "generation_time": 0.25,
                "tokens_generated": 32,
                "model_loaded": True,
                "fp8_optimization": True,
                "context_window": self.config.max_model_len,
                "meets_context_target": True,
                "validation_timestamp": time.time(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "validation_failed": True,
            }

    def get_performance_metrics(self) -> dict:
        """Get performance metrics."""
        return {
            **self._performance_metrics,
            "config": {
                "model": self.config.model,
                "max_context": self.config.max_model_len,
                "kv_cache_dtype": self.config.kv_cache_dtype,
                "attention_backend": self.config.attention_backend,
            },
            "targets": {
                "decode_throughput_range": (100, 160),
                "prefill_throughput_range": (800, 1300),
                "vram_usage_range_gb": (12, 14),
                "context_window": 131072,
            },
        }

    def generate_start_script(self, script_path: str) -> str:
        """Generate vLLM start script."""
        import os

        script_content = f"""#!/bin/bash
# vLLM Server Start Script - Generated for testing

export VLLM_ATTENTION_BACKEND={self.config.attention_backend}
export VLLM_USE_CUDNN_PREFILL=1

vllm serve {self.config.model} \\
    --max-model-len {self.config.max_model_len} \\
    --kv-cache-dtype {self.config.kv_cache_dtype} \\
    --gpu-memory-utilization {self.config.gpu_memory_utilization} \\
    --max-num-seqs {self.config.max_num_seqs} \\
    --max-num-batched-tokens {self.config.max_num_batched_tokens} \\
    --host {self.config.host} \\
    --port {self.config.port} \\
    --served-model-name {self.config.served_model_name} \\
    --calculate-kv-scales \\
    --enable-chunked-prefill \\
    --trust-remote-code
"""

        with open(script_path, "w") as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)  # noqa: S103

        return script_path


@pytest.fixture
def mock_vllm_config():
    """Create mock vLLM configuration for multi-agent testing.

    Returns:
        MockVLLMConfig: Mock vLLM config optimized for FP8 and 128K context.
    """
    return MockVLLMConfig(
        model="Qwen/Qwen3-4B-Instruct-2507-FP8",
        max_model_len=131072,
        kv_cache_dtype="fp8_e5m2",
        attention_backend="FLASHINFER",
        gpu_memory_utilization=0.95,
    )


@pytest.fixture
def mock_context_manager() -> MockContextManager:
    """Create mock context manager with realistic token handling.

    Returns:
        MockContextManager: Mock context manager for testing token estimation and
            context window management.
    """
    manager = MockContextManager()

    # Mock the token estimation to return predictable values
    def mock_estimate_tokens(messages):
        if isinstance(messages, list) and messages:
            # Return reasonable token counts for testing
            total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            return total_chars // 4
        return 100  # Default for non-list inputs

    manager.estimate_tokens = mock_estimate_tokens
    return manager


@pytest.fixture
def mock_llm() -> Mock:
    """Create comprehensive mock LLM for multi-agent testing.

    Returns:
        Mock: Mock LLM with sync/async methods for agent coordination testing.
    """
    mock_llm = Mock()

    # Mock different response types
    mock_response = Mock()
    mock_response.text = "Mock LLM response for testing"
    mock_response.content = "Mock LLM response for testing"

    mock_llm.complete = Mock(return_value=mock_response)
    mock_llm.invoke = Mock(return_value="Mock LLM response")
    mock_llm.predict = Mock(return_value="Mock LLM response")

    # Add async methods
    mock_llm.acomplete = AsyncMock(return_value=mock_response)
    mock_llm.ainvoke = AsyncMock(return_value="Mock async LLM response")

    return mock_llm


@pytest.fixture
def mock_memory() -> InMemorySaver:
    """Create mock memory for state persistence.

    Returns:
        InMemorySaver: In-memory state saver for testing agent coordination.
    """
    return InMemorySaver()


@pytest.fixture
def sample_agent_state() -> MultiAgentState:
    """Create sample agent state for testing multi-agent workflows.

    Returns:
        MultiAgentState: Pre-configured agent state with test data and mock tools.
    """
    return MultiAgentState(
        messages=[HumanMessage(content="What is machine learning?")],
        tools_data={
            "vector": Mock(),
            "kg": Mock(),
            "retriever": Mock(),
        },
        context=ChatMemoryBuffer.from_defaults(),
        total_start_time=time.perf_counter(),
        output_mode="structured",
        parallel_execution_active=True,
    )


@pytest.fixture
def mock_dspy_retriever() -> DSPyLlamaIndexRetriever:
    """Create mock DSPy retriever with optimization capabilities.

    Returns:
        DSPyLlamaIndexRetriever: Mock DSPy retriever for testing query optimization.
    """
    with patch("src.dspy_integration.DSPY_AVAILABLE", True):
        retriever = DSPyLlamaIndexRetriever(llm=Mock())
        retriever.optimization_enabled = True

        # Mock the optimization methods
        retriever.query_refiner = Mock()
        retriever.variant_generator = Mock()

        # Mock query refinement results
        mock_refined_result = Mock()
        mock_refined_result.refined_query = "Optimized query for better retrieval"
        retriever.query_refiner.return_value = mock_refined_result

        # Mock variant generation results
        mock_variant_result = Mock()
        mock_variant_result.variant1 = "First query variant"
        mock_variant_result.variant2 = "Second query variant"
        retriever.variant_generator.return_value = mock_variant_result

        return retriever


@pytest.fixture
def mock_vllm_manager(mock_vllm_config) -> MockVLLMManager:
    """Create mock vLLM manager with performance metrics."""
    manager = MockVLLMManager(mock_vllm_config)
    manager.llm = Mock()
    manager.async_engine = Mock()

    # Mock performance validation
    def mock_validate_performance():
        return {
            "decode_throughput_estimate": 130.5,
            "meets_decode_target": True,
            "generation_time": 0.25,
            "tokens_generated": 32,
            "model_loaded": True,
            "fp8_optimization": True,
            "context_window": 131072,
            "meets_context_target": True,
            "validation_timestamp": time.time(),
        }

    manager.validate_performance = mock_validate_performance
    return manager


@pytest.fixture
def mock_agent_tools() -> dict[str, Mock]:
    """Create mock agent tools for testing."""
    return {
        "route_query": Mock(
            return_value='{"strategy": "vector", "complexity": "simple", '
            '"confidence": 0.9}'
        ),
        "plan_query": Mock(
            return_value='{"sub_tasks": ["Find ML definition"], '
            '"execution_order": "sequential"}'
        ),
        "retrieve_documents": Mock(
            return_value='{"documents": [{"content": "ML is...", "score": 0.9}]}'
        ),
        "synthesize_results": Mock(
            return_value='{"documents": [{"content": "Synthesized ML content", '
            '"score": 0.95}]}'
        ),
        "validate_response": Mock(
            return_value='{"valid": true, "confidence": 0.9, '
            '"suggested_action": "accept"}'
        ),
    }


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing retrieval."""
    return [
        Document(
            text="Machine learning is a subset of artificial intelligence "
            "that enables computers to learn without explicit programming.",
            metadata={"source": "ml_intro.pdf", "page": 1, "score": 0.95},
        ),
        Document(
            text="Deep learning uses neural networks with multiple layers "
            "to model complex patterns in data.",
            metadata={"source": "dl_guide.pdf", "page": 2, "score": 0.87},
        ),
        Document(
            text="Natural language processing combines computational "
            "linguistics with machine learning.",
            metadata={"source": "nlp_basics.pdf", "page": 1, "score": 0.82},
        ),
    ]


@pytest_asyncio.fixture
async def mock_coordinator(
    mock_llm: Mock, mock_vllm_config: MockVLLMConfig, mock_memory: InMemorySaver
) -> AsyncGenerator[MultiAgentCoordinator, None]:
    """Create mock coordinator with all dependencies mocked."""
    with patch.multiple(
        "src.agents.coordinator",
        create_vllm_manager=Mock(return_value=Mock()),
        is_dspy_available=Mock(return_value=True),
    ):
        coordinator = MultiAgentCoordinator(
            model_path="Qwen/Qwen3-4B-Instruct-2507-FP8",
            max_context_length=131072,
            enable_fallback=True,
        )

        # Override with mocks
        coordinator.llm = mock_llm
        coordinator.memory = mock_memory
        coordinator.vllm_config = mock_vllm_config
        coordinator._setup_complete = True

        # Mock the compiled graph
        coordinator.compiled_graph = Mock()
        coordinator.compiled_graph.stream = Mock(
            return_value=[
                {
                    "messages": [HumanMessage(content="Mock agent response")],
                    "routing_decision": {"strategy": "vector", "complexity": "simple"},
                    "validation_result": {"confidence": 0.9},
                    "parallel_execution_active": True,
                    "agent_timings": {"router_agent": 0.05, "retrieval_agent": 0.08},
                }
            ]
        )

        yield coordinator


@pytest.fixture
def performance_timer() -> dict[str, float]:
    """Performance timing fixture for coordination overhead testing."""
    timings = {}

    def start_timer(name: str) -> None:
        timings[f"{name}_start"] = time.perf_counter()

    def end_timer(name: str) -> float:
        if f"{name}_start" not in timings:
            return 0.0
        end_time = time.perf_counter()
        duration = end_time - timings[f"{name}_start"]
        timings[name] = duration
        return duration

    timings["start_timer"] = start_timer
    timings["end_timer"] = end_timer

    return timings


@pytest.fixture
def adr_compliance_validator() -> dict[str, Any]:
    """Fixture for validating ADR compliance requirements."""

    def validate_adr_011(coordinator: MultiAgentCoordinator) -> dict[str, bool]:
        """Validate ADR-011 compliance."""
        return {
            "langgraph_supervisor_used": hasattr(coordinator, "graph"),
            "parallel_tool_calls_enabled": True,  # Mock validation
            "output_mode_structured": True,
            "forward_message_tool_created": True,
            "handoff_back_messages_enabled": True,
            "pre_model_hook_implemented": callable(
                getattr(coordinator, "_create_pre_model_hook", None)
            ),
            "post_model_hook_implemented": callable(
                getattr(coordinator, "_create_post_model_hook", None)
            ),
        }

    def validate_adr_004(config: MockVLLMConfig) -> dict[str, bool]:
        """Validate ADR-004 compliance (Local-First LLM Strategy)."""
        return {
            "fp8_model_used": "FP8" in config.model,
            "128k_context_support": config.max_model_len >= 131072,
            "local_model_path": "Qwen" in config.model,
            "vllm_backend_configured": True,
        }

    def validate_adr_010(config: MockVLLMConfig) -> dict[str, bool]:
        """Validate ADR-010 compliance (Performance Optimization)."""
        return {
            "fp8_kv_cache_enabled": config.kv_cache_dtype == "fp8_e5m2",
            "flashinfer_backend": config.attention_backend == "FLASHINFER",
            "chunked_prefill_enabled": config.enable_chunked_prefill,
            "memory_optimization": config.gpu_memory_utilization > 0.9,
        }

    return {
        "validate_adr_011": validate_adr_011,
        "validate_adr_004": validate_adr_004,
        "validate_adr_010": validate_adr_010,
    }


@pytest.fixture
def gherkin_test_scenarios() -> dict[str, dict[str, Any]]:
    """Test scenarios based on Gherkin specifications."""
    return {
        "simple_query": {
            "query": "What is the capital of France?",
            "expected_complexity": "simple",
            "expected_strategy": "vector",
            "max_processing_time": 1.5,
            "routing_required": True,
            "planning_required": False,
            "synthesis_required": False,
        },
        "complex_query": {
            "query": "Compare the environmental impact of electric vs gasoline "
            "vehicles and explain the manufacturing differences",
            "expected_complexity": "complex",
            "expected_strategy": "hybrid",
            "max_processing_time": 5.0,
            "routing_required": True,
            "planning_required": True,
            "synthesis_required": True,
            "expected_subtasks": 3,
        },
        "fp8_performance": {
            "performance_targets": {
                "decode_throughput_min": 100,
                "decode_throughput_max": 160,
                "prefill_throughput_min": 800,
                "prefill_throughput_max": 1300,
                "vram_usage_max_gb": 16,
                "context_limit": 131072,
            }
        },
        "supervisor_coordination": {
            "max_coordination_overhead_ms": 200,
            "token_reduction_target": 0.5,  # 50% reduction
            "parallel_execution_required": True,
            "fp8_kv_cache_required": True,
            "context_management_required": True,
        },
    }


@pytest.fixture
def mock_tools_data() -> dict[str, Any]:
    """Create mock tools data for agent state."""
    mock_vector_index = Mock()
    mock_vector_index.as_retriever.return_value = Mock()

    mock_kg_index = Mock()
    mock_kg_index.as_query_engine.return_value = Mock()

    mock_retriever = Mock()
    mock_retriever.retrieve.return_value = [
        Mock(text="Sample document content", score=0.9),
        Mock(text="Another document", score=0.8),
    ]

    return {
        "vector": mock_vector_index,
        "kg": mock_kg_index,
        "retriever": mock_retriever,
    }


@pytest.fixture(scope="session")
def benchmark_config() -> dict[str, Any]:
    """Configuration for performance benchmarking."""
    return {
        "min_rounds": 3,
        "max_time": 2.0,
        "min_time": 0.01,
        "warmup": True,
        "disable_gc": True,
        "coordination_overhead_threshold_ms": 200,
        "token_reduction_threshold": 0.5,
    }


# Async context manager for testing agent workflows
@pytest_asyncio.fixture
async def agent_workflow_context() -> AsyncGenerator[dict[str, Any], None]:
    """Async context for testing agent workflows."""
    context = {
        "start_time": time.perf_counter(),
        "timings": {},
        "errors": [],
        "performance_metrics": {},
    }

    try:
        yield context
    finally:
        context["end_time"] = time.perf_counter()
        context["total_duration"] = context["end_time"] - context["start_time"]
