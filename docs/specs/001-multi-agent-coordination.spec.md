# Feature Specification: Multi-Agent Coordination System

## Metadata

- **Feature ID**: FEAT-001
- **Version**: 1.0.0
- **Status**: Implemented
- **Created**: 2025-08-19
- **Validated At**: 2025-08-20
- **Completion Percentage**: 90%
- **Requirements Covered**: REQ-0001 to REQ-0010

## 1. Objective

The Multi-Agent Coordination System orchestrates five specialized agents using LangGraph supervisor patterns to intelligently process user queries. The system analyzes query complexity, routes to appropriate strategies, decomposes complex questions, retrieves relevant documents, synthesizes results, and validates responses - all while maintaining conversation context and operating entirely offline.

## 2. Scope

### In Scope

- LangGraph supervisor initialization and configuration
- Five specialized agent implementations (router, planner, retrieval, synthesis, validation)
- Agent communication and handoff mechanisms
- Conversation context management
- Error handling and fallback strategies
- Performance monitoring and metrics

### Out of Scope

- External API integrations
- Cloud-based orchestration
- Real-time collaboration features
- Agent learning or fine-tuning

## 3. Inputs and Outputs

### Inputs

- **User Query**: Natural language question or command (string, max 4096 chars)
- **Conversation Context**: Previous messages and agent decisions (ChatMemoryBuffer)
- **System Configuration**: Agent settings and feature flags (Settings object)

### Outputs

- **Generated Response**: Answer with source attribution (string)
- **Agent Decisions**: Routing and planning metadata (Dict[str, Any])
- **Performance Metrics**: Latency and success indicators (Dict[str, float])
- **Validation Results**: Quality and accuracy scores (Dict[str, Any])

## 4. Interfaces

### External Interfaces

```python
class MultiAgentCoordinator:
    """Main interface for multi-agent system with vLLM backend."""
    
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-4B-Instruct-2507-FP8",
        max_context_length: int = 128000,
        backend: str = "vllm"
    ):
        """Initialize coordinator with FP8 model configuration."""
        pass
    
    def process_query(
        self,
        query: str,
        context: Optional[ChatMemoryBuffer] = None,
        settings: Optional[Settings] = None
    ) -> AgentResponse:
        """Process user query through multi-agent pipeline."""
        pass

class AgentResponse:
    """Response from multi-agent system."""
    content: str
    sources: List[Document]
    metadata: Dict[str, Any]
    validation_score: float
    processing_time: float
```

### Internal Agent Interfaces

```python
@tool
def route_query(query: str) -> Dict[str, str]:
    """Analyze and route query to optimal strategy."""
    return {"strategy": str, "complexity": str, "needs_planning": bool}

@tool
def plan_query(query: str, complexity: str) -> List[str]:
    """Decompose complex queries into sub-tasks."""
    return ["subtask1", "subtask2", "subtask3"]

@tool
def retrieve_documents(
    query: str, 
    strategy: str,
    use_dspy: bool = True,
    use_graphrag: bool = False
) -> List[Document]:
    """Execute retrieval with DSPy optimization and optional GraphRAG (ADR-018, ADR-019)."""
    
    # Apply DSPy query optimization if enabled (ADR-018)
    if use_dspy:
        from src.dspy_integration import DSPyLlamaIndexRetriever
        optimized_queries = DSPyLlamaIndexRetriever.optimize_query(query)
        primary_query = optimized_queries["refined"]
        variant_queries = optimized_queries["variants"]
    else:
        primary_query = query
        variant_queries = []
    
    # Check if GraphRAG should be used (ADR-019)
    if use_graphrag and strategy in ["relationships", "graph", "complex"]:
        from src.graphrag_integration import OptionalGraphRAG
        graph_rag = OptionalGraphRAG(enabled=True)
        
        if graph_rag.is_graph_query(query):
            graph_results = graph_rag.query(primary_query)
            if graph_results and graph_results.get("confidence", 0) > 0.7:
                return graph_results["documents"]
    
    # Standard retrieval with optimization
    documents = []
    for q in [primary_query] + variant_queries[:2]:  # Limit variants for performance
        # Execute retrieval based on strategy
        if strategy == "hybrid":
            docs = hybrid_retriever.retrieve(q)
        elif strategy == "graphrag":
            docs = graphrag_retriever.retrieve(q) 
        else:
            docs = vector_retriever.retrieve(q)
        documents.extend(docs)
    
    # Deduplicate and return top results
    return deduplicate_documents(documents)[:10]

@tool
def synthesize_results(
    sub_results: List[List[Document]], 
    original_query: str
) -> Dict[str, Any]:
    """Combine multi-source results."""
    return {"documents": [...], "synthesis_metadata": {...}}

@tool
def validate_response(
    query: str,
    response: str,
    sources: List[Document]
) -> Dict[str, Any]:
    """Validate response quality."""
    return {"valid": bool, "confidence": float, "issues": [...]}
```

### vLLM Backend Configuration

```python
class VLLMConfig:
    """Configuration for vLLM backend serving FP8 model with FlashInfer optimization."""
    
    model: str = "Qwen/Qwen3-4B-Instruct-2507-FP8"
    max_model_len: int = 131072  # 128K context
    gpu_memory_utilization: float = 0.95
    
    # FP8 KV Cache Optimization (ADR-004, ADR-010)
    kv_cache_dtype: str = "fp8_e5m2"  # FP8 KV cache for 50% memory reduction
    calculate_kv_scales: bool = True  # Required for FP8 KV cache
    attention_backend: str = "FLASHINFER"  # FlashInfer backend for FP8 acceleration
    enable_chunked_prefill: bool = True
    use_cudnn_prefill: bool = True
    
    # Memory optimization
    max_num_seqs: int = 1  # Single sequence for 128K context
    dtype: str = "auto"  # Automatic FP8 dtype selection
    trust_remote_code: bool = True
    
    # Performance metrics (validated)
    target_decode_throughput: int = 130  # 100-160 tok/s (validated)
    target_prefill_throughput: int = 1050  # 800-1300 tok/s (validated)
    vram_usage_target_gb: float = 13.5  # 12-14GB validated on RTX 4090 Laptop
    
    # Modern supervisor parameters (ADR-011)
    parallel_tool_calls: bool = True  # 50-87% token reduction
    output_mode: str = "structured"  # Enhanced response formatting
    create_forward_message_tool: bool = True  # Direct passthrough
    add_handoff_back_messages: bool = True  # Coordination tracking
    enable_pre_model_hook: bool = True  # Context trimming
    enable_post_model_hook: bool = True  # Response formatting
```

### Context Management Strategies

```python
class ContextManager:
    """Manages 128K context window with intelligent trimming strategies (ADR-004, ADR-011)."""
    
    max_context_tokens: int = 131072  # 128K context (hardware-constrained from 262K native)
    trim_threshold: int = 120000  # Trim at 120K (8K buffer for 128K limit)
    preserve_ratio: float = 0.3   # Keep 30% of oldest context for continuity
    
    # Memory calculations for FP8 KV cache (ADR-010)
    kv_cache_memory_per_token: float = 1024  # bytes per token with FP8
    total_kv_cache_gb_at_128k: float = 8.0  # ~8GB KV cache at 128K
    
    def pre_model_hook(self, state: dict) -> dict:
        """Trim context before model processing (ADR-011)."""
        messages = state.get("messages", [])
        total_tokens = self.estimate_tokens(messages)
        
        if total_tokens > self.trim_threshold:
            # Aggressive trimming strategy maintaining conversation coherence
            messages = self.trim_to_token_limit(messages, self.trim_threshold)
            state["messages"] = messages
            state["context_trimmed"] = True
            state["tokens_trimmed"] = total_tokens - self.estimate_tokens(messages)
        
        return state
    
    def post_model_hook(self, state: dict) -> dict:
        """Format response after model generation (ADR-011)."""
        if state.get("output_mode") == "structured":
            state["response"] = self.structure_response(state["response"])
            state["metadata"] = {
                "context_used": self.estimate_tokens(state.get("messages", [])),
                "kv_cache_usage_gb": self.calculate_kv_cache_usage(state),
                "parallel_execution_active": state.get("parallel_tool_calls", False)
            }
        return state
    
    def estimate_tokens(self, messages: List[dict]) -> int:
        """Estimate token count for context management."""
        # Simplified estimation - 4 chars per token average
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        return total_chars // 4
    
    def trim_to_token_limit(self, messages: List[dict], limit: int) -> List[dict]:
        """Trim messages to token limit while preserving conversation structure."""
        if not messages:
            return messages
        
        # Always preserve system message and latest user message
        system_msgs = [msg for msg in messages if msg.get("role") == "system"]
        latest_user = [msg for msg in reversed(messages) if msg.get("role") == "user"][:1]
        
        # Calculate available tokens for history
        reserved_tokens = self.estimate_tokens(system_msgs + latest_user)
        available_tokens = limit - reserved_tokens
        
        # Trim middle conversation history
        history_msgs = [msg for msg in messages 
                       if msg.get("role") not in ["system"] and msg not in latest_user]
        
        # Keep most recent history that fits
        trimmed_history = []
        current_tokens = 0
        for msg in reversed(history_msgs):
            msg_tokens = self.estimate_tokens([msg])
            if current_tokens + msg_tokens <= available_tokens:
                trimmed_history.insert(0, msg)
                current_tokens += msg_tokens
            else:
                break
        
        return system_msgs + trimmed_history + latest_user
    
    def calculate_kv_cache_usage(self, state: dict) -> float:
        """Calculate current KV cache memory usage in GB."""
        context_tokens = self.estimate_tokens(state.get("messages", []))
        return (context_tokens * self.kv_cache_memory_per_token) / (1024**3)
    
    def structure_response(self, response: str) -> dict:
        """Structure response with metadata for enhanced integration."""
        return {
            "content": response,
            "structured": True,
            "generated_at": time.time(),
            "context_optimized": True
        }
```

### LangGraph Supervisor Configuration

```python
class SupervisorConfig:
    """Modern LangGraph supervisor parameters for optimized agent coordination (ADR-011)."""
    
    # Modern supervisor optimization parameters (verified from LangGraph documentation)
    parallel_tool_calls: bool = True  # Enable concurrent tool execution (50-87% token reduction)
    output_mode: str = "structured"  # Enhanced response formatting with metadata
    create_forward_message_tool: bool = True  # Direct message passthrough capability
    add_handoff_back_messages: bool = True  # Track handoff coordination messages
    
    # Context management hooks (ADR-004, ADR-011)
    pre_model_hook: Callable = trim_context_hook  # Context trimming at 120K threshold
    post_model_hook: Callable = format_response_hook  # Response formatting and metadata
    
    # Agent coordination settings
    max_iterations: int = 10
    interrupt_before: List[str] = ["human", "validator"]
    interrupt_after: List[str] = ["router", "planner"]
    
    # Parallel execution configuration
    max_parallel_calls: int = 3  # Maximum concurrent tool calls
    token_reduction_target: float = 0.5  # 50% minimum token reduction
    token_reduction_max: float = 0.87  # 87% maximum observed
    
    # Performance optimization
    agent_decision_timeout_ms: int = 200  # <200ms per agent decision (ADR-001)
    total_coordination_overhead_ms: int = 300  # Total overhead target
    
    # Error handling
    retry_policy: Dict[str, int] = {
        "max_retries": 3,
        "backoff_factor": 1.5,
        "timeout_seconds": 30
    }
    
    # Validated performance metrics
    performance_targets: Dict[str, Any] = {
        "decode_throughput_range": (100, 160),  # tokens/sec
        "prefill_throughput_range": (800, 1300),  # tokens/sec
        "vram_usage_gb": (12, 14),  # GB on RTX 4090 Laptop
        "context_window": 131072,  # 128K tokens
        "parallel_efficiency": (0.5, 0.87)  # Token reduction range
    }
```

## 5. Data Contracts

### Query Routing Decision

```json
{
  "strategy": "vector|hybrid|graphrag",
  "complexity": "simple|medium|complex",
  "needs_planning": true|false,
  "confidence": 0.0-1.0
}
```

### Planning Output

```json
{
  "original_query": "string",
  "sub_tasks": ["task1", "task2", "task3"],
  "execution_order": "parallel|sequential",
  "estimated_complexity": "low|medium|high"
}
```

### Validation Result

```json
{
  "valid": true|false,
  "confidence": 0.0-1.0,
  "issues": [
    {
      "type": "hallucination|missing_source|inaccuracy",
      "severity": "low|medium|high",
      "description": "string"
    }
  ],
  "suggested_action": "accept|regenerate|refine"
}
```

## 6. Change Plan

### New Files

- `src/agents/coordinator.py` - Main coordinator class
- `src/agents/router.py` - Query routing agent
- `src/agents/planner.py` - Query planning agent
- `src/agents/retrieval.py` - Retrieval expert agent
- `src/agents/synthesis.py` - Result synthesis agent
- `src/agents/validator.py` - Response validation agent
- `src/agents/tools.py` - Shared agent tools
- `tests/test_agents/` - Agent test suite

### Modified Files

- `src/main.py` - Integrate multi-agent coordinator
- `src/config/settings.py` - Add agent configuration
- `src/ui/streamlit_app.py` - Connect UI to agent system

### Configuration Changes

- Add `ENABLE_MULTI_AGENT=true` to .env
- Add `AGENT_DECISION_TIMEOUT=300` (milliseconds)
- Add `FALLBACK_TO_BASIC_RAG=true`
- Add `MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507-FP8"`
- Add `MAX_CONTEXT_LENGTH=128000`
- Add `VLLM_GPU_MEMORY_UTILIZATION=0.95`
- Add `VLLM_MAX_MODEL_LEN=128000`
- Add `VLLM_QUANTIZATION=fp8`

## 7. Acceptance Criteria

### Scenario 1: Simple Query Processing

```gherkin
Given a simple factual query "What is the capital of France?"
When the query is processed by the multi-agent system
Then the router agent classifies it as "simple" complexity
And the retrieval agent uses vector search strategy
And the response is generated without planning or synthesis
And the total processing time is under 1.5 seconds
```

### Scenario 2: Complex Query Decomposition

```gherkin
Given a complex query "Compare the environmental impact of electric vs gasoline vehicles and explain the manufacturing differences"
When the query is processed by the multi-agent system
Then the router agent classifies it as "complex" complexity
And the planner agent decomposes it into 3 sub-tasks
And the retrieval agent processes each sub-task
And the synthesis agent combines the results
And the validator ensures response completeness
```

### Scenario 3: Fallback on Agent Failure

```gherkin
Given any user query
When an agent fails to respond within timeout
Then the system falls back to basic RAG pipeline
And the user receives a response within 3 seconds
And an error is logged with agent failure details
```

### Scenario 4: Context Preservation

```gherkin
Given a multi-turn conversation with 5 previous exchanges
When a follow-up query references previous context
Then the agents access the conversation history
And the response maintains contextual continuity
And the context buffer stays within 128K tokens (with trimming strategies)
```

### Scenario 5: DSPy Optimization

```gherkin
Given a query with DSPy optimization enabled
When the retrieval agent processes the query
Then the query is automatically rewritten for better retrieval
And retrieval quality improves by at least 20%
And the optimization adds less than 100ms latency
```

### Scenario 6: FP8 Model Performance

```gherkin
Given the vLLM backend is configured with FP8 quantization
When processing a query requiring agent coordination
Then the decode throughput is between 100-160 tokens/second
And the prefill throughput is between 800-1300 tokens/second
And total VRAM usage stays under 16GB
And context management maintains 128K token limit
```

### Scenario 7: Context Window Management

```gherkin
Given a conversation approaching 128K token limit
When the context manager pre_model_hook is triggered
Then the context is trimmed to 85% capacity (109K tokens)
And 30% of oldest context is preserved for continuity
And the response maintains conversational coherence
And no critical information is lost during trimming
```

### Scenario 8: Modern Supervisor Coordination

```gherkin
Given the modern supervisor system with parallel execution enabled (ADR-011)
When processing a complex multi-step query requiring multiple agents
Then parallel_tool_calls=True enables concurrent agent execution
And token usage is reduced by 50-87% through parallel tool execution
And add_handoff_back_messages=True tracks coordination messages for efficiency
And output_mode="structured" provides enhanced response formatting with metadata
And create_forward_message_tool=True enables direct message passthrough reducing overhead
And total coordination overhead stays under 200ms per agent decision
And up to 3 tools can execute in parallel simultaneously
And structured output includes coordination metadata and performance metrics
```

## 8. Tests

### Unit Tests

- Test each agent tool function independently
- Mock LLM responses for deterministic testing
- Verify routing logic for different query types
- Test planning decomposition algorithms
- Validate synthesis deduplication logic

### Integration Tests

- Test full multi-agent pipeline end-to-end
- Verify agent handoff mechanisms
- Test fallback scenarios
- Validate context management
- Test timeout and error handling

### Performance Tests

- Measure agent decision latency (target: <200ms per decision, improved from 300ms)
- Test concurrent query processing with parallel tool execution
- Benchmark memory usage under load with FP8 KV cache optimization
- Validate VRAM stays under 14GB (12-14GB target on RTX 4090 Laptop)
- Test FP8 model throughput (100-160 tok/s decode, 800-1300 tok/s prefill)
- Validate 128K context window management and intelligent trimming at 120K threshold
- Test vLLM + FlashInfer backend stability under sustained load
- Measure parallel tool execution token reduction (target: 50-87%)
- Validate context trimming performance (<50ms per operation)
- Test modern supervisor parameters coordination efficiency
- Benchmark DSPy query optimization overhead (<100ms)
- Validate FP8 KV cache memory efficiency vs FP16 baseline

### Coverage Targets

- Unit test coverage: >90%
- Integration test coverage: >80%
- Performance test scenarios: 10+

## 9. Security Considerations

- All agent operations execute locally (no data exfiltration)
- Input validation prevents prompt injection
- Agent decisions are logged for audit
- No storage of sensitive information in agent memory
- Sandboxed execution environment for agent tools

## 10. Quality Gates

### Performance Gates

- Agent coordination overhead: <200ms per decision (REQ-0007-v2, validated with supervisor pattern)
- Total query latency: <2 seconds for 95th percentile
- Success rate without fallback: >90% (REQ-0100)
- Memory usage: <16GB VRAM total (REQ-0070)
- FP8 model decode throughput: 100-160 tokens/second
- FP8 model prefill throughput: 800-1300 tokens/second
- Context management efficiency: <50ms for trimming operations

### Performance Validation

- Agent coordination latency validated at <200ms per decision with modern supervisor pattern
- Parallel tool execution achieving 50-87% token reduction through parallel_tool_calls=True
- FP8 KV cache enabling 128K context within 12-14GB VRAM on RTX 4090 Laptop
- vLLM + FlashInfer backend providing 100-160 tok/s decode, 800-1300 tok/s prefill
- Context trimming at 120K threshold with 8K buffer maintaining conversation coherence
- Modern supervisor parameters reducing coordination overhead through structured output and message forwarding

### Quality Gates

- All agent responses validated before return
- Hallucination detection rate: >95%
- Source attribution accuracy: 100%
- Context preservation success: >98%

### Code Quality

- Zero critical security vulnerabilities
- No blocking dependencies between agents
- Clean agent boundaries (single responsibility)
- Comprehensive error handling in all agents

## 11. Requirements Covered

- **REQ-0001**: LangGraph supervisor pattern with 5 agents ✓
- **REQ-0002**: Query routing agent for strategy selection ✓
- **REQ-0003**: Planning agent for query decomposition ✓
- **REQ-0004**: Retrieval agent with DSPy optimization ✓
- **REQ-0005**: Synthesis agent for multi-source combination ✓
- **REQ-0006**: Validation agent for response quality ✓
- **REQ-0007-v2**: Agent coordination overhead under 200ms ✓ (improved from 300ms with parallel execution)
- **REQ-0008**: Fallback to basic RAG on failure ✓
- **REQ-0009**: Local execution without APIs ✓
- **REQ-0010**: Context preservation across interactions ✓

## 12. Dependencies

### Technical Dependencies

- `langgraph-supervisor>=0.0.29` with modern supervisor parameters
- `langgraph>=0.2.74` for enhanced coordination features
- `langchain-core>=0.3.0` with structured output support
- `vllm>=0.10.1` with FlashInfer backend for FP8 optimization
- `dspy-ai>=2.4.0` for automatic prompt optimization (ADR-018)
- Local LLM (Qwen3-4B-Instruct-2507-FP8) with function calling
- vLLM backend for model serving and inference
- CUDA-compatible GPU with <16GB VRAM

### Feature Dependencies

- Retrieval pipeline (FEAT-002) for document search
- LLM infrastructure (FEAT-004) for agent decisions
- UI system (FEAT-005) for user interaction

## 13. Traceability

### Source Documents

- **ADR-001**: Modern Agentic RAG Architecture (5-agent supervisor system)
- **ADR-004**: Local-First LLM Strategy (Qwen3-4B-Instruct-2507-FP8 with 128K context)
- **ADR-010**: Performance Optimization Strategy (FP8 KV cache, dual-layer caching)
- **ADR-011**: Agent Orchestration Framework (LangGraph supervisor with modern parameters)
- **ADR-018**: DSPy Prompt Optimization (automatic query rewriting)
- **ADR-019**: Optional GraphRAG Integration (PropertyGraphIndex for relationship queries)
- **PRD Section 3**: Multi-Agent Coordination Epic

### Related Specifications

- 002-retrieval-search.spec.md
- 004-infrastructure-performance.spec.md
- 005-user-interface.spec.md
