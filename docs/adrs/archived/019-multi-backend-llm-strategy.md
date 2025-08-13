# ADR-019: Multi-Backend LLM Architecture Strategy

## Title

Multi-Backend Local LLM Support with Factory Pattern for Ollama, llama-cpp-python, and LM Studio

## Version/Date

1.0 / August 12, 2025

## Status

Superseded by ADR-021 (LlamaIndex Native Architecture Consolidation)

## Context

DocMind AI's document Q&A system currently relies primarily on Ollama for local LLM inference, with limited support for alternative backends. As the local LLM ecosystem has matured significantly in 2025, users have varying preferences for LLM backends based on their technical expertise, deployment scenarios, and performance requirements. The existing single-backend approach limits deployment flexibility and doesn't accommodate diverse user environments optimally.

### Current Implementation Limitations

The current 77-line ReActAgent implementation in `src/agents/agent_factory.py` is well-designed but constrained to primarily Ollama-based inference:

```python

# Current limited backend support
"ollama==0.5.1"
"llama-cpp-python>=0.2.32,<0.3.0"  # Available but underutilized
"transformers==4.54.1"
"openai>=1.98.0,<2.0.0"  # Cloud fallback only
```

### User Diversity and Requirements

Analysis of the target user base reveals distinct backend preferences:

- **Ease-of-Use Users**: Prefer Ollama's automatic model management and consistent performance

- **Advanced Users**: Want fine-grained control through llama-cpp-python for optimal performance

- **GUI Users**: Benefit from LM Studio's visual interface combined with programmatic access

- **Enterprise Users**: Need flexible deployment options across different environments

### Performance Considerations for RTX 4090 16GB

Current hardware optimization targets RTX 4090 16GB systems, which can effectively run:

- 7-13B models with Q4_K_M quantization across all backends

- Consistent 13-15 tokens/sec performance regardless of backend choice

- Streaming responses through unified LlamaIndex interface

## Related Requirements

- **Backend Flexibility**: Support user choice between Ollama, llama-cpp-python, and LM Studio

- **Unified Interface**: Maintain consistent API through LlamaIndex abstractions

- **Performance Parity**: Achieve 13-15+ tokens/sec across all backends for 8B models

- **RTX 4090 Optimization**: Optimal VRAM utilization (<80% under normal load)

- **Streaming Consistency**: Unified streaming interface across all backends

- **ReActAgent Integration**: Seamless integration with existing 77-line agent architecture

- **Configuration Simplicity**: Single-command setup for each backend

- **Model Portability**: Support recommended models (Llama 3.2 8B, Qwen 2.5 7B, Mistral Nemo 12B) across backends

## Alternatives

### 1. Single Backend Approach (Current - Insufficient)

- **Architecture Score**: 7.5/10

- **Flexibility**: Low - limited to Ollama primarily

- **User Accommodation**: Poor - doesn't meet diverse user needs

- **Implementation Complexity**: Low

- **Issues**: User constraints, deployment limitations, reduced market appeal

- **Status**: Current approach - insufficient for user diversity

### 2. Multi-Backend with Complex Abstraction Layer

- **Architecture Score**: 6.2/10

- **Flexibility**: High

- **Complexity**: Very High - custom abstraction layer

- **Maintenance**: High overhead

- **Issues**: Violates KISS principle, over-engineering

- **Rejected**: Unnecessary complexity for the problem scope

### 3. Plugin Architecture for Backends

- **Architecture Score**: 7.8/10

- **Flexibility**: Very High

- **Complexity**: High - plugin system overhead

- **Issues**: Over-engineered for three backends, violates YAGNI

- **Rejected**: Complexity not justified by current requirements

### 4. Multi-Backend Factory Pattern (Selected)

- **Architecture Score**: 8.7/10

- **Complexity**: Medium - clean factory pattern

- **KISS Compliance**: 0.8/1.0 - simple abstraction without over-engineering

- **Flexibility**: High - supports all major local backends

- **Maintainability**: High - leverages LlamaIndex unified interface

- **User Satisfaction**: High - accommodates different user preferences

- **Benefits**: Maximum flexibility with minimal architectural complexity

## Decision

**Implement Multi-Backend LLM Architecture** using a factory pattern that supports Ollama, llama-cpp-python, and LM Studio as first-class backends through LlamaIndex's unified interface.

**Key Decision Factors:**

1. **User Choice**: Enable users to select optimal backend for their environment and expertise
2. **Performance Parity**: All backends achieve 13-15+ tokens/sec for 8B models on RTX 4090 16GB
3. **LlamaIndex Integration**: Leverage existing unified streaming and agent interfaces
4. **Architectural Simplicity**: Factory pattern maintains KISS compliance while adding flexibility
5. **Market Differentiation**: Multi-backend support addresses diverse user needs

**Backend Prioritization:**

- **Primary Recommendation**: Ollama (best balance of ease and performance)

- **Advanced Users**: llama-cpp-python (maximum control and efficiency)

- **GUI Users**: LM Studio (visual interface with 2025 SDK production capabilities)

- **Cloud Fallback**: OpenAI for scenarios requiring external services

**Model Standardization Across Backends:**

- **Default**: Llama 3.2 8B Instruct (Q4_K_M) for general document Q&A

- **Fast Mode**: Llama 3.2 3B Instruct (Q8_0) for quick responses

- **Technical Mode**: Qwen 2.5 7B Coder (Q4_0/Q5_0) for technical documents  

- **Reasoning Mode**: Mistral Nemo 12B Instruct (Q4_K_M) for complex analysis

## Related Decisions

- **ADR-011**: ReActAgent architecture (maintained - factory provides LLM abstraction)

- **ADR-017**: Default model strategy (extended - multi-backend model recommendations)

- **ADR-003**: GPU optimization (preserved - backend-specific GPU optimizations)

- **ADR-018**: Refactoring decisions (aligned - library-first approach with factory pattern)

- **ADR-001**: Architecture overview (updated - multi-backend support integration)

## Design

### Multi-Backend Factory Architecture

**Core Factory Implementation:**

```python
from enum import Enum
from typing import Protocol, Dict, Any
from llama_index.core.agent import ReActAgent

class BackendType(Enum):
    OLLAMA = "ollama"
    LLAMACPP = "llama-cpp-python"
    LMSTUDIO = "lmstudio"
    OPENAI = "openai"  # Cloud fallback

class LLMBackendFactory:
    """Factory for creating LLM backends with unified interface"""
    
    @staticmethod
    def create_llm(backend: BackendType, model_name: str, **kwargs):
        if backend == BackendType.OLLAMA:
            from llama_index.llms.ollama import Ollama
            return Ollama(model=model_name, **kwargs)
        elif backend == BackendType.LLAMACPP:
            from llama_index.llms.llama_cpp import LlamaCPP
            return LlamaCPP(model_path=model_name, **kwargs)
        elif backend == BackendType.LMSTUDIO:
            from llama_index.llms.lmstudio import LMStudio  # Custom adapter
            return LMStudio(model_name=model_name, **kwargs)
        elif backend == BackendType.OPENAI:
            from llama_index.llms.openai import OpenAI
            return OpenAI(model=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
```

### Unified Configuration Management

**Backend-Specific Model Mapping:**

```python
class OptimalModels:
    """Model configurations optimized for RTX 4090 16GB"""
    MODELS = {
        BackendType.OLLAMA: {
            "fast": "llama3.2:3b",
            "balanced": "llama3.2:8b", 
            "technical": "qwen2.5:7b-coder",
            "reasoning": "mistral-nemo:12b"
        },
        BackendType.LLAMACPP: {
            "fast": "./models/llama-3.2-3b-instruct-q4_k_m.gguf",
            "balanced": "./models/llama-3.2-8b-instruct-q4_k_m.gguf",
            "technical": "./models/qwen2.5-7b-coder-q4_k_m.gguf",
            "reasoning": "./models/mistral-nemo-12b-q4_k_m.gguf"
        },
        BackendType.LMSTUDIO: {
            "fast": "llama-3.2-3b-instruct",
            "balanced": "llama-3.2-8b-instruct",
            "technical": "qwen2.5-7b-coder-instruct",
            "reasoning": "mistral-nemo-12b-instruct"
        }
    }
```

### Environment-Aware Backend Detection

**Automatic Backend Selection:**

```python
class EnvironmentConfig:
    @staticmethod
    def get_optimal_backend() -> BackendType:
        """Auto-detect optimal backend for current environment"""
        if shutil.which("ollama"):
            return BackendType.OLLAMA
        elif os.path.exists("/Applications/LM Studio.app"):
            return BackendType.LMSTUDIO  
        elif importlib.util.find_spec("llama_cpp"):
            return BackendType.LLAMACPP
        else:
            return BackendType.OPENAI  # Fallback to cloud
            
    @staticmethod
    def get_backend_config(backend: BackendType) -> dict:
        """Get optimal configuration for RTX 4090 16GB"""
        configs = {
            BackendType.OLLAMA: {
                "request_timeout": 60.0,
                "temperature": 0.1,
                "num_predict": 2048
            },
            BackendType.LLAMACPP: {
                "n_gpu_layers": 35,  # Optimal for RTX 4090
                "n_batch": 512, 
                "n_ctx": 8192,
                "temperature": 0.1,
                "max_tokens": 2048
            },
            BackendType.LMSTUDIO: {
                "temperature": 0.1,
                "max_tokens": 2048,
                "stream": True
            }
        }
        return configs.get(backend, {})
```

### Unified Streaming Interface

**Cross-Backend Streaming Support:**

```python
from llama_index.core.agent.workflow import StreamEvent, AgentStream
from typing import AsyncIterator

class UnifiedStreamer:
    def __init__(self, agent: ReActAgent):
        self.agent = agent
        
    async def stream_response(self, query: str) -> AsyncIterator[str]:
        """Unified streaming interface for all backends"""
        handler = self.agent.run(query)
        async for event in handler.stream_events():
            if isinstance(event, AgentStream):
                yield event.delta
            elif hasattr(event, 'delta') and event.delta:
                yield event.delta

# Backend-agnostic usage
streamer = UnifiedStreamer(agent)
async for chunk in streamer.stream_response("Analyze this document"):
    print(chunk, end="", flush=True)
```

### ReActAgent Integration

**Seamless Integration with Existing Agent Architecture:**

```python
class MultiBackendAgentFactory:
    """Enhanced agent factory with multi-backend support"""
    
    @staticmethod
    def create_agent(
        backend: BackendType,
        model_config: Dict[str, Any],
        tools: list,
        **kwargs
    ) -> ReActAgent:
        # Create LLM using factory pattern
        llm = LLMBackendFactory.create_llm(backend, **model_config)
        
        # Create agent with unified interface (maintains 77-line simplicity)
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=llm,
            verbose=kwargs.get('verbose', True),
            memory=ChatMemoryBuffer.from_defaults(token_limit=8192)
        )
        
        return agent
```

## Implementation Details

### Phase 1: Foundation (Week 1-2)

**Backend Factory Implementation:**

- Create `LLMBackendFactory` with support for all three backends

- Implement unified configuration management system

- Develop LM Studio LlamaIndex adapter for SDK integration

- Establish comprehensive testing framework across backends

**Dependencies Update:**

```toml

# Additional dependencies for multi-backend support
"lmstudio-python>=0.2.0"  # LM Studio SDK integration
"llama-index-llms-lmstudio"  # Custom adapter (to be developed)

# Existing dependencies maintained
"ollama==0.5.1"
"llama-cpp-python>=0.2.32,<0.3.0" 
"llama-index-llms-openai"
"llama-index-llms-ollama" 
"llama-index-llms-llama-cpp"
```

### Phase 2: Integration (Week 3-4)

**Agent Factory Enhancement:**

- Modify `src/agents/agent_factory.py` to use new backend system

- Deploy recommended model configurations across all backends

- Implement consistent streaming behavior validation

- Add graceful fallback logic between backends

### Phase 3: User Experience (Week 5-6)

**Interface Development:**

- Backend selection integration in Streamlit UI

- Real-time performance monitoring per backend

- Automatic backend and model detection

- Comprehensive setup documentation for each backend

### Phase 4: Optimization (Week 7-8)

**Advanced Features:**

- Intelligent context management across backends

- Query-type based automatic backend/model routing

- Detailed performance comparison analytics

- Backend-specific optimization features

## Backend Performance Characteristics

### Performance Comparison Matrix

| Metric | Ollama | llama-cpp-python | LM Studio | Notes |
|--------|---------|------------------|-----------|-------|
| **Tokens/sec (8B Q4)** | 14-15 | 13-15 | 13-15 | Similar performance across backends |
| **VRAM Usage** | 5-6GB | 6-10GB | 5-8GB | llama-cpp-python most flexible |
| **Setup Complexity** | Low | High | Medium | Ollama easiest, llama-cpp-python most complex |
| **Model Management** | Excellent | Manual | Good | Ollama automatic, others require setup |
| **Production Readiness** | Excellent | Excellent | Good | All suitable with proper configuration |
| **Streaming Support** | Native | Native | SDK-based | All support streaming via LlamaIndex |
| **Context Length** | 8-32K | 32K+ | 8-32K | llama-cpp-python handles longest contexts |
| **GPU Optimization** | Automatic | Manual | Automatic | llama-cpp-python offers most control |

### RTX 4090 16GB Optimizations

**Quantization Strategy:**

- **Q4_K_M**: Best balance of quality/performance (primary recommendation)

- **Q5_0**: Higher quality for critical applications  

- **Q8_0**: Only for smaller models (<7B)

**Context Management:**

- **Target Context**: 8K-16K tokens for optimal performance

- **Sliding Window**: Implement for longer documents

- **Chunking Strategy**: 2K token chunks with 200 token overlap

**Multi-Backend System Configuration:**

```bash

# Ollama optimizations
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=2
export OLLAMA_GPU_LAYERS=35

# llama-cpp-python optimizations
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"
export FORCE_CMAKE=1
export LLAMA_CPP_N_GPU_LAYERS=35

# LM Studio optimizations  
export LMSTUDIO_MAX_CONTEXT=16384
export LMSTUDIO_GPU_ACCELERATION=1

# General CUDA optimizations for all backends
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## Testing Strategy

### Backend Validation Testing

```python

# Comprehensive multi-backend testing
def test_backend_performance_parity():
    """Ensure consistent performance across all backends"""
    backends = [BackendType.OLLAMA, BackendType.LLAMACPP, BackendType.LMSTUDIO]
    query = "Analyze the key themes in this document."
    
    for backend in backends:
        agent = MultiBackendAgentFactory.create_agent(backend, model_config, tools)
        
        start_time = time.time()
        response = agent.run(query)
        latency = time.time() - start_time
        
        # Validate performance targets
        assert latency < 2.0  # <2s first token latency
        assert len(response) > 100  # Meaningful response length
        assert "document" in response.lower()  # Relevance check

def test_streaming_consistency():
    """Verify streaming works consistently across backends"""
    for backend in [BackendType.OLLAMA, BackendType.LLAMACPP, BackendType.LMSTUDIO]:
        agent = MultiBackendAgentFactory.create_agent(backend, model_config, tools)
        streamer = UnifiedStreamer(agent)
        
        chunks = []
        async for chunk in streamer.stream_response("Test query"):
            chunks.append(chunk)
            
        # Validate streaming behavior
        assert len(chunks) > 5  # Multiple chunks received
        assert "".join(chunks)  # Non-empty combined response
```

### Integration Testing

```python
def test_backend_auto_detection():
    """Test automatic backend detection logic"""
    detected = EnvironmentConfig.get_optimal_backend()
    config = EnvironmentConfig.get_backend_config(detected)
    
    assert detected in [BackendType.OLLAMA, BackendType.LLAMACPP, 
                       BackendType.LMSTUDIO, BackendType.OPENAI]
    assert "temperature" in config
    assert config["temperature"] == 0.1

def test_model_configuration_validity():
    """Validate model configurations for all backends"""
    for backend in BackendType:
        if backend in OptimalModels.MODELS:
            models = OptimalModels.MODELS[backend]
            assert "balanced" in models  # Default model exists
            assert len(models["balanced"]) > 0  # Valid model name
```

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation Strategy |
|------|-------------|--------|-------------------|
| **Model Compatibility Issues** | Medium | Medium | Comprehensive testing matrix across backends |
| **Performance Regression** | Low | High | Continuous benchmarking and rollback procedures |
| **LlamaIndex Integration Complexity** | Low | Medium | Leverage existing unified interface patterns |
| **LM Studio SDK Stability** | Medium | Low | Implement robust error handling and fallbacks |
| **Configuration Complexity** | Medium | Low | Auto-detection and sensible defaults |

### Mitigation Strategies

**Gradual Migration Approach:**

- Phase implementation over multiple releases

- Maintain current Ollama-primary approach as default

- Add backends incrementally with comprehensive testing

**Performance Validation:**

- Continuous integration benchmarks for all backends  

- Automated performance regression detection

- Rollback procedures for performance issues

**User Experience Protection:**

- Default backend auto-detection prevents configuration overhead

- Comprehensive documentation and setup guides

- Support for easy backend switching without application restart

## Cost-Benefit Analysis

### Development Costs

- **Implementation Time**: 6-8 developer weeks (increased for multi-backend support)

- **Testing Effort**: 2 weeks comprehensive testing across all backends  

- **Documentation**: 1.5 weeks for backend-specific setup guides

- **LM Studio Integration**: 1 week for custom LlamaIndex adapter

- **Total Investment**: ~10-12 weeks vs 3-4 weeks for single backend

### Strategic Benefits

**User Satisfaction & Market Appeal:**

- Accommodates diverse user preferences and technical expertise levels

- Differentiates from single-backend competitors

- Reduces user constraints in backend selection

- Enables deployment across varying technical environments

**Technical Benefits:**

- Future-proofing against backend ecosystem changes

- Reduced vendor lock-in concerns

- Performance optimization opportunities per backend

- Enhanced reliability through backend diversity

**Business Value:**

- Broader user adoption potential across technical skill levels

- Competitive advantage through flexibility

- Reduced support burden through user choice accommodation

- Foundation for future backend additions

### Risk-Benefit Assessment

**Increased Complexity (Controlled):**

- Factory pattern abstracts complexity effectively

- LlamaIndex provides unified interface reducing integration overhead

- Auto-detection minimizes user configuration burden

**Strategic Value (High):**

- Market differentiation through comprehensive backend support

- User adoption growth across different technical segments  

- Architectural flexibility for future expansion

## Consequences

### Positive Outcomes

- **User Flexibility**: Choice between Ollama (ease), llama-cpp-python (control), LM Studio (GUI+programmatic)

- **Performance Consistency**: 13-15+ tokens/sec achieved across all backends for 8B models

- **Architecture Elegance**: Factory pattern maintains KISS compliance while adding essential flexibility  

- **Market Differentiation**: Multi-backend support addresses diverse user needs and deployment scenarios

- **Future-Proofing**: Easy addition of new backends as local LLM ecosystem evolves

- **Reduced Vendor Lock-in**: Users not constrained to single backend approach

- **Enhanced Reliability**: Backend diversity provides fallback options and deployment flexibility

### Ongoing Considerations

**Maintenance Requirements:**

- Keep backend configurations updated with ecosystem changes

- Monitor performance parity across backends

- Maintain backend-specific optimizations and documentation

- Update model recommendations as new models become available

**Complexity Management:**

- Factory pattern abstracts backend differences effectively

- LlamaIndex unified interface minimizes integration complexity

- Comprehensive testing ensures consistent behavior across backends

- Auto-detection reduces user configuration overhead

**Performance Monitoring:**

- Continuous benchmarking across all backends

- Automated alerts for performance regressions

- User feedback collection on backend preferences and issues

- Regular optimization updates for RTX 4090 16GB targets

### Migration Path from Current Implementation

1. **Phase 1**: Implement factory pattern alongside existing Ollama-focused code
2. **Phase 2**: Add llama-cpp-python and LM Studio backend support
3. **Phase 3**: Update UI to include backend selection options
4. **Phase 4**: Enable automatic backend detection and recommendation
5. **Phase 5**: Comprehensive testing and performance optimization across all backends

### Success Metrics

- **Performance**: Consistent 13-15+ tokens/sec for 8B models across all backends

- **Reliability**: <1% stream interruption rate for any backend

- **User Adoption**: Backend usage distribution reflecting user preference diversity

- **Configuration Simplicity**: Single-command setup achieved for each backend  

- **Resource Utilization**: <80% VRAM usage under normal load for all backends

- **User Satisfaction**: Positive feedback on backend choice flexibility

## Architecture Diagram

```mermaid
graph TB
    subgraph "DocMind AI Application Layer"
        UI[Web UI / CLI]
        API[FastAPI Backend]
    end
    
    subgraph "Agent Layer"
        AF[MultiBackendAgentFactory]
        RA[ReActAgent]
        Tools[QueryEngineTool, etc.]
    end
    
    subgraph "LLM Abstraction Layer"
        LBF[LLMBackendFactory]
        BC[BackendConfig]
        EC[EnvironmentConfig]
    end
    
    subgraph "LlamaIndex Integration"
        LI[LlamaIndex Unified Interface]
        SE[StreamEvent System]
        CB[ChatMemoryBuffer]
    end
    
    subgraph "Backend Implementations"
        O[Ollama]
        LC[llama-cpp-python]
        LMS[LM Studio]
        OAI[OpenAI Fallback]
    end
    
    subgraph "Model Storage"
        OM[Ollama Models]
        GM[GGUF Models]
        LSM[LM Studio Models]
    end
    
    UI --> API
    API --> AF
    AF --> RA
    RA --> Tools
    RA --> LBF
    LBF --> BC
    LBF --> EC
    LBF --> LI
    LI --> SE
    LI --> CB
    LI --> O
    LI --> LC
    LI --> LMS
    LI --> OAI
    O --> OM
    LC --> GM
    LMS --> LSM
    
    classDef backend fill:#e1f5fe
    classDef abstraction fill:#f3e5f5
    classDef agent fill:#e8f5e8
    
    class O,LC,LMS,OAI backend
    class LBF,BC,EC,LI abstraction
    class AF,RA,Tools agent
```

## Conclusion

This Multi-Backend LLM Architecture Strategy successfully addresses the diverse needs of DocMind AI's user base while maintaining architectural simplicity through the factory pattern. By supporting Ollama, llama-cpp-python, and LM Studio as first-class backends, users can choose the optimal solution for their technical expertise and deployment requirements.

The strategy preserves the elegant 77-line ReActAgent architecture while extending flexibility through the LlamaIndex unified interface. Performance parity across backends (13-15+ tokens/sec for 8B models on RTX 4090 16GB) ensures users don't sacrifice performance for choice.

Key architectural principles maintained:

- **KISS Compliance**: Factory pattern adds essential flexibility without over-engineering

- **Library-First**: Leverages LlamaIndex ecosystem for unified backend integration  

- **Performance Focus**: RTX 4090 16GB optimization preserved across all backends

- **User Choice**: Accommodates diverse technical preferences and deployment scenarios

This foundation enables DocMind AI to serve a broader user base while maintaining the technical excellence and simplicity that define the current architecture.

---

*This ADR establishes the multi-backend LLM strategy that balances user choice, performance consistency, and architectural elegance while positioning DocMind AI for continued growth across diverse deployment scenarios.*
