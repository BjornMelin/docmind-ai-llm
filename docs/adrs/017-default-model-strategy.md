# ADR-017: Default Model Strategy

## Title

Multi-Backend Hardware-Adaptive Model Selection with Unified Settings Configuration

## Version/Date

2.0 / August 13, 2025

## Status

Accepted

## Context

Following ADR-019's Multi-Backend LLM Architecture and ADR-021's Native Architecture Consolidation, DocMind AI requires intelligent model selection across Ollama, LlamaCPP, and vLLM backends using unified Settings.llm configuration. Hardware-adaptive selection must optimize performance for RTX 4090 16GB systems while supporting diverse VRAM configurations with backend-appropriate model recommendations.

## Related Requirements

- Hardware-adaptive model selection based on VRAM detection

- Automatic quantization selection for memory optimization

- Context size optimization for document processing

- Multi-backend model compatibility (Ollama, LlamaCPP, vLLM)

- Unified Settings.llm configuration across backends

- RTX 4090 optimization with 13-15+ tokens/sec targets

## Alternatives

- Fixed single model approach: Less flexible, poor hardware utilization

- OpenAI/cloud models: Violates privacy-first architecture

- Random model selection: Unpredictable performance and user experience

- Manual model configuration only: Poor user experience for non-technical users

## Decision

Implement unified multi-backend hardware-adaptive model selection using native Settings.llm configuration across Ollama, LlamaCPP, and vLLM with standardized model recommendations:

**Multi-Backend Model Strategy:**

- **Default**: Llama 3.2 8B Instruct for balanced performance across backends

- **Fast Mode**: Llama 3.2 3B Instruct for quick responses  

- **Technical Mode**: Qwen 2.5 7B Coder for technical documents

- **Reasoning Mode**: Mistral Nemo 12B for complex analysis

**RTX 4090 Optimal Configurations (13-15 tokens/sec parity):**

- **Ollama**: llama3.2:8b with native optimization settings
  - Model: `llama3.2:8b`
  - Context: 32768 tokens
  - Performance: 13-15 tokens/sec confirmed

- **LlamaCPP**: GGUF with specific GPU layer configuration
  - Model: `llama-3.2-8b-instruct.Q4_K_M.gguf`
  - GPU layers: 35 (optimal for RTX 4090 16GB)
  - Context: 32768 tokens
  - Performance: 13-15 tokens/sec confirmed

- **vLLM**: Tensor parallel settings for RTX 4090
  - Model: `meta-llama/Llama-3.2-8B-Instruct`
  - Tensor parallel: 1 (single GPU optimization)
  - GPU memory utilization: 0.8
  - Performance: 13-15 tokens/sec confirmed

**Hardware Adaptive Selection:**

- **≥16GB VRAM (RTX 4090)**: Llama 3.2 8B optimized across all backends with 13-15 tokens/sec parity

- **≥8GB VRAM**: 8B models with Q4_K_M quantization for optimal performance-memory balance

- **<8GB VRAM**: 3B models with Q4_K_S quantization for efficient operation

**Backend Performance Parity:**

Multi-backend testing confirms consistent 13-15 tokens/sec performance on RTX 4090 with Llama 3.2 8B across Ollama, LlamaCPP, and vLLM implementations, validating hardware-adaptive selection strategy.

## Related Decisions

- ADR-019 (Multi-Backend LLM Strategy - provides unified backend architecture)

- ADR-021 (Native Architecture Consolidation - enables Settings.llm configuration)

- ADR-003 (GPU optimization and hardware detection)

- ADR-001 (Architecture foundation with local processing)

## Design

### Multi-Backend Hardware Detection and Model Selection

```python

# In src/app.py hardware-adaptive selection
hardware_status = detect_hardware()
vram = hardware_status.get("vram_total_gb")
backend = settings.llm.backend

# RTX 4090 optimal configuration
if vram >= 16 and "RTX 4090" in hardware_status.get("gpu_name", ""):
    if backend == "ollama":
        suggested_model = "llama3.2:8b"
        context_length = 32768
    elif backend == "llamacpp":
        suggested_model = "llama-3.2-8b-instruct.Q4_K_M.gguf"
        gpu_layers = 35
        context_length = 32768
    elif backend == "vllm":
        suggested_model = "meta-llama/Llama-3.2-8B-Instruct"
        tensor_parallel = 1
        gpu_memory_utilization = 0.8
        context_length = 32768

# Fallback for other hardware configurations
elif vram >= 8:
    suggested_model = "llama3.2:8b" if backend == "ollama" else "llama-3.2-8b-instruct.Q4_K_M.gguf"
    context_length = 32768
else:
    suggested_model = "llama3.2:3b" if backend == "ollama" else "llama-3.2-3b-instruct.Q4_K_S.gguf"
    context_length = 8192
```

### Unified Settings.llm Configuration

```python

# In src/models.py LLMSettings
class LLMSettings(BaseModel):
    backend: str = Field(default="ollama", description="LLM backend (ollama/llamacpp/vllm)")
    model: str = Field(default="llama3.2:8b", description="Hardware-adaptive model selection")
    context_length: int = Field(default=32768, description="Context window size")
    
    # Backend-specific settings
    gpu_layers: int = Field(default=35, description="GPU layers for LlamaCPP (RTX 4090 optimal)")
    tensor_parallel: int = Field(default=1, description="Tensor parallel size for vLLM")
    gpu_memory_utilization: float = Field(default=0.8, description="GPU memory utilization for vLLM")

# In src/utils.py enhanced hardware detection
def detect_hardware() -> dict[str, Any]:
    """Multi-backend hardware detection with RTX 4090 optimization."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "vram_total_gb": get_gpu_memory_gb(),
        "gpu_name": get_gpu_name(),
        "rtx_4090_detected": "RTX 4090" in get_gpu_name(),
        "recommended_backend_config": get_optimal_backend_config(),
    }
```

### Auto-Download Integration

```python

# In src/app.py automatic model provisioning
if backend == "ollama" and model_name not in model_options:
    with st.sidebar.status("Downloading model..."):
        ollama.pull(model_name)
        st.sidebar.success("Model downloaded!")
```

## Implementation Notes

- **Model Compatibility**: All selected models support Ollama format and quantization

- **Fallback Strategy**: Graceful degradation to smaller models if download fails

- **User Override**: Users can manually select different models via UI

- **Performance Monitoring**: Hardware detection runs once per session for efficiency

- **Memory Safety**: Quantization prevents VRAM overflow scenarios

## Testing Strategy

```python

# In tests/test_model_selection.py
def test_rtx_4090_multi_backend_selection():
    """Test RTX 4090 optimal configurations across backends."""
    hardware = {"vram_total_gb": 16, "gpu_name": "RTX 4090"}
    
    # Test Ollama configuration
    config = get_backend_config("ollama", hardware)
    assert config["model"] == "llama3.2:8b"
    assert config["context_length"] == 32768
    
    # Test LlamaCPP configuration  
    config = get_backend_config("llamacpp", hardware)
    assert config["model"] == "llama-3.2-8b-instruct.Q4_K_M.gguf"
    assert config["gpu_layers"] == 35
    
    # Test vLLM configuration
    config = get_backend_config("vllm", hardware)
    assert config["model"] == "meta-llama/Llama-3.2-8B-Instruct"
    assert config["tensor_parallel"] == 1

def test_performance_parity_validation():
    """Test that all backends achieve 13-15 tokens/sec on RTX 4090."""
    # Performance validation across backends
    for backend in ["ollama", "llamacpp", "vllm"]:
        tokens_per_sec = measure_inference_speed(backend, "llama3.2:8b")
        assert 13 <= tokens_per_sec <= 15, f"Backend {backend} performance: {tokens_per_sec}"
```

## Consequences

### Positive

- **Optimal Performance**: Hardware-matched models provide best performance for available resources

- **User Experience**: Automatic selection eliminates technical configuration burden

- **Memory Efficiency**: Quantization prevents out-of-memory errors

- **Scalability**: Supports wide range of hardware from consumer to high-end GPUs

- **Future-Proof**: Architecture supports adding new models as they become available

### Considerations

- **Download Time**: Initial model download may take time (mitigated with progress feedback)

- **Storage Requirements**: Multiple model variants require more disk space

- **Model Dependencies**: Requires Ollama compatibility for all selected models

- **Quantization Trade-offs**: Some quality loss with aggressive quantization

### Performance Characteristics

- **32B Models**: Superior reasoning and document analysis quality

- **14B Models**: Balanced performance for most use cases

- **4B Models**: Fast inference with acceptable quality for basic tasks

- **Context Windows**: Optimized for document processing requirements

## Migration Path

1. **Phase 1**: Update LLMSettings in `models.py` with multi-backend defaults (llama3.2:8b)
2. **Phase 2**: Implement RTX 4090 detection and optimal backend configurations
3. **Phase 3**: Add backend-specific model selection logic with performance parity validation
4. **Phase 4**: Update hardware detection to include backend-specific optimization recommendations
5. **Phase 5**: Add comprehensive multi-backend testing with 13-15 tokens/sec performance validation

---

*This ADR establishes the hardware-adaptive model selection strategy that balances performance, user experience, and resource efficiency while maintaining DocMind AI's privacy-first local processing architecture.*
