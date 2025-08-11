# ADR-017: Default Model Strategy

## Title

Hardware-Adaptive Default Model Selection with Gemma 3n and Nvidia Nemotron

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

DocMind AI requires intelligent default model selection that automatically adapts to user hardware capabilities. The application should provide optimal performance across different VRAM configurations while maintaining consistent user experience. The selected models must support the application's core features: document analysis, multi-agent coordination, and local processing.

## Related Requirements

- Hardware-adaptive model selection based on VRAM detection

- Automatic quantization selection for memory optimization

- Context size optimization for document processing

- Support for multi-agent workflows via LangGraph

- Local processing with Ollama backend compatibility

## Alternatives

- Fixed single model approach: Less flexible, poor hardware utilization

- OpenAI/cloud models: Violates privacy-first architecture

- Random model selection: Unpredictable performance and user experience

- Manual model configuration only: Poor user experience for non-technical users

## Decision

Implement tiered hardware-adaptive model selection with Google Gemma 3n as the base model and Nvidia OpenReasoning Nemotron for high-VRAM configurations:

**Model Hierarchy:**

- **≥16GB VRAM**: `nvidia/OpenReasoning-Nemotron-32B-Q4_K_M` (65536 context)

- **≥8GB VRAM**: `nvidia/OpenReasoning-Nemotron-14B-Q8_0` (32768 context)  

- **<8GB VRAM**: `google/gemma-3n-E4B-it-Q4_K_S` (8192 context)

- **Default fallback**: `google/gemma-3n-E4B-it` (8192 context)

**Automatic Quantization:**

- Q4_K_M for 32B models (optimal quality/memory trade-off)

- Q8_0 for 14B models (higher precision with sufficient VRAM)

- Q4_K_S for minimal VRAM scenarios (maximum compression)

## Related Decisions

- ADR-001 (Architecture foundation with local processing)

- ADR-003 (GPU optimization and hardware detection)

- ADR-011 (Multi-agent coordination requirements)

## Design

### Hardware Detection and Model Selection

```python

# In src/app.py hardware-adaptive selection
hardware_status = detect_hardware()
vram = hardware_status.get("vram_total_gb")

# Model selection logic
if vram >= 16:
    suggested_model = "nvidia/OpenReasoning-Nemotron-32B"
    quant_suffix = "-Q4_K_M"
    suggested_context = 65536
elif vram >= 8:
    suggested_model = "nvidia/OpenReasoning-Nemotron-14B"
    quant_suffix = "-Q8_0"
    suggested_context = 32768
else:
    suggested_model = "google/gemma-3n-E4B-it"
    quant_suffix = "-Q4_K_S" if vram else ""
    suggested_context = 8192
```

### Configuration Integration

```python

# In src/models.py AppSettings
default_model: str = Field(
    default="google/gemma-3n-E4B-it",
    description="Default base model with hardware-adaptive variants"
)

# In src/utils.py hardware detection
def detect_hardware() -> dict[str, Any]:
    """Enhanced hardware detection for model selection."""
    return {
        "cuda_available": torch.cuda.is_available(),
        "vram_total_gb": get_gpu_memory_gb(),
        "gpu_name": get_gpu_name(),
        "recommended_model": get_recommended_model(),
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
def test_hardware_adaptive_selection():
    """Test model selection adapts to hardware."""
    # Mock different VRAM scenarios
    assert get_model_for_vram(16) == "nvidia/OpenReasoning-Nemotron-32B-Q4_K_M"
    assert get_model_for_vram(8) == "nvidia/OpenReasoning-Nemotron-14B-Q8_0"
    assert get_model_for_vram(4) == "google/gemma-3n-E4B-it-Q4_K_S"

def test_model_download_fallback():
    """Test graceful fallback on download failure."""
    # Simulate download failure and verify fallback
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

1. **Phase 1**: Update `models.py` default from `qwen2.5:7b` to `google/gemma-3n-E4B-it`
2. **Phase 2**: Implement hardware detection in model selection UI
3. **Phase 3**: Add automatic quantization suffix logic
4. **Phase 4**: Update `.env.example` with new defaults
5. **Phase 5**: Add comprehensive testing for all hardware scenarios

---

*This ADR establishes the hardware-adaptive model selection strategy that balances performance, user experience, and resource efficiency while maintaining DocMind AI's privacy-first local processing architecture.*
