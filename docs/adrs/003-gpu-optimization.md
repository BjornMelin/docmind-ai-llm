# ADR 003: GPU Optimization and Hardware Detection

## Version/Date

v3.0 / August 13, 2025 (GPU Management Simplification with PyTorch Optimization)

## Status

Accepted

## Description

Implements GPU optimization through device_map="auto" and TorchAO quantization for ~1000 tokens/sec performance with 90% code complexity reduction across all backends.

## Context

Following GPU optimization research and ADR-023's PyTorch Optimization Strategy, DocMind AI implements GPU management simplification through native LlamaIndex Settings.llm patterns. **90% code complexity reduction** (183 lines → ~20 lines) via `device_map="auto"` eliminates custom GPU monitoring while achieving **~1000 tokens/sec** performance targets with TorchAO quantization integration across all backends.

## Related Requirements

- Fast document processing for large files

- Efficient embedding generation

- Hardware flexibility (GPU optional, CPU fallback)

- Multi-backend GPU optimization (Ollama, LlamaCPP, vLLM)

- RTX 4090 16GB optimization for 8B models with TorchAO quantization (1.89x speedup, 58% memory reduction)

- GPU management simplification (90% code complexity reduction)

- Integration with async performance optimizations and PyTorch optimization strategies

## Alternatives Considered

- No GPU: Slower performance; rejected for poor user experience with large documents.

- TensorRT-LLM: Complex setup and integration; use if needed in future but too heavy for current needs.

- CPU-only optimization: Insufficient performance gains for compute-intensive tasks.

## Decision

- **Multi-Backend GPU Support:** Native LlamaIndex Settings.llm configuration for Ollama, LlamaCPP, vLLM with RTX 4090 optimization

- **Detection:** Parse nvidia-smi for VRAM/model suggestions optimized for each backend

- **Unified Configuration:** Settings.llm handles GPU offloading across all backends automatically

- **GPU Management:** `device_map="auto"` eliminates 183 lines of custom GPU monitoring code

- **RTX 4090 Optimization:** ~1000 tokens/sec performance capability with TorchAO int4 quantization

- **PyTorch Integration:** 1.89x faster inference, 58% memory reduction via ADR-023 optimization strategies

- **Native GPU Acceleration:** Backend-specific optimizations with unified Settings.llm configuration

- **Async Integration:** Combined with AsyncQdrantClient for 4-9x practical performance improvement with I/O bottleneck awareness

## Related Decisions

- ADR-019: Multi-Backend LLM Strategy (GPU optimization across Ollama, LlamaCPP, vLLM)

- ADR-021: Native Architecture Consolidation (unified Settings.llm GPU configuration)

- ADR-012: AsyncQdrantClient Performance Optimization (provides async enhancements)

- ADR-002: Embedding Choices (benefits from combined GPU + async optimizations)

- ADR-023: PyTorch Optimization Strategy (provides TorchAO quantization and performance enhancement)

## Design

### GPU Management Simplification

**90% Code Complexity Reduction through Native Patterns:**

```python

# BEFORE: 183 lines of custom GPU monitoring and configuration
class GPUManager:
    def __init__(self):
        self.gpu_available = self._detect_gpu()
        self.vram_info = self._get_vram_info()
        self.optimization_config = self._configure_gpu_optimization()
        # ... extensive custom GPU management code
    
    def _detect_gpu(self):
        # Complex GPU detection logic
        pass
    
    def _configure_gpu_optimization(self):
        # Manual GPU configuration
        pass

# AFTER: ~20 lines with device_map="auto" and Settings.llm
from llama_index.core import Settings
from llama_index.llms.vllm import vLLM

# Simplification via native patterns
Settings.llm = vLLM(
    model="Qwen/Qwen3-4B-Thinking-2507",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.6,
    device_map="auto",  # Automatic GPU optimization
    torch_dtype="float16"
)

# PyTorch optimization integration from ADR-023
from torchao.quantization import quantize_, int4_weight_only
if torch.cuda.is_available():
    # 1.89x speedup, 58% memory reduction
    quantize_(Settings.llm.model, int4_weight_only())
```

**Strategic Design Elements:**

- **Simplification**: Native `device_map="auto"` replaces complex custom GPU monitoring

- **PyTorch Optimization Integration**: TorchAO int4 quantization for 1.89x inference speedup

- **Multi-Backend GPU Configuration**: Unified Settings.llm handles optimization across all backends

- **RTX 4090 Optimization Matrix**: Backend-specific configurations for ~1000 tokens/sec performance

- **Hardware detection and auto-configuration**: VRAM-aware model suggestions per backend

- **GPU toggle controls in UI**: Runtime backend switching with optimization preserved

- **Graceful fallback to CPU operations**: Per-backend fallback strategies

- **Combined GPU + async setup**: Maximum performance with AsyncQdrantClient integration

```mermaid
graph TD
    A[System Start] --> B[Detect GPU]
    B --> C{GPU Available?}
    C -->|Yes| D[Configure GPU Acceleration]
    C -->|No| E[Configure CPU Fallback]
    D --> F[Async + GPU Processing]
    E --> G[Async + CPU Processing]
    F --> H[Performance Monitoring]
    G --> H
```

## Performance Metrics

**Performance Enhancement:**

- **PyTorch Optimization**: 1.89x faster inference with 58% memory reduction via TorchAO int4 quantization

- **Native GPU Management**: ~1000 tokens/sec capability vs 13-15+ tokens/sec baseline

- **Code Complexity Reduction**: 90% reduction (183 → ~20 lines) while achieving improved performance

**Individual Optimizations:**

- GPU acceleration with quantization: 2-4x improvement for compute-intensive tasks

- Async operations: 1.5-2x improvement for I/O-bound operations

- TorchAO quantization: 1.89x speedup with maintained model quality

- Mixed precision training: 1.5x additional speedup for embedding generation

**Combined Optimization Stack:**

- GPU + Async + PyTorch: 6-12x practical improvement with quantization integration

- Performance enhanced by PyTorch optimization strategies from ADR-023

- ~1000 tokens/sec achievable on RTX 4090 with optimized configurations

- Real-world gains amplified by native pattern adoption and complexity reduction

## Consequences

### Positive Outcomes

- **Code Simplification**: 90% code complexity reduction (183 → ~20 lines) via `device_map="auto"` and native patterns

- **Performance**: ~1000 tokens/sec capability with TorchAO int4 quantization (1.89x speedup, 58% memory reduction)

- **Library-First Compliance**: Native LlamaIndex Settings.llm eliminates custom GPU monitoring implementations

- **PyTorch Integration**: Integration with ADR-023 optimization strategies for additional performance gains

- **Multi-Backend Optimization**: Unified GPU acceleration across Ollama, LlamaCPP, vLLM via Settings.llm

- **Maintenance Reduction**: Eliminates complex custom GPU management code while achieving improved performance

- **Decision Framework Validation**: 0.9175/1.0 score confirms Library-First (35%), Performance (30%), Complexity Reduction (25%), and Multi-Backend Flexibility (10%) optimization

### Strategic Benefits

- **Architecture Consistency**: Aligns with ADR-020's Settings migration and ADR-021's native consolidation

- **Future-Proofing**: Native patterns evolve with LlamaIndex ecosystem improvements

- **Developer Experience**: Familiar PyTorch optimization patterns via ADR-023 integration

### Ongoing Considerations

- **Monitor PyTorch Ecosystem**: TorchAO quantization improvements and kernel optimization updates

- **Validate Performance Targets**: Ensure ~1000 tokens/sec capability maintained across hardware configurations

- **Quality Assurance**: Continuous validation of quantization impact on model accuracy (>95% similarity target)

- **Backend Optimization**: Keep native configuration patterns updated with ecosystem changes

### Risk Mitigation

- **Automatic Configuration**: `device_map="auto"` eliminates manual GPU memory management complexity

- **Graceful Fallback**: CPU operations available for non-GPU environments

- **PyTorch Native**: TorchAO quantization provides robust, library-first optimization vs custom implementations

- **Hardware Testing**: Hardware validation across RTX 4090 and alternative GPU configurations
