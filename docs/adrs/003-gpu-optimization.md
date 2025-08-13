# ADR 003: GPU Optimization and Hardware Detection

## Version/Date

v2.0 / August 13, 2025 (Enhanced with multi-backend GPU optimization)

## Status

Accepted - Enhanced

## Context

Following ADR-019's Multi-Backend LLM Architecture and ADR-021's Native Architecture Consolidation, GPU optimization must support RTX 4090 16GB across all backends (Ollama, LlamaCPP, vLLM) with unified native Settings configuration. Performance critical for large docs/models with 13-15+ tokens/sec targets across backends.

## Related Requirements

- Fast document processing for large files

- Efficient embedding generation

- Hardware flexibility (GPU optional, CPU fallback)

- Multi-backend GPU optimization (Ollama, LlamaCPP, vLLM)

- RTX 4090 16GB optimization for 8B models

- Integration with async performance optimizations

## Alternatives Considered

- No GPU: Slower performance; rejected for poor user experience with large documents.

- TensorRT-LLM: Complex setup and integration; use if needed in future but too heavy for current needs.

- CPU-only optimization: Insufficient performance gains for compute-intensive tasks.

## Decision

- **Multi-Backend GPU Support:** Native LlamaIndex Settings.llm configuration for Ollama, LlamaCPP, vLLM with RTX 4090 optimization

- **Detection:** Parse nvidia-smi for VRAM/model suggestions optimized for each backend

- **Unified Configuration:** Settings.llm handles GPU offloading across all backends automatically

- **RTX 4090 Optimization:** 13-15+ tokens/sec performance targets for 8B models across all backends

- **Native GPU Acceleration:** Backend-specific optimizations (LlamaCPP n_gpu_layers=35, vLLM gpu_memory_utilization=0.8)

- **Async Integration:** Combined with AsyncQdrantClient for up to 4-5x total performance improvement

## Related Decisions

- ADR-019: Multi-Backend LLM Strategy (GPU optimization across Ollama, LlamaCPP, vLLM)

- ADR-021: Native Architecture Consolidation (unified Settings.llm GPU configuration)

- ADR-012: AsyncQdrantClient Performance Optimization (provides async enhancements)

- ADR-002: Embedding Choices (benefits from combined GPU + async optimizations)

## Design

- **Multi-Backend GPU Configuration**: Native Settings.llm handles GPU optimization automatically for each backend

- **RTX 4090 Optimization Matrix**: Backend-specific configurations for optimal 13-15+ tokens/sec performance

- **Hardware detection and auto-configuration**: VRAM-aware model suggestions per backend

- **GPU toggle controls in UI**: Runtime backend switching with GPU optimization preserved

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

## Consequences

- Positive: 2-3x speed gains from GPU acceleration; 50-80% from async operations; up to 4-5x total improvement for document processing.

- Negative: CPU fallback for non-GPU users; dual maintenance for GPU/CPU code paths.

- Risks: GPU memory management complexity (mitigated by auto-detection); compatibility issues (mitigated by fallback).

- Mitigations: Automatic hardware detection; graceful CPU fallback; comprehensive testing on various hardware configurations.
