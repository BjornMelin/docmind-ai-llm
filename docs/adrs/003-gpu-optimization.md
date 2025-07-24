# ADR 003: GPU Optimization and Hardware Detection

## Version/Date

v1.0 / July 22, 2025

## Status

Accepted

## Context

Performance critical for large docs/models; leverage NVIDIA GPUs without mandating them.

## Decision

- **Detection:** Parse nvidia-smi for VRAM/model suggestions (e.g., 72B for 16GB+).
- **Offload:** Full (n_gpu_layers=-1) for LlamaCpp; device='cuda' for embeddings/reranking.
- **Toggles:** UI checkbox; auto-config.
- **Efficiency:** PEFT for parameter-efficient loading.

## Rationale

- Auto-detection ensures usability; full offload achieves 50+ TPS on RTX 4090.
- PEFT reduces VRAM use for embeddings.

## Alternatives Considered

- No GPU: Slower; rejected.
- TensorRT-LLM: Complex; use if needed in future.

## Consequences

- Pros: 2-3x speed gains.
- Cons: CPU fallback for non-GPU users.
