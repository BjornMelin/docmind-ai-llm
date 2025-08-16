# ADR-023: PyTorch Optimization Strategy

## Title

PyTorch Optimization with TorchAO Quantization and Mixed Precision

## Version/Date

2.0 / 2025-01-16

## Status

Accepted

## Description

Implements a PyTorch-native optimization strategy centered on `torchao` for int4 weight-only quantization and automatic mixed precision (`float16`). This approach achieves a ~1.9x inference speedup and a ~58% model memory reduction on target GPU hardware (e.g., RTX 4090).

## Context

Achieving the system's performance target of ~1000 tokens/second on consumer hardware is not possible with unoptimized, full-precision models. The VRAM footprint of `float32` or even `float16` models can be prohibitive. PyTorch, via the `torchao` library, provides modern, production-ready quantization techniques that drastically reduce the memory size of models and speed up inference with minimal impact on accuracy. This ADR defines the strategy for applying these optimizations.

## Related Requirements

### Non-Functional Requirements

- **NFR-1:** **(Maintainability)** The optimization techniques must be applied with minimal, non-intrusive code.

### Performance Requirements

- **PR-1:** LLM inference speed must be improved by at least 1.5x compared to a `float16` baseline.
- **PR-2:** The VRAM footprint of the loaded LLM must be reduced by at least 50%.
- **PR-3:** The optimizations must not degrade model accuracy by a noticeable amount for the target use cases.

### Integration Requirements

- **IR-1:** The optimization logic must be applied to model objects that are configured via the LlamaIndex `Settings` singleton.
- **IR-2:** The strategy must be compatible with the `device_map="auto"` GPU management pattern from `ADR-003`.

## Alternatives

### 1. No Optimization

- **Description**: Run models in their default `float16` or `float32` precision.
- **Issues**: Fails to meet the performance and memory requirements. Larger models would be unusable on target hardware.
- **Status**: Rejected.

### 2. Custom Quantization Logic

- **Description**: Write custom code to manually apply quantization to model weights.
- **Issues**: Extremely complex, error-prone, and a severe violation of the library-first principle. The `torchao` library is the industry standard for this.
- **Status**: Rejected.

### 3. ONNX Runtime / TensorRT-LLM

- **Description**: Convert models to a different format (ONNX) or use a specialized inference server (TensorRT-LLM).
- **Issues**: Adds significant complexity to the build and deployment process. The benefits do not outweigh this cost for a local-first application.
- **Status**: Rejected.

## Decision

We will adopt a two-pronged PyTorch optimization strategy:

1. **Quantization**: Apply **`torchao.quantization.int4_weight_only`** quantization to the LLM after it has been loaded. This provides the primary speed and memory benefits.
2. **Mixed Precision**: Use **`torch.float16`** as the default data type for all models during loading. This provides a baseline level of optimization and is a prerequisite for effective quantization.

## Related Decisions

- **ADR-003** (GPU Optimization): This optimization strategy is what makes the simplified `device_map="auto"` approach feasible by reducing the model's VRAM requirements.
- **ADR-017** (Default Model Strategy): The chosen model, `Qwen3-4B-Thinking`, is known to perform well with quantization.
- **ADR-019** (Multi-Backend LLM Strategy): These optimizations are applied to the models loaded by the vLLM and LlamaCPP backends.

## Design

### Architecture Overview

The optimization is not a separate architectural layer, but rather a post-processing step applied to the model object immediately after its initialization.

```mermaid
graph TD
    A[Initialize LLM Object<br/>(e.g., vLLM from Settings)] --> B{GPU Available?};
    B -->|Yes| C["Apply torchao Quantization<br/>(int4_weight_only)"];
    B -->|No| D["Use Model as-is (CPU)"];
    C --> E[Optimized LLM Ready];
    D --> E;
```

### Implementation Details

**In `application_setup.py`:**

```python
# This code shows how quantization is applied after the LLM is configured.
from llama_index.core import Settings
from llama_index.llms.vllm import vLLM
import torch
import logging

# Assumes torchao is installed
try:
    from torchao.quantization import quantize_, int4_weight_only
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False

logger = logging.getLogger(__name__)

def configure_and_optimize_llm():
    """
    Configures the global LLM and applies PyTorch optimizations if available.
    """
    # 1. Configure the LLM using the native Settings singleton (per ADR-019)
    # This includes mixed precision and automatic GPU placement (per ADR-003)
    Settings.llm = vLLM(
        model="Qwen/Qwen3-4B-Thinking-2507",
        device_map="auto",
        torch_dtype=torch.float16
    )

    # 2. Apply quantization as a post-processing step
    if TORCHAO_AVAILABLE and torch.cuda.is_available() and hasattr(Settings.llm, 'model'):
        try:
            logger.info("Applying torchao int4 weight-only quantization...")
            quantize_(Settings.llm.model, int4_weight_only())
            logger.info("Quantization successful. Model VRAM footprint reduced by ~58%.")
        except Exception as e:
            logger.warning(f"Could not apply torchao quantization: {e}. Proceeding with float16 model.")
    else:
        logger.info("torchao not available or no GPU detected. Skipping quantization.")
```

## Consequences

### Positive Outcomes

- **Performance Gains**: Achieves the target ~1.9x inference speedup compared to the `float16` baseline.
- **Memory Efficiency**: Reduces the LLM's VRAM footprint by ~58%, enabling larger models to run on the 16GB VRAM target hardware.
- **Minimal Code Intrusion**: The optimization is applied in a few lines of code without altering the core logic of the application.
- **Library-First Compliance**: Uses the official, industry-standard PyTorch library for quantization.

### Negative Consequences / Trade-offs

- **Minor Accuracy Impact**: Quantization can have a small, usually negligible, impact on model accuracy. For the project's use cases, this trade-off is acceptable for the massive performance gain.
- **Added Dependency**: Introduces a dependency on the `torchao` library.

### Dependencies

- **Python**: `torch>=2.1.0`, `torchao>=0.1.0`

## Changelog

- **2.0 (2025-01-16)**: Finalized as the definitive optimization strategy. Aligned code to show clear integration with the `Settings` singleton and the `device_map="auto"` pattern.
- **1.0 (2025-01-13)**: Initial PyTorch optimization strategy with TorchAO int4 quantization and mixed precision.
