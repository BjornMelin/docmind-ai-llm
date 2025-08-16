# ADR-017: Default Model Strategy

## Title

Hardware-Adaptive Model Selection with a Standardized Default LLM

## Version/Date

5.0 / 2025-01-16

## Status

Accepted

## Description

Standardizes the default reasoning LLM to `Qwen/Qwen3-4B-Thinking-2507`. The system will employ a hardware-adaptive strategy to automatically adjust context length and apply quantization, ensuring optimal performance and memory usage across a range of consumer hardware.

## Context

The choice of the core reasoning LLM is one of the most critical decisions for a RAG system, impacting performance, reasoning quality, and resource consumption. The system requires a default model that is a strong reasoner (for agentic workflows), has a large context window (for document analysis), and is efficient enough to run on consumer GPUs after optimization. `Qwen3-4B-Thinking` was selected for its benchmark-leading agentic capabilities and its large native context window, which performs well under quantization.

## Related Requirements

### Functional Requirements

- **FR-1:** The default LLM must be proficient at chain-of-thought reasoning and tool use for the ReAct agent.

### Non-Functional Requirements

- **NFR-1:** **(Usability)** The system must automatically configure the model for the user's hardware, avoiding complex manual setup.
- **NFR-2:** **(Flexibility)** The system must support multiple LLM backends (Ollama, LlamaCPP, vLLM).

### Performance Requirements

- **PR-1:** The model must be capable of achieving ~1000 tokens/second inference speed on target hardware after optimization.
- **PR-2:** The model's VRAM footprint must be manageable on consumer GPUs (e.g., <8GB on an RTX 4090 16GB).

### Integration Requirements

- **IR-1:** The model must be configured via the LlamaIndex `Settings.llm` singleton.

## Alternatives

### 1. Fixed Single Model Configuration

- **Description**: Use a single, static configuration for the model regardless of the user's hardware.
- **Issues**: This is not a viable approach. A configuration optimized for a high-end GPU would fail on a lower-end machine, while a configuration for a low-end machine would underutilize high-end hardware.
- **Status**: Rejected.

### 2. User-Only Manual Configuration

- **Description**: Require the user to manually set all model parameters (context length, quantization, etc.).
- **Issues**: Creates a poor user experience, especially for non-technical users. The system should provide a smart default.
- **Status**: Rejected.

## Decision

1. **Standard Model**: We will standardize on **`Qwen/Qwen3-4B-Thinking-2507`** as the default reasoning LLM for the v1.0 release.
2. **Hardware-Adaptive Configuration**: At startup, the application will detect the available VRAM and automatically apply an appropriate configuration profile (context length) to balance performance and capability.
3. **Optimization by Default**: PyTorch optimizations (quantization, mixed precision) will be applied by default on compatible hardware to ensure the best performance.

## Related Decisions

- **ADR-019** (Multi-Backend LLM Strategy): This ADR defines the standard model that is configured across all supported backends.
- **ADR-003** (GPU Optimization): This ADR provides the `device_map="auto"` pattern and the hardware detection logic that this adaptive strategy relies on.
- **ADR-023** (PyTorch Optimization Strategy): This ADR provides the quantization and mixed precision techniques that are applied to the model.
- **ADR-011** (LlamaIndex ReAct Agent Architecture): This model was chosen specifically to power the ReAct agent's reasoning capabilities.

## Design

### Architecture Overview

The hardware-adaptive logic is a simple startup routine that inspects the hardware and applies the best-fit configuration to the global `Settings.llm` object.

```mermaid
graph TD
    A[Application Startup] --> B[Detect Hardware (VRAM)];
    B --> C{VRAM >= 16GB?};
    C -->|Yes| D["Config Profile: High<br/>(65K Context)"];
    C -->|No| E{VRAM >= 8GB?};
    E -->|Yes| F["Config Profile: Medium<br/>(32K Context)"];
    E -->|No| G["Config Profile: Low<br/>(16K Context)"];
    D --> H[Configure Settings.llm];
    F --> H;
    G --> H;
    H --> I[Apply PyTorch Optimizations];
    I --> J[LLM Ready];
```

### Implementation Details

**In `hardware_setup.py`:**

```python
import torch
from llama_index.core import Settings
import logging

logger = logging.getLogger(__name__)

def get_vram_gb():
    """Detects available GPU VRAM in gigabytes."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / (1024**3)

def apply_adaptive_model_config():
    """
    Detects hardware and applies the optimal context length for the default model.
    This should be called *after* the base Settings.llm object is created.
    """
    if not hasattr(Settings, 'llm') or Settings.llm is None:
        logger.error("LLM is not configured. Cannot apply adaptive settings.")
        return

    vram_gb = get_vram_gb()
    context_length = 16384  # Default for CPU or low-VRAM GPUs

    if vram_gb >= 16:
        context_length = 65536
        logger.info(f"High VRAM ({vram_gb:.1f}GB) detected. Setting context to {context_length}.")
    elif vram_gb >= 8:
        context_length = 32768
        logger.info(f"Medium VRAM ({vram_gb:.1f}GB) detected. Setting context to {context_length}.")
    else:
        logger.info(f"Low VRAM ({vram_gb:.1f}GB) or CPU detected. Setting context to {context_length}.")

    # Update the context length on the globally configured LLM
    if hasattr(Settings.llm, 'n_ctx'): # For LlamaCPP
        Settings.llm.n_ctx = context_length
    if hasattr(Settings.llm, 'additional_kwargs'): # For Ollama
        Settings.llm.additional_kwargs["num_ctx"] = context_length
    if hasattr(Settings.llm, 'max_model_len'): # For vLLM
        Settings.llm.max_model_len = context_length
```

## Consequences

### Positive Outcomes

- **Improved Reasoning**: `Qwen3-4B-Thinking` provides benchmark-leading performance for agentic tool use, directly improving the core feature of the application.
- **Optimal User Experience**: The hardware-adaptive configuration ensures that users get the best possible performance from their specific hardware without needing to understand complex settings.
- **Efficient Resource Use**: By applying quantization and adjusting context length, the system makes efficient use of available VRAM.
- **Large Document Coverage**: The ability to use a 65K context window on capable hardware allows the agent to analyze the majority of documents in a single pass.

### Negative Consequences / Trade-offs

- **Model Dependency**: The system's reasoning quality is tightly coupled to the performance of this specific model. A future model may require retuning the agent's prompts.

## Changelog

- **5.0 (2025-01-16)**: Finalized as the definitive model strategy. Aligned all code with the `Settings` singleton and the final hardware-adaptive configuration logic.
- **4.0 (2025-01-13)**: Integrated `Settings.llm` configuration with GPU optimization and quantization.
- **3.1 (2025-01-13)**: Added cross-references to optimization ADRs.
- **3.0 (2025-01-13)**: Updated to support Qwen3-4B-Thinking as the unified model.
