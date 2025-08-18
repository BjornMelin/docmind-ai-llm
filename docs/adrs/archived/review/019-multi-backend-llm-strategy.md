# ADR-019: Multi-Backend LLM Architecture Strategy

## Title

Multi-Backend Local LLM Support with Native LlamaIndex Settings

## Version/Date

4.0 / 2025-01-16

## Status

Accepted

## Description

Enables unified, multi-backend support for local Large Language Models (LLMs) through the native LlamaIndex `Settings.llm` singleton. This approach replaces a complex factory pattern, reducing configuration code by over 98% and allowing users to easily switch between Ollama, LlamaCPP, and vLLM backends.

## Context

The local LLM ecosystem is diverse, and users have different preferences and hardware setups. Forcing a single backend (like Ollama) creates a poor user experience and limits the application's reach. The architecture must support the most popular local LLM backends (Ollama for ease of use, LlamaCPP for CPU/GGUF efficiency, and vLLM for high-throughput GPU serving). An initial proposal involved a complex, custom-built factory pattern to manage these backends, but this was identified as a violation of the library-first and KISS principles. The native LlamaIndex `Settings` singleton provides a much simpler and more elegant solution.

## Related Requirements

### Functional Requirements

- **FR-1:** The system must support user choice between Ollama, LlamaCPP, and vLLM backends.

### Non-Functional Requirements

- **NFR-1:** **(Maintainability)** The backend switching mechanism must be simple, with minimal custom code.
- **NFR-2:** **(Performance)** The architecture must achieve high performance (~1000 tokens/sec on target hardware) consistently across all supported backends.

### Integration Requirements

- **IR-1:** The LLM configuration must be managed by the central LlamaIndex `Settings.llm` object.

## Alternatives

### 1. Single Backend Only (Ollama)

- **Description**: Hardcode the application to only support the Ollama backend.
- **Issues**: Fails to accommodate the diverse needs and preferences of the user base, limiting adoption.
- **Status**: Rejected.

### 2. Custom Factory Pattern

- **Description**: A complex, 150+ line class-based factory for instantiating and configuring different LLM backends.
- **Issues**: Grossly over-engineered. It introduced a massive amount of custom code to solve a problem the framework already solved natively.
- **Status**: Rejected.

## Decision

We will implement multi-backend LLM support by leveraging the native LlamaIndex **`Settings.llm` singleton**. A simple dictionary will hold the configuration objects for each supported backend (Ollama, LlamaCPP, vLLM). The application will select the appropriate configuration from this dictionary based on a user setting and assign it to `Settings.llm` at startup. This approach eliminates all custom factory code and provides a clean, maintainable, and highly effective solution.

## Related Decisions

- **ADR-020** (LlamaIndex Native Settings Migration): This decision is a direct application of the principles established in `ADR-020`.
- **ADR-017** (Default Model Strategy): This ADR defines the default model (Qwen3-4B-Thinking) that will be configured for each backend.
- **ADR-003** (GPU Optimization): Each backend configuration will include the necessary parameters for GPU acceleration (e.g., `device_map="auto"`).
- **ADR-023** (PyTorch Optimization Strategy): Each backend configuration will include parameters for PyTorch optimizations like mixed precision.

## Design

### Architecture Overview

The design is radically simple. A central setup function reads a configuration value and assigns the corresponding pre-configured LLM object to the global `Settings.llm`. All other components in the system, such as the `ReActAgent`, then automatically use this globally configured LLM without needing to know which specific backend is active.

```mermaid
graph TD
    A[User Config (e.g., .env)] --> B{LLM Backend?};
    B -->|ollama| C[Ollama LLM Object];
    B -->|llamacpp| D[LlamaCPP LLM Object];
    B -->|vllm| E[vLLM LLM Object];
    C --> F["Settings.llm = ..."];
    D --> F;
    E --> F;
    F --> G[All Other Components<br/>(Agent, Pipeline, etc.)];
```

### Implementation Details

#### **The 98% Code Reduction: Before vs. After**

**In `llm_factory.py` (BEFORE - Now Deleted):**

```python
# BEFORE: 150+ lines of a complex, unnecessary factory pattern
class LLMBackendFactory:
    def __init__(self):
        self.backends = {}
        self.configuration_managers = {}
        # ... extensive, hard-to-maintain factory implementation ...
        # ... with methods for registration, creation, configuration ...
```

**In `application_setup.py` (AFTER):**

```python
# AFTER: A simple, clear dictionary and a single line of code.
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.llms.vllm import vLLM
import os

def configure_llm_backend():
    """
    Selects and configures the LLM backend based on an environment variable
    and assigns it to the global Settings.llm object.
    """
    backend_choice = os.getenv("LLM_BACKEND", "ollama").lower()
    
    # Configurations include GPU and PyTorch optimizations per ADR-003 and ADR-023
    backend_configs = {
        "ollama": Ollama(model="qwen3:4b-thinking", request_timeout=120.0),
        "llamacpp": LlamaCPP(
            model_path="./models/qwen3-4b-thinking.Q4_K_M.gguf",
            n_gpu_layers=-1,
            device_map="auto",
            n_ctx=65536
        ),
        "vllm": vLLM(
            model="Qwen/Qwen3-4B-Thinking-2507",
            gpu_memory_utilization=0.6,
            device_map="auto",
            torch_dtype="float16"
        )
    }

    # Single line to set the global LLM
    Settings.llm = backend_configs.get(backend_choice, backend_configs["ollama"])
```

## Consequences

### Positive Outcomes

- **Code Simplification**: Eliminated over 150 lines of custom factory code, a 98% reduction, improving maintainability.
- **Native Ecosystem Integration**: Fully embraces the native LlamaIndex `Settings` singleton, a core principle of the final architecture.
- **Flexibility & Simplicity**: Provides full multi-backend flexibility to the user with the simplest possible implementation.
- **High Performance**: The architecture supports high-performance backends like vLLM and GPU-accelerated LlamaCPP, enabling the ~1000 tokens/sec target.

### Negative Consequences / Trade-offs

- **Configuration at Startup**: The backend is chosen at application startup. Changing it at runtime would require re-initializing all dependent components. This is an acceptable limitation for a local, single-user application.

### Dependencies

- **Python**: `llama-index-llms-ollama`, `llama-index-llms-llama-cpp`, `llama-index-llms-vllm`

## Changelog

- **4.0 (2025-01-16)**: Finalized decision to use the native `Settings` singleton, completely removing the custom factory pattern. Updated design to show the simplified dictionary-based approach.
- **3.1 (2025-01-13)**: Added cross-references to GPU optimization ADR.
- **3.0 (2025-01-13)**: Updated to support Qwen3-4B-Thinking as the unified model.
