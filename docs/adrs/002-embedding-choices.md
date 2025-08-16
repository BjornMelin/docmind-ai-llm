# ADR-002: Embedding Choices

## Title

Selection of Embedding Models for Hybrid Search and Multimodal Processing

## Version/Date

4.0 / 2025-01-16

## Status

Accepted

## Description

Standardizes the embedding models for the system: `BAAI/bge-large-en-v1.5` for dense semantic search, `prithvida/Splade_PP_en_v1` for sparse keyword search, and `openai/clip-vit-base-patch32` for image embeddings. These models are all executed locally and configured via the LlamaIndex `Settings` singleton.

## Context

The quality of the RAG system is fundamentally dependent on the quality of its embeddings. The architecture requires a hybrid approach, combining dense and sparse vectors for text, and a separate model for handling image data. All models must be available for local, offline execution to meet the project's privacy requirements. The selection was based on a balance of performance on standard benchmarks (e.g., MTEB), resource efficiency (VRAM usage), and ease of integration with the LlamaIndex framework.

## Related Requirements

### Functional Requirements

- **FR-1:** The system must generate dense vector embeddings to capture semantic meaning.
- **FR-2:** The system must generate sparse vector embeddings to improve keyword matching.
- **FR-3:** The system must generate embeddings for images to enable multimodal search.

### Non-Functional Requirements

- **NFR-1:** **(Security)** All embedding models must run locally with no external API calls.
- **NFR-2:** **(Performance)** The models must be efficient in terms of VRAM usage and inference speed on consumer hardware.

### Integration Requirements

- **IR-1:** All models must be configurable via the global LlamaIndex `Settings` singleton.
- **IR-2:** The initial download of models from online hubs (e.g., HuggingFace) must be resilient to network failures.

## Alternatives

### 1. OpenAI Ada v2

- **Description**: Use OpenAI's proprietary, high-performance embedding models.
- **Issues**: Requires an internet connection and API key, violating the core offline-first requirement.
- **Status**: Rejected.

### 2. Jina v4 Embeddings

- **Description**: A previously considered multimodal embedding model.
- **Issues**: Has a significantly higher VRAM footprint (~3.4GB) compared to the chosen CLIP model (~1.4GB) and offers less straightforward native integration with LlamaIndex.
- **Status**: Rejected.

### 3. Single Dense-Only Model

- **Description**: Rely exclusively on a dense embedding model like BGE-large.
- **Issues**: Lacks the keyword precision of a sparse model, leading to lower recall on queries with specific terms or acronyms.
- **Status**: Rejected.

## Decision

We will adopt a three-model strategy for embeddings, all configured via the `Settings` singleton:

1. **Dense Text Embeddings**: **`BAAI/bge-large-en-v1.5`** (1024 dimensions) will be the default model for semantic text representation, implemented via `HuggingFaceEmbedding`.
2. **Sparse Text Embeddings**: **`prithvida/Splade_PP_en_v1`** will be used for keyword-based retrieval, implemented via a sparse embedding integration.
3. **Image Embeddings**: **`openai/clip-vit-base-patch32`** (ViT-B/32, 512 dimensions) will be used for all image data, implemented via the native `ClipEmbedding` model.

## Related Decisions

- **ADR-013** (RRF Hybrid Search): This decision provides the dense and sparse embeddings that are fused by the hybrid retriever.
- **ADR-016** (Multimodal Embeddings): This decision is tightly coupled, with `ADR-016` detailing the implementation of the CLIP model chosen here.
- **ADR-020** (LlamaIndex Native Settings Migration): Provides the `Settings` singleton used for configuring these models.
- **ADR-003** (GPU Optimization): The embedding models leverage `device_map="auto"` for GPU acceleration.
- **ADR-023** (PyTorch Optimization Strategy): The models benefit from mixed precision for improved performance.
- **ADR-022** (Tenacity Resilience Integration): The resilience pattern is applied to the initial, one-time download of the models.

## Design

### Global Configuration

The embedding models are configured on the global `Settings` object during application startup.

**In `application_setup.py`:**

```python
# This code shows how the embedding models are set globally
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

logger = logging.getLogger(__name__)

# ADR-022: Apply Tenacity to make the one-time model download resilient
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=30)
)
def configure_embedding_models():
    """
    Initializes and configures all embedding models, setting them on the
    global Settings object. The download is wrapped in a retry mechanism.
    """
    try:
        logger.info("Initializing embedding models...")
        # Dense model for semantic search
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-large-en-v1.5",
            device_map="auto" # ADR-003: Automatic GPU offload
        )
        
        # Sparse model for keyword search (details in retriever setup)
        Settings.sparse_embed_model = "prithvida/Splade_PP_en_v1"

        # Multimodal model for images (details in ADR-016)
        Settings.image_embed_model = ClipEmbedding(
            model_name="openai/clip-vit-base-patch32"
        )
        logger.info("Embedding models initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to download or initialize embedding models after multiple retries: {e}")
        raise
```

## Consequences

### Positive Outcomes

- **High-Quality Retrieval**: The combination of a top-performing dense model (BGE) and a strong sparse model (SPLADE++) ensures high relevance for a wide range of queries.
- **Efficient Multimodality**: The chosen CLIP model provides robust image embedding capabilities with a 60% lower VRAM footprint compared to alternatives.
- **Offline and Secure**: The entire embedding process is local, ensuring data privacy.
- **Resilient Setup**: Wrapping the model download in a retry mechanism prevents application startup failure due to transient network issues.

### Negative Consequences / Trade-offs

- **Initial Download Size**: The models must be downloaded on first use, which can be several gigabytes. This is a one-time cost.
- **Resource Usage**: Running three models (even if the sparse model is lightweight) consumes more resources than a single-model approach, but is necessary for high-quality hybrid search.

### Dependencies

- **Python**: `llama-index-embeddings-huggingface`, `llama-index-embeddings-clip`, `transformers`, `torch`, `tenacity`

## Changelog

- **4.0 (2025-01-16)**: Finalized model choices, removing the superseded Jina model. Aligned all code with the `Settings` singleton and incorporated the `Tenacity` resilience pattern for model downloads.
- **3.1 (2025-01-13)**: Added cross-references to GPU and PyTorch optimization ADRs.
- **3.0 (2025-01-13)**: Replaced Jina v4 with CLIP ViT-B/32.
- **2.0 (2025-07-25)**: Added Jina v4 multimodal with quantization.
