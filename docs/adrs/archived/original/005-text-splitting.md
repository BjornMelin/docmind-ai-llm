# ADR-005: Text Splitting

## Title

Semantic Chunking Strategy with Native IngestionPipeline Integration

## Version/Date

4.0 / August 14, 2025

## Status

Accepted

## Description

Utilizes the native LlamaIndex `SentenceSplitter` within the `IngestionPipeline` for semantic-aware chunking. The configuration, defaulting to 1024-token chunks with a 200-token overlap, is managed globally via the `Settings` singleton to optimize for the chosen embedding models.

## Context

Effective Retrieval-Augmented Generation (RAG) depends on splitting documents into meaningful chunks that fit within an embedding model's context window while preserving semantic coherence. Simple character or token counting can break sentences and destroy context. A semantic chunking strategy that respects sentence boundaries is required. This process must be integrated directly into the main document processing pipeline to benefit from caching and a unified workflow.

## Related Requirements

- **Native IngestionPipeline Integration**: Chunking must be a standard transformation step within the pipeline.
- **Configurable Chunking**: Chunk size and overlap must be configurable globally.
- **Semantic Awareness**: The splitting strategy must prioritize keeping whole sentences together.
- **Embedding Model Optimization**: Chunk sizes must be optimized for the dimensions of the project's embedding models (e.g., BGE-large).

## Alternatives

- **TokenTextSplitter**: A basic splitter that counts tokens. Rejected because it is not semantically aware and can easily split sentences mid-thought.
- **Custom Splitting Logic**: Would require writing and maintaining complex regex or NLP-based splitting rules. Rejected for violating the library-first principle.
- **Fixed-Size Splitting**: The simplest method, but the least effective as it has no regard for content structure. Rejected for producing low-quality chunks.

## Decision

Use the native LlamaIndex **`SentenceSplitter`** as the standard text splitting component. It will be integrated as a transformation within the main **`IngestionPipeline`**. All configuration parameters, such as `chunk_size` and `chunk_overlap`, will be managed globally and accessed via the **`llama_index.core.Settings`** singleton.

## Related Decisions

- `ADR-021` (LlamaIndex Native Architecture Consolidation): This decision is a core part of the native ingestion workflow.
- `ADR-020` (LlamaIndex Native Settings Migration): Provides the `Settings` singleton used for configuring the splitter.
- `ADR-004` (Document Loading): Text splitting is the transformation step that immediately follows document parsing in the `IngestionPipeline`.
- `ADR-008` (Session Persistence): The output of the chunking process is cached by the `IngestionCache` within the pipeline.
- `ADR-002` (Embedding Choices): The default chunk size of 1024 is chosen to align with the `BGE-large-en-v1.5` embedding model's optimal input size.

## Design

### Global Configuration via Settings

The chunking parameters are set once on the global `Settings` object.

```python
# In settings_setup.py
from llama_index.core import Settings

def configure_global_settings():
    """Sets up all global configuration for LlamaIndex."""
    # ... other settings ...
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200
    # ... other settings ...
```

### Integration into IngestionPipeline

The `SentenceSplitter` is instantiated using the global `Settings` and placed into the pipeline.

```python
# In pipeline_factory.py
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings

# The splitter automatically uses the globally configured chunk_size and chunk_overlap
splitter = SentenceSplitter.from_defaults()

# The splitter is a key transformation in the pipeline
ingestion_pipeline = IngestionPipeline(
    transformations=[
        splitter,
        # Other transformations like an embedding model would follow
    ],
    cache=IngestionCache() # The results of the splitting are cached
)
```

### Dynamic Configuration for Multimodal Content

While the default is 1024, the chunk size can be dynamically adjusted for different content types if needed, for instance, for smaller text chunks associated with images.

```python
# Example of dynamically creating a splitter for a specific use case
from llama_index.core.node_parser import SentenceSplitter

# This splitter is optimized for the CLIP model's context
multimodal_splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=100
)

# This could be used in a separate pipeline for image-specific text
multimodal_pipeline = IngestionPipeline(transformations=[multimodal_splitter])
```

## Consequences

### Positive Outcomes

- **Improved Semantic Quality**: `SentenceSplitter` respects natural language structure, leading to higher-quality chunks and better RAG performance.
- **Native Caching Benefits**: As part of the `IngestionPipeline`, all chunking results are automatically cached by `IngestionCache`, saving significant processing time.
- **Simplified Architecture**: Using a native component and the `Settings` singleton eliminates custom logic and keeps the codebase clean and maintainable.
- **Embedding Optimization**: The chunk sizes are aligned with the chosen embedding models, ensuring optimal performance.

### Ongoing Considerations

- **Language Specificity**: `SentenceSplitter` is optimized for English and other common languages. If support for less common languages is required, its behavior will need to be tested.
- **Configuration Tuning**: The optimal `chunk_size` and `chunk_overlap` may vary slightly between different embedding models. These values in the `Settings` should be reviewed if the embedding model is changed.

## Changelog

- **4.0 (August 14, 2025)**: Rewritten to align with final architecture. Replaced all `AppSettings` references with the native `Settings` singleton. Clarified integration within the `IngestionPipeline`.
- **3.1 (August 13, 2025)**: Added cross-references to PyTorch optimization (`ADR-023`).
- **3.0 (August 13, 2025)**: Native integration with `IngestionCache` and multi-modal chunk optimization. Aligned with CLIP and BGE embedding dimensions.
- **2.0 (July 25, 2025)**: Switched to `SentenceSplitter` in `IngestionPipeline`; Added `AppSettings` configs; Enhanced testing.
