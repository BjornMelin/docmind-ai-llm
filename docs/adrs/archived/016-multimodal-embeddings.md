# ADR-016: Multimodal Embeddings

## Title

Multimodal Embeddings with CLIP ViT-B/32 and Unstructured Parsing

## Version/Date

3.0 / 2025-01-16

## Status

Accepted

## Description

Implements multimodal search capabilities by using the `UnstructuredReader` to extract both text and images from documents, and the `openai/clip-vit-base-patch32` (CLIP) model to generate embeddings for the extracted images.

## Context

A significant portion of information in documents like PDFs is contained in images, diagrams, and charts. A text-only RAG system is blind to this information. To provide a complete analysis capability, the system must be able to parse images from documents and embed them in the same vector space as the text, enabling search based on image content. The solution must be efficient, offline, and tightly integrated with the native LlamaIndex data structures.

## Related Requirements

### Functional Requirements

- **FR-1:** The system must extract image objects from PDF and DOCX files.
- **FR-2:** The system must generate vector embeddings for extracted images.

### Non-Functional Requirements

- **NFR-1:** **(Security)** The entire multimodal parsing and embedding process must be local and offline.
- **NFR-2:** **(Performance)** The chosen image embedding model must have a low VRAM footprint (under 2GB).

### Integration Requirements

- **IR-1:** The image embedding model must be a native or well-supported LlamaIndex component.
- **IR-2:** The process must integrate with the main `IngestionPipeline` and `VectorStoreIndex`.

## Alternatives

### 1. Jina v4 Embeddings

- **Description**: A powerful multimodal embedding model.
- **Issues**: Has a significantly higher VRAM footprint (~3.4GB) compared to CLIP (~1.4GB), making it less suitable for the target consumer hardware.
- **Status**: Rejected.

### 2. Custom Parsing Logic

- **Description**: Use a library like PyMuPDF to manually extract images and then pass them to an embedding model.
- **Issues**: Requires significant custom code to manage file types, image data, and temporary files. The `UnstructuredReader` provides this functionality out of the box.
- **Status**: Rejected.

## Decision

We will adopt a two-part strategy for multimodal ingestion:

1. **Parsing**: Use the **`UnstructuredReader`** with the `"hi_res"` strategy, as defined in `ADR-004`, to parse documents into a list of text and image elements.
2. **Embedding**: Use the native LlamaIndex **`ClipEmbedding`** model (`openai/clip-vit-base-patch32`) to generate embeddings for the image elements. The text and image embeddings will be stored together in a **`MultiModalVectorStoreIndex`**.

## Related Decisions

- **ADR-002** (Embedding Choices): This ADR implements the decision to use the CLIP model for image embeddings.
- **ADR-004** (Document Loading): This ADR relies on the `UnstructuredReader` defined in `ADR-004` as its primary source of image data.
- **ADR-020** (LlamaIndex Native Settings Migration): The CLIP model is configured and accessed via the global `Settings` singleton.
- **ADR-003** (GPU Optimization): The CLIP model will automatically use the GPU if available.

## Design

### Architecture Overview

The multimodal ingestion process is a clear, linear workflow that separates text and image data after parsing and then indexes them into a specialized multimodal vector store.

```mermaid
graph TD
    A[Input Document (PDF)] --> B["UnstructuredReader<br/>(hi_res strategy)"];
    B --> C{"Element Type?"};
    C -->|Text/Table| D["Text Documents"];
    C -->|Image| E["Image Documents"];
    D --> F["MultiModalVectorStoreIndex"];
    E -- Embedded by CLIP --> F;
```

### Implementation Details

**In `multimodal_ingestion.py`:**

```python
# This code shows the end-to-end multimodal ingestion process
from llama_index.core import SimpleDirectoryReader, MultiModalVectorStoreIndex
from llama_index.core.schema import ImageDocument
from llama_index.core import Settings
import os

# Assume Settings.embed_model (BGE) and Settings.image_embed_model (CLIP)
# have already been configured per ADR-002.

def ingest_multimodal_documents(directory_path: str, index_path: str):
    """
    Parses all documents in a directory, extracts text and images,
    and builds a multimodal vector index.
    """
    # ADR-004: Use UnstructuredReader via SimpleDirectoryReader to parse documents
    # into text and image elements.
    # The file_extractor can be customized for more file types.
    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        file_extractor={".pdf": "UnstructuredReader"}
    )
    
    documents = reader.load_data()

    # The MultiModalVectorStoreIndex automatically handles both text and image documents.
    # It will use Settings.embed_model for text and Settings.image_embed_model for images.
    multimodal_index = MultiModalVectorStoreIndex.from_documents(
        documents,
        show_progress=True
    )

    # Persist the index to disk
    multimodal_index.storage_context.persist(persist_dir=index_path)
```

## Consequences

### Positive Outcomes

- **Full Content Analysis**: The system can now search based on the content of images, providing a much more complete analysis of documents.
- **High Efficiency**: The chosen CLIP model is highly optimized and has a low VRAM footprint (~1.4GB), making it ideal for consumer hardware.
- **Simplified Architecture**: The entire process is handled by native LlamaIndex components, requiring minimal custom code and ensuring high maintainability.

### Negative Consequences / Trade-offs

- **Increased Indexing Time**: Parsing images and generating image embeddings is more computationally intensive than text-only processing, leading to longer initial indexing times. This is a necessary trade-off for the added functionality.

### Dependencies

- **System**: Requires system-level dependencies for `unstructured`, as detailed in `ADR-004` (e.g., `poppler-utils`, `tesseract-ocr`).
- **Python**: `llama-index-readers-unstructured`, `llama-index-embeddings-clip`, `unstructured[pdf,image]`

## Changelog

- **3.0 (2025-01-16)**: Finalized as the definitive multimodal strategy. Aligned all code with the `Settings` singleton and the native `MultiModalVectorStoreIndex` workflow.
- **2.0 (2025-01-13)**: Replaced Jina v4 with CLIP ViT-B/32 for VRAM reduction and native integration.
