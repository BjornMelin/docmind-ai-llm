# ADR-016: Multimodal Embeddings

## Title

Multimodal Embeddings with Jina v4 and Unstructured Parsing

## Version/Date

1.0 / July 25, 2025

## Status

Accepted

## Context

Offline multimodal embeddings for PDFs (text/image/table). Jina v4 512D MRL efficient, Unstructured local parsing (hi_res for elements).

## Related Requirements

- Offline (local HuggingFace Jina, Unstructured parsing).
- Multimodal Phase 3.1 (integrate with hybrid).

## Alternatives

- CLIP: Lower accuracy (81% vs. Jina 84% CLIP benchmark).
- Custom parsing: Leaky/maintenance-heavy.

## Decision

Use HuggingFaceEmbedding("jinaai/jina-embeddings-v4", dim=512, int8 quant) for multimodal. Parse with UnstructuredReader (hi_res strategy).

## Related Decisions

- ADR-004 (Parsing with Unstructured).
- ADR-002 (Embeddings: Jina v4).

## Design

- **Parsing/Embedding**: In utils.py: from llama_index.readers.unstructured import UnstructuredReader; elements = UnstructuredReader().load_data(file_path, strategy="hi_res"); docs = [Document.from_element(e) for e in elements]; multimodal_index = MultiModalVectorStoreIndex.from_documents(docs, image_embed_model=embed_model).
- **Integration**: Use in HybridFusionRetriever/QueryPipeline (dim=512 for images). Toggle strategy via AppSettings.parse_strategy.
- **Implementation Notes**: Int8 quant for VRAM: quantization_config=BitsAndBytesConfig(load_in_8bit=True). Error handling: Fallback to text if image extraction fails.
- **Testing**: tests/test_utils.py: def test_multimodal_embed_parse(): elements = reader.load_data(pdf); assert any(e.type=="image" for e in elements); emb = embed_model.embed_image("img"); assert len(emb) == 512; def test_quant_vram(): if gpu: assert quant reduces memory.

## Consequences

- Offline multimodal hybrid (Jina accuracy, Unstructured parsing).
- Efficient (512D MRL, int8 quant).

- Deps: unstructured[all-docs]==0.15.13 (Docker: apt-get tesseract-ocr poppler-utils).
- Future: Toggle dim/strategy via AppSettings.
