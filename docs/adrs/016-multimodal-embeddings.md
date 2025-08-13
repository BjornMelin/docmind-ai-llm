# ADR-016: Multimodal Embeddings

## Title

Multimodal Embeddings with CLIP ViT-B/32 and Unstructured Parsing

## Version/Date

2.0 / August 13, 2025

## Status

Accepted

## Description

Integrates CLIP ViT-B/32 for 512D multimodal embeddings with 60% VRAM reduction and UnstructuredReader for comprehensive text, image, and table extraction from documents.

## Context

Offline multimodal embeddings for PDFs (text/image/table). CLIP ViT-B/32 512D with 60% VRAM reduction vs jina-v4, Unstructured local parsing (hi_res for elements).

## Related Requirements

- Offline (local CLIP ViT-B/32, Unstructured parsing with native LlamaIndex integration).

- Multimodal Phase 3.1 (integrate with hybrid).

## Alternatives

- Jina v4: Higher VRAM usage (3.4GB vs 1.4GB CLIP), less native integration.

- Custom parsing: Leaky/maintenance-heavy.

## Decision

Use ClipEmbedding("ViT-B/32", embed_batch_size=10, normalize=True) for multimodal with 60% VRAM reduction. Parse with UnstructuredReader (hi_res strategy).

## Related Decisions

- ADR-004 (Parsing with Unstructured).

- ADR-002 (Embeddings: CLIP ViT-B/32).

- ADR-020 (LlamaIndex Settings Migration - unified embedding configuration).

- ADR-022 (Tenacity Resilience Integration - robust multimodal processing with retry patterns).

## Design

- **Parsing/Embedding**: In utils.py: from llama_index.readers.unstructured import UnstructuredReader; from llama_index.embeddings.clip import ClipEmbedding; embed_model = ClipEmbedding(model_name="ViT-B/32", embed_batch_size=10, normalize=True); elements = UnstructuredReader().load_data(file_path, strategy="hi_res"); docs = [Document.from_element(e) for e in elements]; multimodal_index = MultiModalVectorStoreIndex.from_documents(docs, image_embed_model=embed_model).

- **Integration**: Use in HybridFusionRetriever/QueryPipeline (dim=512 for images). CLIP provides native LlamaIndex integration with 60% VRAM savings. Toggle strategy via AppSettings.parse_strategy.

- **Implementation Notes**: CLIP ViT-B/32 uses only 1.4GB VRAM natively (vs 3.4GB jina-v4). Error handling: Fallback to text if image extraction fails. Native optimization eliminates need for custom quantization.

- **Testing**: tests/test_utils.py: def test_multimodal_embed_parse(): elements = reader.load_data(pdf); assert any(e.type=="image" for e in elements); emb = embed_model.embed_image("img"); assert len(emb) == 512; def test_clip_vram(): if gpu: assert CLIP uses < 1.5GB VRAM; assert performance > jina baseline.

## Consequences

- Offline multimodal hybrid (CLIP native integration, Unstructured parsing).

- Efficient (512D, 60% VRAM reduction: 1.4GB vs 3.4GB).

- Deps: unstructured[all-docs]==0.15.13 (Docker: apt-get tesseract-ocr poppler-utils), llama-index-embeddings-clip.

- Future: Toggle dim/strategy via AppSettings.

**Changelog:**

- 2.0 (August 13, 2025): Replaced Jina v4 with CLIP ViT-B/32 for 60% VRAM reduction (1.4GB vs 3.4GB) and native LlamaIndex integration. Simplified quantization approach and enhanced performance optimization.
