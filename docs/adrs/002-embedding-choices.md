# ADR-002: Embedding Choices

## Title

Selection of Embedding Models for Semantic Search and Multimodal Processing

## Version/Date

3.1 / August 13, 2025

## Status

Accepted

## Description

Defines BGE-large-en-v1.5 (1024D) for dense text embeddings, SPLADE++ for sparse retrieval, and CLIP ViT-B/32 (512D) for multimodal processing with 60% VRAM reduction.

## Context

Embeddings drive retrieval accuracy/efficiency. Offline: Local HuggingFace/FastEmbed models. Optimal dims: 1024D for dense text (balance accuracy/speed), sparse no fixed dim (vocab-based), 512D for multimodal/images (VRAM savings). Research: BGE-Large 1024D MTEB leader for text; sparse SPLADE++ for term expansion; CLIP ViT-B/32 for multimodal with 60% VRAM reduction vs jina-v4.

## Related Requirements

- Offline/local (HuggingFace/FastEmbed).

- Hybrid: Dense (BGE 1024D) + sparse (SPLADE++).

- Multimodal: CLIP ViT-B/32 512D (text+image from Unstructured).

- Configurable: AppSettings.dense_embedding_dimension for dims (default 1024 text, 512 multimodal).

## Alternatives

- OpenAI Ada: API-dependent (rejected).

- Sentence-Transformers: 384D—lower accuracy than BGE/Jina, no multimodal.

- Jina v4: Higher VRAM usage (3.4GB vs 1.4GB CLIP), less native LlamaIndex integration.

## Decision

Dense: FastEmbedEmbedding("BAAI/bge-large-en-v1.5", dim=1024 for text). Sparse: SparseTextEmbedding("prithvida/Splade_PP_en_v1", no fixed dim). Multimodal: ClipEmbedding("ViT-B/32", dim=512, normalize=True) and native LlamaIndex integration. Use different dims: 1024D text/general, 512D images/multimodal (configurable).

## Related Decisions

- ADR-001 (Integrates with retrieval foundation).

- ADR-016 (Multimodal with CLIP ViT-B/32).

- ADR-020 (LlamaIndex Settings Migration - unified embedding configuration).

- ADR-022 (Tenacity Resilience Integration - robust embedding operations with retry patterns)

- ADR-003 (GPU Optimization - provides RTX 4090 optimization for embedding generation)

- ADR-023 (PyTorch Optimization Strategy - enables mixed precision and quantization for embedding models)

## Design

- **Init/Setup**: In src/utils.py, dense_model = FastEmbedEmbedding(AppSettings.dense_embedding_model, dim=AppSettings.dense_embedding_dimension or 1024); multimodal_model = ClipEmbedding(model_name="ViT-B/32", embed_batch_size=10, normalize=True) with 1.4GB VRAM usage.

- **Integration**: VectorStoreIndex(embed_model=dense_model); MultiModalVectorStoreIndex(image_embed_model=multimodal_model). HybridFusionRetriever combines dense/sparse. Use dim=512 for multimodal in AppSettings toggle (e.g., if multimodal: set_dim(512)).

- **Implementation Notes**: Embed in IngestionPipeline post-Unstructured parsing. For images: multimodal_model.embed_image(extracted_image_paths). CLIP provides native LlamaIndex integration and 60% VRAM savings.

- **Testing**: In tests/test_embeddings.py: def test_dims_types(): text_emb = dense_model.embed(["text"]); assert len(text_emb[0]) == 1024; img_emb = multimodal_model.embed_image(["img"]); assert len(img_emb[0]) == 512; @pytest.mark.parametrize("dim", [512, 1024]) def test_config_dim(dim): AppSettings.dense_embedding_dimension = dim; emb = embed_model.embed(["test"]); assert len(emb[0]) == dim.

## Consequences

- High accuracy/efficiency (BGE top MTEB, CLIP multimodal with VRAM optimization, SPLADE expansion).

- Offline (local downloads on first use).

- Flexible (dims/quant via AppSettings, different for text/multimodal).

- VRAM: CLIP provides 60% reduction (1.4GB vs 3.4GB jina-v4) with native optimization.

- Deps: transformers for quant/HuggingFace (pinned).

**Changelog:**  

- 3.1 (August 13, 2025): Added cross-references to GPU optimization (ADR-003) and PyTorch optimization (ADR-023) for integrated embedding performance.

- 3.0 (August 13, 2025): Replaced Jina v4 with CLIP ViT-B/32 for 60% VRAM reduction (1.4GB vs 3.4GB) and native LlamaIndex integration; Simplified quantization approach; Aligned with simplified architecture decisions.

- 2.0 (July 25, 2025): Added Jina v4 multimodal with 512D MRL/different dims by type; int8 quantization; Enhanced integration/testing notes; Updated alternatives/research for streamlined dev.
