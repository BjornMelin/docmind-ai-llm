# ADR-002: Embedding Choices

## Title

Selection of Embedding Models for Semantic Search and Multimodal Processing

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Embeddings drive retrieval accuracy/efficiency. Offline: Local HuggingFace/FastEmbed models. Optimal dims: 1024D for dense text (balance accuracy/speed), sparse no fixed dim (vocab-based), 512D MRL for multimodal/images (VRAM savings). Research: BGE-Large 1024D MTEB leader for text; sparse SPLADE++ for term expansion; Jina v4 512D for multimodal.

## Related Requirements

- Offline/local (HuggingFace/FastEmbed).
- Hybrid: Dense (BGE 1024D) + sparse (SPLADE++).
- Multimodal: Jina v4 512D (text+image from Unstructured).
- Configurable: AppSettings.dense_embedding_dimension for dims (default 1024 text, 512 multimodal).

## Alternatives

- OpenAI Ada: API-dependent (rejected).
- Sentence-Transformers: 384D—lower accuracy than BGE/Jina, no multimodal.
- CLIP: Basic multimodal but inferior to Jina v4 (81% vs. 84% CLIP benchmark).

## Decision

Dense: FastEmbedEmbedding("BAAI/bge-large-en-v1.5", dim=1024 for text). Sparse: SparseTextEmbedding("prithvida/Splade_PP_en_v1", no fixed dim). Multimodal: HuggingFaceEmbedding("jinaai/jina-embeddings-v4", dim=512 MRL, int8 quant, task="retrieval.passage"). Use different dims: 1024D text/general, 512D images/multimodal (configurable).

## Related Decisions

- ADR-001 (Integrates with retrieval foundation).
- ADR-016 (Multimodal with Jina v4).

## Design

- **Init/Quant**: In utils.py, dense_model = FastEmbedEmbedding(AppSettings.dense_embedding_model, dim=AppSettings.dense_embedding_dimension or 1024); multimodal_model = HuggingFaceEmbedding("jinaai/jina-embeddings-v4", dim=512, quantization_config=BitsAndBytesConfig(load_in_8bit=True) if AppSettings.enable_quantization else None, device="cuda" if AppSettings.gpu_acceleration else "cpu").
- **Integration**: VectorStoreIndex(embed_model=dense_model); MultiModalVectorStoreIndex(image_embed_model=multimodal_model). HybridFusionRetriever combines dense/sparse. Use dim=512 for multimodal in AppSettings toggle (e.g., if multimodal: set_dim(512)).
- **Implementation Notes**: Embed in IngestionPipeline post-Unstructured parsing. For images: multimodal_model.embed_image(extracted_image_paths).
- **Testing**: In tests/test_embeddings.py: def test_dims_types(): text_emb = dense_model.embed(["text"]); assert len(text_emb[0]) == 1024; img_emb = multimodal_model.embed_image(["img"]); assert len(img_emb[0]) == 512; @pytest.mark.parametrize("dim", [512, 1024]) def test_config_dim(dim): AppSettings.dense_embedding_dimension = dim; emb = embed_model.embed(["test"]); assert len(emb[0]) == dim.

## Consequences

- High accuracy/efficiency (BGE top MTEB, Jina multimodal, SPLADE expansion).
- Offline (local downloads on first use).
- Flexible (dims/quant via AppSettings, different for text/multimodal).

- VRAM: int8 quant mitigates (halves usage).
- Deps: transformers for quant/HuggingFace (pinned).

**Changelog:**  

- 2.0 (July 25, 2025): Added Jina v4 multimodal with 512D MRL/different dims by type; int8 quantization; Enhanced integration/testing notes; Updated alternatives/research for streamlined dev.
