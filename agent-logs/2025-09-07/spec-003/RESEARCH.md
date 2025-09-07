# SPEC-003 Research Notes (Library-first)

Date: 2025-09-07

## BGE-M3 unified embeddings
- Source: BAAI/bge-m3 model card and FlagEmbedding docs
- API: `BGEM3FlagModel(model_name_or_path="BAAI/bge-m3", use_fp16=True, device=...)`
- Outputs: `encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)` -> keys: `dense_vecs`, `lexical_weights`, `colbert_vecs` (optional)
- Normalization: L2 recommended for dense outputs before similarity.

## LlamaIndex BGEM3Index/BGEM3Retriever
- Provides tri-mode retrieval; accepts `weights_for_different_modes=[dense, sparse, colbert]`.
- We route retrieval via these factories to avoid custom glue.

## OpenCLIP
- Load: `open_clip.create_model_and_transforms('ViT-L-14'|'ViT-H-14', pretrained='laion2b_s34b_b79k')`
- Encode: `model.encode_image(tensor)`; normalize feature vectors.
- Dimensions: ViT-L/14 frequently 768; ViT-H/14 frequently 1024. Derived at runtime.

## SigLIP (Transformers)
- Models: `google/siglip-base-patch16-224`
- Processor: `SiglipProcessor.from_pretrained(...)`
- Encode: `model.get_image_features(pixel_values=...)`; normalize.

## Notes on Visualized-BGE
- Optional; disabled by default due to heavier GPU requirements. Can be wired behind a capability flag if needed later.

## Decisions
- Prefer LI-managed retriever (BGEM3Index/BGEM3Retriever) over custom embedders.
- In LI contexts, prefer LI ClipEmbedding for image vectors; use SigLIP only outside LI.
- Derive image dims from model outputs; avoid hard-coded dims.

