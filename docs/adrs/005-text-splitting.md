# ADR-005: Text Splitting

## Title

Chunking and Text Splitting Strategy

## Version/Date

2.0 / July 25, 2025

## Status

Accepted

## Context

Chunking balances context/recall (semantic-aware, 1024 size/200 overlap optimal from research—fits embeddings dims).

## Related Requirements

- Configurable (AppSettings.chunk_size=1024, chunk_overlap=200).
- Post-parsing (after Unstructured).
- Semantic for RAG quality.

## Alternatives

- TokenTextSplitter: Basic, no semantics (lower recall).
- Custom: Maintenance-heavy.

## Decision

Use SentenceSplitter in IngestionPipeline (chunk_size=AppSettings.chunk_size, chunk_overlap=AppSettings.chunk_overlap). Semantic-aware for better splits.

## Related Decisions

- ADR-004 (Post-loading in pipeline).
- ADR-002 (Chunks fit dims, e.g., 1024 for dense).

## Design

- **Splitter**: In utils.py create_index: from llama_index.core.node_parser import SentenceSplitter; splitter = SentenceSplitter(chunk_size=AppSettings.chunk_size, chunk_overlap=AppSettings.chunk_overlap); pipeline = IngestionPipeline(transformations=[splitter, MetadataExtractor()]).
- **Integration**: Run pipeline on parsed elements from Unstructured → nodes to indexes. UI/AppSettings toggle for size/overlap.
- **Implementation Notes**: Handle large docs (e.g., if > max, recursive split). Error: if chunk_size < overlap, raise ValueError.
- **Testing**: In tests/test_utils.py: def test_chunking(): nodes = pipeline.run([long_doc]); assert len(nodes) > 1; assert len(nodes[0].text.split()) <= AppSettings.chunk_size; assert nodes[0].text[-AppSettings.chunk_overlap:] in nodes[1].text[:AppSettings.chunk_overlap]; @pytest.mark.parametrize("size, overlap", [(512, 100), (1024, 200)]) def test_config_chunk(size, overlap): AppSettings.chunk_size = size; AppSettings.chunk_overlap = overlap; nodes = pipeline.run([doc]); assert approx(len(nodes[0].text)) == size.

## Consequences

- Better recall (semantic chunks/overlap).
- Configurable (tune via AppSettings for data types, e.g., smaller for multimodal).

- Minor overhead (semantic slower than token ~10%).
- Deps: llama-index==0.12.52.

**Changelog:**  

- 2.0 (July 25, 2025): Switched to SentenceSplitter in IngestionPipeline; Added AppSettings configs/integration post-Unstructured; Enhanced testing with param for dev.
