# ADR 005: Text Splitting and Chunking

## Version/Date

v1.0 / July 22, 2025

## Status

Accepted

## Context

Large documents require splitting to fit LLM context sizes and enable efficient vector search. Late chunking can improve embedding accuracy.

## Decision

- Use **RecursiveCharacterTextSplitter** (LangChain) with chunk_size=1000, overlap=200 for general splitting.
- Implement **late chunking** in `utils.py:late_chunking()` using NLTK sentence tokenization and mean-pooled embeddings for precision.
- Apply late chunking optionally for large documents or accuracy-critical use cases.
- Store chunked embeddings in Document metadata for multi-vector support.

## Rationale

- RecursiveCharacterTextSplitter is robust and customizable.
- Late chunking with NLTK and mean pooling preserves semantic coherence.
- Optional late chunking balances performance and accuracy.

## Alternatives Considered

- SpaCy-based splitting: Slower, heavier dependency.
- Simple length-based splits: Less accurate for embeddings.

## Consequences

- Pros: Flexible, accurate for embeddings.
- Cons: Late chunking increases compute; mitigated by GPU and optional toggle.
