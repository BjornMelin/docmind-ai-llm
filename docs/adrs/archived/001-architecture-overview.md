# ADR 001: Overall Architecture Overview

## Version/Date

v1.0 / July 22, 2025

## Status

Accepted

## Context

DocMind AI needs a scalable, privacy-focused architecture for local document analysis. Key considerations: Local execution, performance on varied hardware, extensibility.

## Decision

- **Frontend:** Streamlit for simple, reactive UI.
- **LLM Backend:** Ollama primary; LlamaCpp/LM Studio alternatives for flexibility.
- **Orchestration:** LangChain for chains/retrievers.
- **Vector Store:** Qdrant for hybrid search (dense + sparse).
- **Embeddings:** Mixed: Jina v4 (dense, multimodal) + FastEmbed SPLADE++ (sparse).
- **Reranking:** Jina v2 with submodular optimization.
- **GPU Support:** torch-based detection/offload.

## Rationale

- Local focus ensures privacy.
- Hybrid search improves recall (15-20%) over dense-only.
- GPU optimization boosts TPS (2-3x on RTX 4090).
- LangChain simplifies RAG/chains.

## Alternatives Considered

- Pinecone/Weaviate for vector store: Qdrant chosen for local mode and hybrid support.
- OpenAI embeddings: Rejected for privacy; local Jina/FastEmbed preferred.

## Consequences

- Pros: High performance, extensible.
- Cons: Dependency on Ollama setup; GPU optional but recommended.
