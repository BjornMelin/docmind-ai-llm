# ADR 001: Overall Architecture Overview

## Version/Date

v1.2 / July 24, 2025 (Updated for multi-agent toggle)

## Status

Accepted

## Context

DocMind AI needs a scalable, privacy-focused architecture for local document analysis. Key considerations: Local execution, performance on varied hardware, extensibility. For advanced workflows, optional multi-agent support via LangGraph. Toggle (config in Settings, UI checkbox below text input like ChatGPT/Grok/Claude for custom options/tools) to enable/disable; default ReAct single agent for v1 simplicity.

## Decision

- **Frontend:** Streamlit for simple, reactive UI.
- **LLM Backend:** Ollama primary; LlamaCpp/LM Studio alternatives for flexibility.
- **Orchestration:** LangChain for chains/retrievers; optional LangGraph for multi-agent workflows (toggled via config/UI).
- **Vector Store:** Qdrant for hybrid search (dense + sparse).
- **Embeddings:** Mixed: Jina v4 (dense, multimodal) + FastEmbed SPLADE++ (sparse).
- **Reranking:** Jina v2 with submodular optimization.
- **GPU Support:** torch-based detection/offload.

## Rationale

- Local focus ensures privacy.
- Hybrid search improves recall (15-20%) over dense-only.
- GPU optimization boosts TPS (2-3x on RTX 4090).
- LangChain simplifies RAG/chains; LangGraph adds optional multi-agent parallelism without core changes. Toggle maintains KISS default.

## Alternatives Considered

- Pinecone/Weaviate for vector store: Qdrant chosen for local mode and hybrid support.
- OpenAI embeddings: Rejected for privacy; local Jina/FastEmbed preferred.

## Consequences

- Pros: High performance, extensible; multi-agent enables complex workflows.
- Cons: Dependency on Ollama setup; GPU optional but recommended; multi-agent adds optional complexity (mitigated by toggle/config).
