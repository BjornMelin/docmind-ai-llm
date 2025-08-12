# ADR 010: LangChain Integration and Usage

## Version/Date

v1.3 / July 24, 2025 (Updated for LangGraph toggle implementation)

## Status

Superseded by ADR 015: Migration to LlamaIndex (July 2025)

## Context

DocMind AI requires an orchestration framework to manage LLM interactions, document processing, retrieval, and chaining operations for efficient document analysis and Retrieval-Augmented Generation (RAG). LangChain provides a modular, extensible set of tools for building LLM applications, making it suitable for integrating with local LLMs like Ollama and vector stores like Qdrant. For advanced multi-agent support, extend with LangGraph (v0.5.4) for stateful, parallel workflows. Compared to LlamaIndex's ReAct (simple single-agent), LangGraph offers better parallelism but more setup—make optional via toggle.

## Decision

- Integrate **LangChain v0.3.27** (core and community modules) as the primary orchestration framework.

- Key components used:
  - **Chains:** LLMChain for prompt-based analysis, RetrievalQA for chat-based RAG, load_summarize_chain for handling large documents via map-reduce.
  - **Retrievers:** ContextualCompressionRetriever for compressing and reranking retrieved documents in RAG pipelines.
  - **Text Splitters:** RecursiveCharacterTextSplitter for chunking documents into manageable sizes (chunk_size=1000, overlap=200) to fit LLM context windows.
  - **Embeddings:** HuggingFaceEmbeddings (for Jina v4 dense) and FastEmbedSparse (community integration) for vector representations in hybrid search.
  - **Vector Stores:** QdrantVectorStore (from langchain_community) for indexing and hybrid retrieval (dense + sparse modes).
  - **Output Parsers:** PydanticOutputParser for structuring LLM outputs into AnalysisOutput models.

- **LangGraph Extension:** Integrate LangGraph v0.5.4 for optional multi-agent support: build graphs with nodes (e.g., RetrievalAgent via LlamaIndex tools, AnalysisAgent) for parallel/stateful RAG. Toggle via config (default: ReAct for simplicity; multi_agent: True for advanced). UI checkbox below text input (like ChatGPT/Grok/Claude for custom options/tools) in app.py; conditional init in utils.py for agent (ReAct if false, LangGraph if true).

- Usage in codebase:
  - **Document Loading and Splitting:** In `utils.py:load_documents()`, documents are loaded and split using RecursiveCharacterTextSplitter before indexing.
  - **Vectorstore Creation:** In `utils.py:create_vectorstore()`, QdrantVectorStore.from_documents() indexes chunks with embeddings for hybrid search.
  - **Analysis Pipeline:** In `utils.py:analyze_documents()`, LLMChain processes prompts with Pydantic parsing; load_summarize_chain handles oversized texts.
  - **RAG/Chat:** In `utils.py:chat_with_context()`, RetrievalQA.from_chain_type() combines retrievers with LLMs; ContextualCompressionRetriever integrates custom rerankers. For multi-agent, use LangGraph graph with agent nodes (toggleable).

- Community Modules: langchain-community for Qdrant and FastEmbed integrations, ensuring compatibility with third-party tools.

## Rationale

- LangChain simplifies building complex workflows like RAG by providing standardized interfaces for chains, retrievers, and vector stores.

- Chains (e.g., LLMChain, RetrievalQA) enable modular prompt execution and retrieval-augmented responses, crucial for customizable analysis and chat.

- Text splitters and embeddings support scalable document processing, with hybrid search (dense/sparse) improving recall by 15-20% in RAG.

- Output parsers ensure structured results, aligning with Pydantic models for type safety.

- Community integrations (e.g., Qdrant) allow seamless hybrid search without custom code, adhering to DRY and library-first principles.

- **LangGraph Rationale:** Extends LangChain for multi-agent (parallel reasoning, state persistence), integrating tools for advanced RAG without replacement. Optional to maintain KISS (default ReAct: simpler/faster for basic; LangGraph better for complex but adds overhead—toggle avoids YAGNI violations). UI toggle below text input for easy access like major apps.

- Best Practices: Use chains for orchestration to avoid direct LLM calls; retrievers with compression reduce noise; splitters preserve context via overlap. For LangGraph, use graphs for multi-agent flows.

## Alternatives Considered

- Direct LLM calls without LangChain: More boilerplate, harder to maintain RAG pipelines.

- Haystack or LlamaIndex-only: LlamaIndex sufficient for simple agents (ReAct), but LangChain+Graph better for multi-agent; hybrid maximizes strengths (ReAct default, Graph optional).

- Custom orchestration: Violates YAGNI and DRY; LangChain/Graph handles edge cases like chunking/retrieval efficiently.

## Consequences

- Pros: Modular, extensible; supports advanced features like hybrid search, submodular reranking, and optional multi-agent parallelism.

- Cons: Dependency on LangChain/Graph updates; mitigated by pinning (v0.3.27/v0.5.4) and monitoring (e.g., Pydantic 2 support).

- As of July 2025, no major breaking changes; frameworks stable for use case.
