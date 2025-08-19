# Technical Glossary

## Overview

This glossary defines technical terms, acronyms, and concepts used throughout the DocMind AI specifications. Terms are organized alphabetically within categories for easy reference.

**Version**: 1.0.0  
**Updated**: 2025-08-19

## Table of Contents

- [Technical Glossary](#technical-glossary)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [AI/ML Terms](#aiml-terms)
  - [Architecture Terms](#architecture-terms)
  - [Document Processing](#document-processing)
  - [Infrastructure](#infrastructure)
  - [Libraries \& Frameworks](#libraries--frameworks)
  - [Models \& Embeddings](#models--embeddings)
  - [Performance Metrics](#performance-metrics)
  - [Retrieval \& Search](#retrieval--search)
  - [Acronyms Quick Reference](#acronyms-quick-reference)
  - [Usage Notes](#usage-notes)

---

## AI/ML Terms

**Agent**: An autonomous component that performs specific tasks within the multi-agent system. DocMind uses 5 specialized agents coordinated by a supervisor.

**Context Window**: The maximum number of tokens an LLM can process in a single interaction. DocMind uses 32K native context, extensible to 128K with YaRN.

**DSPy (Declarative Self-improving Python)**: A framework for automatic prompt optimization that improves retrieval quality by 20-30% through query rewriting and expansion.

**Few-shot Learning**: Providing a small number of examples to guide LLM behavior without fine-tuning.

**Function Calling**: The ability of an LLM to invoke predefined functions/tools, essential for agent operations.

**Hallucination**: When an LLM generates information not supported by its training data or provided context. The validation agent detects and prevents hallucinations.

**Inference**: The process of generating predictions or responses from a trained model. Target: ~1000 tokens/second.

**Prompt Engineering**: The practice of designing effective prompts to elicit desired responses from LLMs.

**Quantization**: Reducing model precision (e.g., from fp16 to int4) to decrease memory usage while maintaining accuracy. TorchAO reduces VRAM by ~58%.

**RAG (Retrieval-Augmented Generation)**: Combining information retrieval with text generation to provide factual, source-based responses.

**Token**: The basic unit of text processed by LLMs. Roughly 3/4 of a word on average.

**YaRN (Yet another RoPE extensioN)**: A technique to extend context windows beyond native training length through positional encoding modifications.

---

## Architecture Terms

**ADR (Architecture Decision Record)**: Documents that capture important architectural decisions with context and consequences. DocMind has 23 active ADRs.

**Async/Await**: Programming pattern for non-blocking I/O operations, ensuring UI responsiveness during processing.

**Dependency Injection**: Design pattern where components receive dependencies rather than creating them, improving testability.

**Event-Driven Architecture**: System design where components communicate through events rather than direct calls.

**Feature Flag**: Boolean configuration that enables/disables functionality at runtime without code changes.

**Library-First Principle**: Architectural approach prioritizing proven libraries over custom code (KISS > DRY > YAGNI).

**Microservices**: Architectural style where applications are built as suites of independently deployable services.

**Monorepo**: Single repository containing all project code, promoting code sharing and atomic changes.

**Singleton Pattern**: Design pattern ensuring only one instance of a class exists (e.g., LlamaIndex Settings).

**Supervisor Pattern**: Orchestration approach where a central supervisor coordinates multiple specialized agents.

---

## Document Processing

**Chunking**: Dividing documents into smaller, semantically coherent segments for processing. Default: 512 tokens with 50 token overlap.

**Hi-res Strategy**: High-resolution parsing mode in UnstructuredReader for accurate layout preservation.

**IngestionCache**: LlamaIndex component that caches processed documents to avoid redundant parsing.

**IngestionPipeline**: LlamaIndex pipeline for document processing with stages for parsing, chunking, and embedding.

**Metadata Extraction**: Capturing document properties like title, author, creation date during processing.

**Multimodal Extraction**: Processing text, tables, and images from documents as distinct elements.

**OCR (Optical Character Recognition)**: Converting images of text into machine-readable text (future enhancement).

**Semantic Chunking**: Splitting text at natural boundaries (sentences, paragraphs) to preserve meaning.

**SentenceSplitter**: LlamaIndex component for semantic text chunking with configurable size and overlap.

**UnstructuredReader**: Library for parsing various document formats (PDF, DOCX, HTML) with structure preservation.

---

## Infrastructure

**Backend**: The LLM inference engine. DocMind supports Ollama, LlamaCPP, and vLLM backends.

**CUDA**: NVIDIA's parallel computing platform for GPU acceleration.

**Device Map**: PyTorch configuration for distributing model layers across available devices (device_map="auto").

**Docker**: Containerization platform for consistent deployment across environments.

**Environment Variables**: Configuration values set outside the application code (stored in .env file).

**GPU (Graphics Processing Unit)**: Hardware accelerator for parallel computation. RTX 4090 recommended.

**Health Check**: Endpoint that reports system status and availability (/health).

**SQLite**: Lightweight embedded database used for persistence with WAL mode for concurrency.

**Tenacity**: Python library for retry logic with exponential backoff on transient failures.

**VRAM (Video RAM)**: GPU memory for storing models and intermediate computations. Target: <14GB usage.

**WAL (Write-Ahead Logging)**: SQLite mode enabling concurrent reads while writing.

---

## Libraries & Frameworks

**LangChain**: Popular LLM framework (not used - DocMind is pure LlamaIndex).

**LangGraph**: Framework for building stateful multi-agent applications with graph-based workflows.

**langgraph-supervisor**: Pre-built library providing supervisor patterns for agent coordination.

**LlamaIndex**: Core framework for building RAG applications, providing document processing, retrieval, and agent capabilities.

**Loguru**: Modern Python logging library with structured output and rich formatting.

**Ollama**: Local LLM inference server supporting various model formats.

**Pydantic**: Data validation library using Python type annotations.

**PyTorch**: Deep learning framework used for model inference and GPU operations.

**Qdrant**: High-performance vector database supporting hybrid search.

**Streamlit**: Python framework for building data applications with minimal frontend code.

**TorchAO**: PyTorch library for model optimization including quantization.

**Transformers**: Hugging Face library for working with transformer models.

**vLLM**: High-throughput LLM inference engine with PagedAttention.

---

## Models & Embeddings

**BGE-large-en-v1.5**: BAAI's dense embedding model producing 1024-dimensional vectors for semantic search.

**BGE-reranker-v2-m3**: Cross-encoder model for reranking retrieved documents by relevance.

**CLIP ViT-B/32**: OpenAI's multimodal model for image embeddings (512 dimensions).

**Dense Embeddings**: Continuous vector representations capturing semantic meaning.

**Multimodal Embeddings**: Vector representations for non-text content like images.

**PropertyGraphIndex**: LlamaIndex component for GraphRAG enabling relationship extraction and multi-hop reasoning.

**Qwen3-14B**: Default LLM with 14B parameters, 32K native context, excellent performance/efficiency ratio.

**Sparse Embeddings**: High-dimensional vectors with mostly zero values, capturing exact term matches.

**SPLADE++**: Sparse embedding model with learned term expansion for keyword search.

---

## Performance Metrics

**Latency**: Time delay between request and response. Target: <2 seconds P95.

**NDCG (Normalized Discounted Cumulative Gain)**: Metric for ranking quality evaluation. Target: >0.8 at rank 10.

**P95/P99**: 95th/99th percentile metrics, indicating performance for most users.

**Precision@K**: Fraction of relevant documents in top K results.

**QPS (Queries Per Second)**: Throughput metric for search systems.

**Recall**: Fraction of relevant documents retrieved from total relevant documents.

**Throughput**: Amount of work completed per unit time. Document processing target: >50 pages/second.

**Time to First Token (TTFT)**: Latency before streaming response begins.

**Tokens Per Second**: LLM generation speed. Target: ~1000 tokens/sec on RTX 4090.

---

## Retrieval & Search

**BM25**: Traditional keyword-based ranking algorithm (part of sparse search).

**GraphRAG**: Graph-based retrieval using entity relationships and knowledge graphs for complex queries.

**Hybrid Search**: Combining dense (semantic) and sparse (keyword) search strategies.

**k-NN (k-Nearest Neighbors)**: Algorithm for finding similar vectors in embedding space.

**Reranking**: Post-retrieval step that re-scores documents using a more sophisticated model.

**Reciprocal Rank Fusion (RRF)**: Algorithm for combining rankings from multiple retrieval strategies (k=60).

**Semantic Search**: Finding documents based on meaning rather than exact keyword matches.

**Vector Database**: Specialized database optimized for storing and searching high-dimensional vectors.

**Vector Index**: Data structure enabling efficient similarity search in high-dimensional space.

---

## Acronyms Quick Reference

| Acronym | Full Form |
|---------|-----------|
| ADR | Architecture Decision Record |
| API | Application Programming Interface |
| BDD | Behavior-Driven Development |
| CPU | Central Processing Unit |
| CRUD | Create, Read, Update, Delete |
| CSRF | Cross-Site Request Forgery |
| DRY | Don't Repeat Yourself |
| GPU | Graphics Processing Unit |
| JSON | JavaScript Object Notation |
| KISS | Keep It Simple, Stupid |
| LLM | Large Language Model |
| NDCG | Normalized Discounted Cumulative Gain |
| OCR | Optical Character Recognition |
| PRD | Product Requirements Document |
| QPS | Queries Per Second |
| RAG | Retrieval-Augmented Generation |
| RRF | Reciprocal Rank Fusion |
| SQL | Structured Query Language |
| UI | User Interface |
| UUID | Universally Unique Identifier |
| VRAM | Video Random Access Memory |
| WAL | Write-Ahead Logging |
| XSS | Cross-Site Scripting |
| YAGNI | You Aren't Gonna Need It |

---

## Usage Notes

1. **Context-Specific Definitions**: Some terms may have specific meanings within DocMind that differ from general usage.

2. **Version-Specific Information**: Model names and version numbers reflect the current implementation (2025-08-19).

3. **Cross-References**: Terms may reference each other. Use browser search (Ctrl+F) to find related concepts.

4. **Updates**: This glossary will be updated as new terms are introduced during implementation.

---

*For questions about terms not included in this glossary, consult the specification documents or technical team.*
