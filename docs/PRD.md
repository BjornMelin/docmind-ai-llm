# Product Requirements Document (PRD) for DocMind AI

## 1. Introduction

### 1.1 Purpose

This PRD defines the requirements for DocMind AI, a Streamlit-based application for local AI-powered document analysis using Ollama LLMs. It ensures privacy by processing documents locally while providing advanced features like hybrid search and GPU acceleration.

### 1.2 Scope

DocMind AI allows users to upload documents, analyze them with customizable prompts, and interact via chat. It supports multiple formats, structured outputs, and optimized embeddings. Out of scope: Cloud integration, real-time collaboration, mobile apps.

### 1.3 Target Audience

- End-users: Professionals needing document insights (e.g., researchers, analysts).
- Developers: Contributors extending the app.

### 1.4 Assumptions and Dependencies

- Users have Ollama installed and running locally.
- Python 3.9+ and dependencies via uv.
- Optional: NVIDIA GPU for performance boosts.
- Dependencies: Streamlit, LangChain, Qdrant, FastEmbed, Jina embeddings.

## 2. Functional Requirements

### 2.1 Core Features

- **Document Upload:** Support for PDF, DOCX, TXT, XLSX, MD, JSON, XML, RTF, CSV, MSG, PPTX, ODT, EPUB, and code files.
- **Analysis Options:** Customizable prompts, tones, instructions, length/detail, and modes (individual/combined).
- **Analysis Processing:** Extract summaries, key insights, action items, open questions using LLMChain and Pydantic.
- **Interactive Chat:** RAG-based chat with hybrid search (dense: Jina v4, sparse: FastEmbed SPLADE++) and submodular reranking.
- **Export Results:** JSON and Markdown exports.
- **Session Persistence:** Save/load sessions via pickle.

### 2.2 User Interface

- Sidebar: Model selection, backend (Ollama/LlamaCpp/LM Studio), GPU toggle, context size.
- Main: Upload, previews (multimodal for PDF), analysis configs, results display, chat input.

### 2.3 Advanced Features

- **Hybrid Search:** Qdrant v1.15.0 with dense/sparse embeddings for improved recall.
- **GPU Optimization:** Auto-detect VRAM, suggest models, full offload for inference/embeddings.
- **Chunking:** Late chunking with NLTK and mean pooling for accuracy.

## 3. Non-Functional Requirements

### 3.1 Performance

- Latency: <5s for analysis on standard hardware; <2s with GPU.
- Throughput: 50+ TPS for 8B models on RTX 4090.
- Scalability: Handle up to 100MB documents via chunking.

### 3.2 Security and Privacy

- Local-only processing; no data transmission.
- Secure session storage (pickle with warnings).

### 3.3 Usability

- Intuitive Streamlit UI with theming (light/dark/auto).
- Error handling: Graceful failures with logs.

### 3.4 Reliability

- Fallbacks: CPU if no GPU; raw output if parsing fails.
- Testing: Unit tests for utils, integration for app.py.

### 3.5 Maintainability

- Code Quality: Ruff linting, Google docstrings, type hints.
- Principles: KISS, DRY, YAGNI.

## 4. Risks and Mitigations

- Risk: Model incompatibility—Mitigation: Test with suggested models (Qwen series).
- Risk: High resource use—Mitigation: GPU toggles, chunking.
- Risk: Dependency updates—Mitigation: Pinned versions in pyproject.toml.

## 5. Timeline and Milestones

- MVP: Core upload/analysis (Complete).
- v0.2: Hybrid search/GPU (In Progress).
- v1.0: Full multimodal, submodular opt (Q3 2025).

## 6. Appendices

- Architecture: See [developers/architecture.md](developers/architecture.md).
- ADRs: See [adrs/](adrs/).
