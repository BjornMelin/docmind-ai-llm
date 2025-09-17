# 🧠 DocMind AI: Local LLM for AI-Powered Document Analysis

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/🦙_LlamaIndex-000000?style=for-the-badge)
![LangGraph](https://img.shields.io/badge/🔗_LangGraph-4A90E2?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/🦙_Ollama-000000?style=for-the-badge)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GitHub](https://img.shields.io/badge/GitHub-BjornMelin-181717?logo=github)](https://github.com/BjornMelin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Bjorn%20Melin-0077B5?logo=linkedin)](https://www.linkedin.com/in/bjorn-melin/)

**DocMind AI** provides local document analysis with zero cloud dependency. This system combines hybrid search (dense + sparse embeddings), knowledge graph extraction, and a 5-agent coordination system to extract and analyze information from your PDFs, Office docs, and multimedia content. Built on LlamaIndex pipelines with LangGraph supervisor orchestration and Qwen3-4B-Instruct-2507's FULL 262K context capability through INT8 KV cache optimization, it provides document intelligence that runs entirely on your hardware—with GPU acceleration and agent coordination.

**Architecture**: Traditional document analysis tools either send your data to the cloud (privacy risk) or provide basic keyword search (limited intelligence). DocMind AI provides AI reasoning with complete data privacy. Process complex queries that require multiple reasoning strategies, extract entities and relationships, and get contextual answers—all while your sensitive documents never leave your machine.

## ✨ Features of DocMind AI

- **Privacy-Focused:** Local processing ensures data security without cloud dependency.

- **Library-First Ingestion Pipeline:** LlamaIndex `IngestionPipeline` orchestrates Unstructured parsing, deterministic hashing, DuckDB caching, and AES-GCM page image handling with OpenTelemetry spans for each run.

- **Versatile Document Handling:** Supports multiple file formats:
  - 📄 PDF
  - 📑 DOCX
  - 📝 TXT
  - 📊 XLSX
  - 🌐 MD (Markdown)
  - 🗃️ JSON
  - 🗂️ XML
  - 🔤 RTF
  - 📇 CSV
  - 📧 MSG (Email)
  - 🖥️ PPTX (PowerPoint)
  - 📘 ODT (OpenDocument Text)
  - 📚 EPUB (E-book)
  - 💻 Code files (PY, JS, JAVA, TS, TSX, C, CPP, H, and more)

- **Multi-Agent Coordination:** LangGraph supervisor coordinating 5 specialized agents: query router, query planner, retrieval expert, result synthesizer, and response validator.

- **Retrieval/Router:** RouterQueryEngine composed via `router_factory` with tools `semantic_search`, `hybrid_search` (Qdrant server‑side fusion), and optional `knowledge_graph`; uses async/batching where appropriate.

- **Hybrid Retrieval:** Qdrant Query API server‑side fusion (RRF default, DBSF optional) over named vectors `text-dense` (BGE‑M3; COSINE) and `text-sparse` (FastEmbed BM42/BM25 with IDF). Dense via LlamaIndex; sparse via FastEmbed.

- **Knowledge Graph (optional):** Adds a `knowledge_graph` router tool when a PropertyGraphIndex is present and healthy; uses spaCy entity extraction; selector prefers `PydanticSingleSelector` then `LLMSingleSelector`; falls back to vector/hybrid when absent.

- **Multimodal Processing:** Unstructured hi‑res parsing for PDFs with text, tables, and images; visual features scored with SigLIP by default (CLIP optional).

- **Always-on Reranking:** Text via BGE Cross-Encoder and visual via SigLIP; optional ColPali on capable GPUs. Deterministic, batch‑wise cancellation; fail‑open; SigLIP loader cached.

- **Offline-First Design:** 100% local processing with no external API dependencies.

- **GPU Acceleration:** CUDA support with mixed precision and FP8 quantization via vLLM FlashInfer backend for optimized performance.

- **Session Persistence:** SQLite WAL with local multi-process support for concurrent access.

- **Docker Support:** Easy deployment with Docker and Docker Compose.

- **Intelligent Caching:** High-performance document processing cache for rapid re-analysis.

- **Robust Error Handling:** Reliable retry strategies with exponential backoff.

- **Structured Logging:** Contextual logging with automatic rotation and JSON output.
- **Encrypted Page Images (AES-GCM):** Optional at-rest encryption for rendered PDF page images using AES-GCM with KID as AAD; `.enc` files are decrypted just-in-time for visual scoring and immediately cleaned up.

- **Simple Configuration:** Environment variables and Streamlit native config for easy setup.

## 📖 Table of Contents

- [🧠 DocMind AI: Local LLM for AI-Powered Document Analysis](#-docmind-ai-local-llm-for-ai-powered-document-analysis)
  - [✨ Features of DocMind AI](#-features-of-docmind-ai)
  - [📖 Table of Contents](#-table-of-contents)
  - [🚀 Getting Started with DocMind AI](#-getting-started-with-docmind-ai)
    - [📋 Prerequisites](#-prerequisites)
    - [⚙️ Installation](#️-installation)
    - [▶️ Running the App](#️-running-the-app)
  - [💻 Usage](#-usage)
    - [🎛️ Selecting a Model](#️-selecting-a-model)
    - [📁 Uploading Documents](#-uploading-documents)
    - [✍️ Choosing Prompts](#️-choosing-prompts)
    - [😃 Selecting Tone](#-selecting-tone)
    - [🧮 Selecting Instructions](#-selecting-instructions)
    - [📏 Setting Length/Detail](#-setting-lengthdetail)
    - [🗂️ Choosing Analysis Mode](#️-choosing-analysis-mode)
    - [🧠 Analyzing Documents](#-analyzing-documents)
    - [💬 Interacting with the LLM](#-interacting-with-the-llm)
  - [🔧 API Usage Examples](#-api-usage-examples)
    - [Programmatic Document Analysis](#programmatic-document-analysis)
    - [Custom Configuration](#custom-configuration)
    - [Batch Document Processing](#batch-document-processing)
  - [🏗️ Architecture](#️-architecture)
  - [🛠️ Implementation Details](#️-implementation-details)
    - [Document Processing Pipeline](#document-processing-pipeline)
    - [Hybrid Retrieval Architecture](#hybrid-retrieval-architecture)
    - [Multi-Agent Coordination](#multi-agent-coordination)
    - [Performance Optimizations](#performance-optimizations)
  - [⚙️ Configuration](#️-configuration)
    - [Configuration Philosophy](#configuration-philosophy)
    - [Environment Variables](#environment-variables)
    - [Enable DSPy Optimization (optional)](#enable-dspy-optimization-optional)
    - [Additional Configuration](#additional-configuration)
  - [📊 Performance Benchmarks](#-performance-benchmarks)
    - [Performance Metrics](#performance-metrics)
    - [Caching Performance](#caching-performance)
    - [Hybrid Search Performance](#hybrid-search-performance)
    - [System Resource Usage](#system-resource-usage)
    - [Scalability Benchmarks](#scalability-benchmarks)
    - [Retrieval \& Reranking Defaults](#retrieval--reranking-defaults)
      - [Operational Flags (local-first)](#operational-flags-local-first)
  - [🔧 Offline Operation](#-offline-operation)
    - [Prerequisites for Offline Use](#prerequisites-for-offline-use)
    - [Prefetch Model Weights](#prefetch-model-weights)
    - [Snapshots \& Staleness](#snapshots--staleness)
    - [GraphRAG Exports \& Seeds](#graphrag-exports--seeds)
    - [Model Requirements](#model-requirements)
  - [🛠️ Troubleshooting](#️-troubleshooting)
    - [Common Issues](#common-issues)
      - [1. Ollama Connection Errors](#1-ollama-connection-errors)
      - [2. GPU Not Detected](#2-gpu-not-detected)
      - [3. Model Download Issues](#3-model-download-issues)
      - [4. Memory Issues](#4-memory-issues)
      - [5. Document Processing Errors](#5-document-processing-errors)
      - [6. vLLM FlashInfer Installation Issues](#6-vllm-flashinfer-installation-issues)
      - [7. PyTorch 2.7.1 Compatibility Issues](#7-pytorch-271-compatibility-issues)
      - [8. GPU Memory Issues (16GB RTX 4090)](#8-gpu-memory-issues-16gb-rtx-4090)
      - [9. Performance Validation](#9-performance-validation)
    - [Performance Optimization](#performance-optimization)
    - [Getting Help](#getting-help)
  - [📖 How to Cite](#-how-to-cite)
  - [🙌 Contributing](#-contributing)
    - [Development Guidelines](#development-guidelines)
      - [🧪 Tests and CI](#-tests-and-ci)
  - [📃 License](#-license)
  - [📡 Observability](#-observability)

## 🚀 Getting Started with DocMind AI

### 📋 Prerequisites

- [Ollama](https://ollama.com/) installed and running locally.

- Python 3.11+ (tested with 3.11, 3.12)

- (Optional) Docker and Docker Compose for containerized deployment.

- (Optional) NVIDIA GPU (e.g., RTX 4090 Laptop) with at least 16GB VRAM for 262K context capability and accelerated performance.

### ⚙️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/BjornMelin/docmind-ai-llm.git
   cd docmind-ai-llm
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

   _Need OTLP exporters and cross-platform snapshot locking?_ Install the optional observability extras as well:

   ```bash
   uv sync --extra observability
   ```

   **Key Dependencies Included:**
   - **LlamaIndex Core**: Retrieval, RouterQueryEngine, IngestionPipeline, PropertyGraphIndex
   - **LangGraph (0.5.4)**: 5-agent supervisor orchestration with langgraph-supervisor library
   - **Streamlit (1.48.0)**: Web interface framework
   - **Ollama (0.5.1)**: Local LLM integration
   - **Qdrant Client (1.15.1)**: Vector database operations
   - **FastEmbed (0.3.0+)**: High-performance embeddings
   - **Tenacity (8.0.0+)**: Retry strategies with exponential backoff
   - **Loguru (0.7.0+)**: Structured logging
   - **Pydantic (2.11.7)**: Data validation and settings

3. **Install spaCy language model:**

   DocMind AI uses spaCy for named entity recognition and linguistic analysis. Install the English language model:

   ```bash
   # Install the small English model (recommended, ~15MB)
   uv run python -m spacy download en_core_web_sm
   
   # Optional: Install larger models for better accuracy
   # Medium model (~50MB): uv run python -m spacy download en_core_web_md
   # Large model (~560MB): uv run python -m spacy download en_core_web_lg
   ```

   **Note:** spaCy models are downloaded and cached locally. The application will automatically attempt to download `en_core_web_sm` if not found, but manual installation ensures offline functionality.

4. **Set up environment configuration:**

   Copy the example environment file and configure your settings:

   ```bash
   cp .env.example .env
   # Edit .env with your preferred settings

   # Example — LM Studio (local, OpenAI-compatible):
   #   DOCMIND_LLM_BACKEND=lmstudio
   #   DOCMIND_OPENAI__BASE_URL=http://localhost:1234/v1
   #   DOCMIND_OPENAI__API_KEY=not-needed

   # Example — vLLM OpenAI-compatible server:
   #   DOCMIND_LLM_BACKEND=vllm
   #   DOCMIND_OPENAI__BASE_URL=http://localhost:8000/v1
   #   DOCMIND_OPENAI__API_KEY=not-needed

   # Example — llama.cpp server:
   #   DOCMIND_LLM_BACKEND=llamacpp
   #   DOCMIND_OPENAI__BASE_URL=http://localhost:8080/v1
   #   DOCMIND_OPENAI__API_KEY=not-needed

   # Offline-first recommended:
   #   HF_HUB_OFFLINE=1
   #   TRANSFORMERS_OFFLINE=1

   # Optional — OpenAI Cloud (breaks strict offline):
   #   DOCMIND_OPENAI__BASE_URL=https://api.openai.com/v1
   #   DOCMIND_OPENAI__API_KEY=sk-...
   #   DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true
   ```

For a complete overview (including a Local vs Cloud matrix), see
`docs/developers/configuration-reference.md#openai-compatible-local-servers-lm-studio-vllm-llamacpp`.

5. **(Optional) Install GPU support for RTX 4090 with vLLM FlashInfer:**

   **RECOMMENDED: vLLM FlashInfer Stack** for Qwen3-4B-Instruct-2507-FP8 with 128K context:

   ```bash
   # Phase 1: Verify CUDA installation
   nvcc --version  # Should show CUDA 12.8+
   nvidia-smi     # Verify RTX 4090 detection

   # Phase 2: Install PyTorch 2.7.1 with CUDA 12.8 (DEFINITIVE - TESTED APPROACH)
   uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
       --extra-index-url https://download.pytorch.org/whl/cu128

   # Phase 3: Install vLLM with FlashInfer support (includes FlashInfer automatically)
   uv pip install "vllm[flashinfer]>=0.10.1" \
       --extra-index-url https://download.pytorch.org/whl/cu128

   # Phase 4: Install remaining GPU dependencies
   uv sync --extra gpu
   
   # Phase 5: Verify installation
   python -c "import vllm; import torch; print(f'vLLM: {vllm.__version__}, PyTorch: {torch.__version__}')"
   ```

   **Hardware Requirements:**
   - NVIDIA RTX 4090 (16GB VRAM minimum for 128K context)
   - CUDA Toolkit 12.8+
   - NVIDIA Driver 550.54.14+
   - Compute Capability 8.9 (RTX 4090)

   **Performance Targets Achieved:**
   - **100-160 tok/s decode speed** (typical: 120-180 with FlashInfer)
   - **800-1300 tok/s prefill speed** (typical: 900-1400 with RTX 4090)
   - **FP8 quantization** for optimal 16GB VRAM usage (12-14GB typical)
   - **128K context support** with INT8 KV cache optimization

   **Fallback Installation** (if FlashInfer fails):

   ```bash
   # Fallback: vLLM CUDA-only installation with PyTorch 2.7.1
   uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
       --extra-index-url https://download.pytorch.org/whl/cu128
   uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
   uv sync --extra gpu
   ```

   See [GPU Setup Guide](docs/developers/archived/gpu-setup.md) for detailed configuration and troubleshooting.

### ▶️ Running the App

**Locally:**

```bash
streamlit run src/app.py
```

**With Docker:**

```bash
docker-compose up --build
```

Access the app at `http://localhost:8501`.

## 💻 Usage

### 🎛️ Selecting a Model

1. **Start Ollama service** (if not already running):

   ```bash
   ollama serve
   ```

2. **Enter the Ollama Base URL** (default: `http://localhost:11434`).

3. **Select an Ollama Model Name** (e.g., `qwen3-4b-instruct-2507` for 128K context). If the model isn't installed:

   ```bash
   ollama pull qwen3-4b-instruct-2507
   ```

4. **Toggle "Use GPU if available"** for accelerated processing (recommended for NVIDIA GPUs with 4GB+ VRAM).

5. **Adjust Context Size** based on your model and hardware:
   - **2048**: Small models, limited VRAM
   - **4096**: Standard setting for most use cases  
   - **8192+**: Large models with sufficient resources
   - **262144**: FULL 262K context with INT8 KV cache (Qwen3-4B-Instruct-2507 + 16GB VRAM)

### 📁 Uploading Documents

Upload one or more documents via the **"Browse files"** button. Supported formats include PDF, DOCX, TXT, and more (see [Features](#-features-of-docmind-ai)). PDF previews include first-page images for multimodal support.

### ✍️ Choosing Prompts

DocMind uses a file-based prompt template system (SPEC‑020) powered by LlamaIndex’s `RichPromptTemplate` (Jinja under the hood).

- The app lists templates from `templates/prompts/*.prompt.md` (YAML front matter + Jinja body).
- Presets for tone/role/length live in `templates/presets/*.yaml` and are exposed in the UI selectors.

Add or edit a template by creating a new `*.prompt.md` file; restart the app to pick it up. Example:

```yaml
---
id: comprehensive-analysis
name: Comprehensive Document Analysis
description: Summary, key insights, action items, open questions
required: [context]
defaults:
  tone: { description: "Use a professional, objective tone." }
version: 1
---
{% chat role="system" %}
{{ tone.description }}
{% endchat %}

{% chat role="user" %}
Context:
{{ context }}
Tasks: summarize, extract insights, list actions, raise open questions
{% endchat %}
```

You can also use the prompting API programmatically (see developer guide: [Adding a Prompt Template](docs/developers/guides/adding-prompt-template.md)):

```python
from src.prompting import list_templates, render_prompt

tpl = next(t for t in list_templates() if t.id == "comprehensive-analysis")
ctx = {"context": "…", "tone": {"description": "Use a neutral tone."}}
prompt = render_prompt(tpl.id, ctx)
```

### 😃 Selecting Tone

Choose the desired tone for LLM responses:

- **Professional:** Formal and objective.

- **Academic:** Scholarly and research-focused.

- **Informal:** Casual and conversational.

- **Creative:** Imaginative and expressive.

- **Neutral:** Balanced and unbiased.

- **Direct:** Concise and straightforward.

- **Empathetic:** Compassionate and understanding.

- **Humorous:** Lighthearted and witty.

- **Authoritative:** Confident and expert-like.

- **Inquisitive:** Curious and exploratory.

### 🧮 Selecting Instructions

Select the LLM's role or provide custom instructions:

- **General Assistant:** Helpful and versatile.

- **Researcher:** Deep, analytical insights.

- **Software Engineer:** Technical and code-focused.

- **Product Manager:** Strategic and user-centric.

- **Data Scientist:** Data-driven analysis.

- **Business Analyst:** Business and strategic focus.

- **Technical Writer:** Clear and concise documentation.

- **Marketing Specialist:** Branding and engagement-oriented.

- **HR Manager:** Human resources perspective.

- **Legal Advisor:** Legal and compliance-focused.

- **Custom Instructions:** Specify your own role or instructions.

### 📏 Setting Length/Detail

Select the desired output length and detail:

- **Concise:** Brief and to-the-point.

- **Detailed:** Thorough and in-depth.

- **Comprehensive:** Extensive and exhaustive.

- **Bullet Points:** Structured list format.

### 🗂️ Choosing Analysis Mode

Choose how documents are analyzed:

- **Analyze each document separately:** Individual analysis for each file.

- **Combine analysis for all documents:** Holistic analysis across all uploaded files.

### 🧠 Analyzing Documents

1. Upload documents.
2. Configure analysis options (prompt, tone, instructions, length, mode).
3. Enable **Chunked Analysis** for large documents, **Late Chunking** for accuracy, or **Multi-Vector Embeddings** for enhanced retrieval.
4. Click **"Extract and Analyze"** to process.

Results include summaries, insights, action items, and open questions, exportable as JSON or Markdown.

### 💬 Interacting with the LLM

Use the chat interface to ask follow-up questions. The LLM leverages hybrid search (BGE‑M3 unified dense + sparse embeddings) with multimodal reranking (BGE text + SigLIP visual; ColPali optional) for context‑aware, high‑quality responses.

## 🔧 API Usage Examples

### Programmatic Document Analysis

```python
import asyncio
from pathlib import Path
from src.config import settings
from src.utils.document import load_documents_unstructured
from src.utils.embedding import create_index_async
from src.agents.coordinator import MultiAgentCoordinator

async def analyze_document(file_path: str, query: str):
    """Example: Analyze a document programmatically."""
    
    # Load and process document
    documents = await load_documents_unstructured([Path(file_path)], settings)
    index = await create_index_async(documents, settings)
    
    # Create coordinator and run analysis
    coordinator = MultiAgentCoordinator()
    response = coordinator.process_query(query, context=None)
    return response

# Usage
async def main():
    result = await analyze_document(
        "path/to/document.pdf", 
        "Summarize the key findings and action items"
    )
    print(result)

asyncio.run(main())
```

### Custom Configuration

```python
from src.config import settings
import os

# Override default settings
os.environ["DOCMIND_VLLM__MODEL"] = "qwen3-4b-instruct-2507-FP8"
os.environ["DOCMIND_ENABLE_GPU_ACCELERATION"] = "true"
os.environ["DOCMIND_RETRIEVAL__USE_RERANKING"] = "true"

print(f"Using model: {settings.vllm.model}")
print(f"GPU enabled: {settings.enable_gpu_acceleration}")
print(f"Reranking enabled: {settings.retrieval.use_reranking}")
```

### Batch Document Processing

```python
import asyncio
from pathlib import Path
from src.config import settings
from src.utils.document import load_documents_unstructured
from src.utils.embedding import create_index_async

async def process_document_folder(folder_path: str):
    """Process all supported documents in a folder."""
    
    # Find all supported documents
    folder = Path(folder_path)
    supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.json'}
    documents_paths = [
        f for f in folder.rglob("*") 
        if f.suffix.lower() in supported_extensions
    ]
    
    if not documents_paths:
        print("No supported documents found")
        return
    
    print(f"Processing {len(documents_paths)} documents...")
    
    # Load and index all documents
    documents = await load_documents_unstructured(documents_paths, settings)
    index = await create_index_async(documents, settings)
    
    print("Documents processed and indexed successfully!")
    return index

# Usage
asyncio.run(process_document_folder("/path/to/documents"))
```

## 🏗️ Architecture

```mermaid
graph TD
    A[Document Upload<br/>Streamlit UI] --> B[Unstructured Parser<br/>hi-res parsing]
    B --> C[Text + Images + Tables<br/>Multimodal Content]
    C --> D[LlamaIndex Ingestion Pipeline<br/>Document Processing]
    
    D --> E[Unstructured Title-Based Chunking<br/>IngestionPipeline]
    D --> F[spaCy NLP Pipeline<br/>Entity Recognition]
    
    E --> G[Multi-Modal Embeddings]
    F --> H[Knowledge Graph Builder<br/>Entity Relations]
    
    G --> I[BGE‑M3 Unified (dense+sparse)<br/>Multimodal: SigLIP (default) or CLIP]
    I --> J[Qdrant Vector Store<br/>Server-side Fusion (RRF default; DBSF optional)]
    
    H --> K[Knowledge Graph Index<br/>PropertyGraphIndex (path_depth=1)]
    
    J --> L[LlamaIndex Retrieval/Router<br/>Multi-Strategy Composition]
    K --> L
    
    L --> M[LangGraph Supervisor System<br/>5-Agent Coordination]
    M --> N[5 Specialized Agents:<br/>• Query Router<br/>• Query Planner<br/>• Retrieval Expert<br/>• Result Synthesizer<br/>• Response Validator]
    
    N --> O[Multimodal Reranking<br/>BGE + SigLIP (ColPali optional)]
    O --> P[Local LLM Backend<br/>Ollama/LM Studio/LlamaCpp]
    P --> Q[Supervisor Coordination<br/>Agent-to-Agent Handoffs]
    Q --> R[Response Synthesis<br/>Quality Validation]
    R --> S[Streamlit Interface<br/>Chat + Analysis Results]
    
    T[SQLite WAL Database<br/>Session Persistence] <--> M
    T <--> L
    U[Ingestion Cache<br/>Document Processing] <--> D
    V[GPU Acceleration<br/>CUDA + Mixed Precision] <--> I
    V <--> O
    W[Human-in-the-Loop<br/>Agent Interrupts] <--> M
    
    subgraph "Local Infrastructure"
        P
        T
        U
        J
    end
    
    subgraph "AI Processing"
        I
        O
        M
        N
    end
```

## 🛠️ Implementation Details

### Document Processing Pipeline

- **Parsing:** Unstructured hi-res strategy extracts text, tables, and images from PDFs/Office docs with OCR support

- **Chunking:** Unstructured title‑based chunking via LlamaIndex IngestionPipeline; preserves tables, page images, and rich metadata

- **Metadata:** spaCy en_core_web_sm for entity extraction and relationship mapping

### Hybrid Retrieval Architecture

- **Unified Text Embeddings:** BGE-M3 (BAAI/bge-m3) provides both dense (1024D) and sparse embeddings in a single model for semantic similarity and neural lexical matching

- **Multimodal:** SigLIP (default) visual scoring (CLIP optional) with FP16 acceleration

- **Fusion:** Server‑side RRF via Qdrant Query API (DBSF optional via env); no client‑side fusion knobs
- **De‑duplication:** Configurable key via `DOCMIND_RETRIEVAL__DEDUP_KEY` (page_id|doc_id); default = `page_id`.
- **Router composition:** See `src/retrieval/router_factory.py` (tools: `semantic_search`, `hybrid_search`, `knowledge_graph`). Selector preference: `PydanticSingleSelector` (preferred) → `LLMSingleSelector` fallback. The `knowledge_graph` tool is activated only when a PropertyGraphIndex is present and healthy; otherwise the router uses vector/hybrid only.

- **Storage:** Qdrant vector database with metadata filtering and concurrent access

### Multi-Agent Coordination

- **Supervisor Pattern:** LangGraph supervisor using `langgraph-supervisor` library for proven coordination patterns with automatic state management

- **5 Specialized Agents:**
  - **Query Router:** Analyzes query complexity and determines optimal retrieval strategy
  - **Query Planner:** Decomposes complex queries into manageable sub-tasks for better processing
  - **Retrieval Expert:** Executes optimized retrieval with server‑side hybrid (Qdrant) and optional GraphRAG; supports optional DSPy query optimization when enabled
  - **Result Synthesizer:** Combines and reconciles results from multiple retrieval passes with deduplication
  - **Response Validator:** Validates response quality, accuracy, and completeness before final output

- **Enhanced Capabilities:** Optional GraphRAG for multi‑hop reasoning and optional DSPy query optimization for query rewriting

- **Workflow Coordination:** Supervisor automatically routes between agents based on query complexity with <300ms coordination overhead

- **Session Management:** SQLite WAL database with built-in conversation context preservation and error recovery

- **Async Execution:** Concurrent agent operations with automatic resource management and fallback mechanisms

### Performance Optimizations

- **GPU Acceleration:** CUDA support with FP8 quantization via vLLM FlashInfer backend and torch.compile optimization

- **Async processing:** Concurrent ingestion and retrieval with built‑in caching and bounded timeouts; no client‑side fusion.

- **Reranking:** Always‑on BGE Cross‑Encoder (text) + SigLIP (visual) RRF merge; optional ColPali on capable GPUs.

- **Memory Management:** Quantization and model size auto-selection based on available VRAM

## ⚙️ Configuration

DocMind AI uses a simple, distributed configuration approach optimized for local desktop applications:

- **Environment Variables**: Runtime configuration via `.env` file
- **Streamlit Native Config**: UI settings via `.streamlit/config.toml`
- **Library Defaults**: Sensible defaults from LlamaIndex, Qdrant, etc.
- **Feature Flags**: Boolean environment variables for experimental features

### Configuration Philosophy

Following KISS principles, configuration is intentionally simple and distributed rather than centralized, avoiding over-engineering for a single-user local application.

### Environment Variables

DocMind AI uses environment variables for configuration. Copy the example file and customize:

```bash
cp .env.example .env
```

Key configuration options in `.env`:

```bash
# Model & Backend Services
DOCMIND_MODEL=Qwen/Qwen3-4B-Instruct-2507
DOCMIND_DEVICE=cuda
DOCMIND_CONTEXT_LENGTH=262144
DOCMIND_LLM_BASE_URL=http://localhost:11434

# Embedding Models (BGE-M3 unified)
DOCMIND_EMBEDDING_MODEL=BAAI/bge-m3
DOCMIND_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Feature Flags
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true
DOCMIND_ENABLE_GRAPHRAG=false
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_QUANT_POLICY=fp8  # FP8 KV cache

# Performance Tuning
DOCMIND_RETRIEVAL_TOP_K=10
DOCMIND_RERANK_TOP_K=5
DOCMIND_CACHE_SIZE_LIMIT=1073741824  # 1GB
```

See the complete [.env.example](.env.example) file for all available configuration options.

### Enable DSPy Optimization (optional)

To turn on query optimization via DSPy:

- Install DSPy: `pip install dspy-ai`
- Enable the feature flag in your `.env`:

```bash
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true
```

Optional tuning (defaults are sensible):

```bash
DOCMIND_DSPY_OPTIMIZATION_ITERATIONS=10
DOCMIND_DSPY_OPTIMIZATION_SAMPLES=20
DOCMIND_DSPY_MAX_RETRIES=3
DOCMIND_DSPY_TEMPERATURE=0.1
DOCMIND_DSPY_METRIC_THRESHOLD=0.8
DOCMIND_ENABLE_DSPY_BOOTSTRAPPING=true
```

Notes:

- DSPy runs in the agents layer and augments retrieval by refining the query; retrieval remains library‑first (server‑side hybrid via Qdrant + reranking).
- If DSPy is not installed or the flag is false, the system falls back gracefully to standard retrieval.

### Additional Configuration

**Streamlit UI Configuration** (`.streamlit/config.toml`):

```toml
[theme]
base = "light"
primaryColor = "#FF4B4B"

[server]
maxUploadSize = 200
```

**Cache Configuration** (automatic via LlamaIndex):

- Document processing cache: `./cache/documents` (1GB limit)
- Embedding cache: In-memory with LRU eviction
- Model cache: Automatic via Hugging Face transformers

## 📊 Performance Benchmarks

> **Note**: The following performance metrics are estimates based on hardware specifications and typical usage patterns. Actual performance may vary depending on hardware configuration, model size, document complexity, and system load. For measured test suite performance, see [Testing Guide](docs/testing/current-testing-guide.md).

### Performance Metrics

| Operation | Performance | Notes |
|-----------|-------------|--------|
| **Document Processing (Cold)** | ~15-30 seconds | 50-page PDF with GPU acceleration |
| **Document Processing (Warm)** | ~2-5 seconds | DiskCache + index caching |
| **Query Response** | 1-3 seconds | Hybrid retrieval + multimodal reranking |
| **5-Agent System Response** | 3-8 seconds | LangGraph supervisor coordination with <200ms overhead |
| **128K Context Processing** | 1.5-3 seconds | 128K context with FP8 KV cache |
| **Vector Search** | <500ms | Qdrant in-memory with GPU embeddings |
| **Test Suite (2,263 tests)** | Varies by tier | Unit/integration/system testing - 3.51% measured coverage |
| **Memory Usage (Idle)** | 400-500MB | Base application |

Note: Realized latency is hardware‑dependent. Reranking uses bounded timeouts (text ≈ 250ms, visual ≈ 150ms) and fails open.
| **Memory Usage (Processing)** | 1.2-2.1GB | During document analysis |
| **GPU Memory Usage** | ~12-14GB | Model + 128K context + embedding cache |

### Caching Performance

**Document Processing Cache:**

- **Cache hit ratio**: High rate for repeated documents

- **Storage efficiency**: ~1GB handles 1000+ documents

- **Cache invalidation**: Automatic based on file content + settings hash

- **Concurrent access**: Multi-process safe with WAL mode

### Hybrid Search Performance

**Retrieval Quality Metrics:**

- **Dense + Sparse RRF**: Improved recall vs single-vector

- **Multimodal Reranking**: Enhanced context quality via SigLIP/ColPali

- **Top-K Retrieval**: <2 seconds for 10K document corpus

- **Knowledge Graph**: Entity extraction <1 second per document

### System Resource Usage

**Memory Profile:**

- **Base application**: ~400MB

- **Document processing**: +500-900MB (depends on file size)

- **Embedding cache**: ~200MB for 1000 documents

- **GPU memory**: 8-16GB (model dependent)

**Disk Usage:**

- **Application**: ~50MB

- **Document cache**: Configurable (default 1GB limit)

- **Vector database**: ~100MB per 1000 documents

- **Model weights**: 2-8GB (embedding + reranking models)

### Scalability Benchmarks

| Document Count | Processing Time | Query Time | Memory Usage |
|---------------|-----------------|------------|--------------|
| 100 docs | 5 minutes | <1 second | 800MB |
| 1,000 docs | 45 minutes | <2 seconds | 1.2GB |
| 5,000 docs | 3.5 hours | <5 seconds | 2.1GB |
| 10,000 docs | 7 hours | <8 seconds | 3.5GB |

> _Benchmarks performed on RTX 4090 Laptop GPU, 16GB RAM, NVMe SSD_

### Retrieval & Reranking Defaults

- Hybrid retrieval uses Qdrant named vectors `text-dense` (1024D COSINE; BGE‑M3) and `text-sparse` (FastEmbed BM42/BM25 + IDF) with server-side fusion via the Query API (Prefetch + FusionQuery; dense uses VectorInput).
- Default fusion = RRF; DBSF is available experimentally with `DOCMIND_RETRIEVAL__FUSION_MODE=dbsf`.
- Prefetch: dense≈200, sparse≈400; fused_top_k=60; page_id de-dup.
- Reranking is always-on: BGE v2‑m3 (text) + SigLIP (visual) with optional ColPali; SigLIP loader is cached; batch‑wise cancellation only.
- No UI toggles; ops overrides via env only.
- Router parity: RouterQueryEngine tools (vector/hybrid/KG) apply the same reranking policy via `node_postprocessors` behind `DOCMIND_RETRIEVAL__USE_RERANKING`.

#### Operational Flags (local-first)

- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` to disable network egress (after predownload).
- `DOCMIND_RETRIEVAL__FUSION_MODE=rrf|dbsf` to control Qdrant fusion.
- `DOCMIND_RETRIEVAL__USE_RERANKING=true|false` (canonical env override; no UI toggle)
- Qdrant runs bound to `127.0.0.1` by default; remote endpoints are disallowed unless explicitly configured.

## 🔧 Offline Operation

DocMind AI is designed for complete offline operation:

### Prerequisites for Offline Use

1. **Install Ollama locally:**

   ```bash
   # Download from https://ollama.com/download
   ollama serve  # Start the service
   ```

2. **Pull required models:**

   ```bash
   ollama pull qwen3-4b-instruct-2507  # Recommended for 128K context
   ollama pull qwen2:7b  # Alternative lightweight model
   ```

3. **Verify GPU setup (optional):**

   ```bash
   nvidia-smi  # Check GPU availability
   uv run python scripts/test_gpu.py --quick  # Validate CUDA setup
   ```

### Prefetch Model Weights

Run once (online) to predownload required models for offline use:

```bash
uv run python scripts/predownload_models.py --cache-dir ./models_cache
```

### Snapshots & Staleness

DocMind snapshots persist indices atomically for reproducible retrieval.

- Manifest fields: `schema_version`, `persist_format_version`, `complete=true`, `created_at`, and `versions` (`app`, `llama_index`, `qdrant_client`, `embed_model`), along with `corpus_hash` and `config_hash`.
- Hashing: `corpus_hash` computed with POSIX relpaths relative to `uploads/` for OS‑agnostic stability.
- Chat autoload: the Chat page loads the latest non‑stale snapshot when available; otherwise it shows a staleness badge and offers to rebuild.

### GraphRAG Exports & Seeds

- Graph exports preserve relation labels when provided by `get_rel_map` (fallback label `related`). Exports: JSONL baseline (portable) and Parquet (optional, requires PyArrow). Export seeding follows a retriever‑first policy: graph → vector → deterministic fallback with dedup and stable tie‑break.
Set env for offline operation:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Model Requirements

| Model Size | RAM Required | GPU VRAM | Performance | Context |
|------------|-------------|----------|-------------|---------|
| 4B (qwen3-4b-instruct-2507-fp8) | 16GB+ | 12-14GB | Best | 128K |
| 7B (e.g., qwen2:7b) | 8GB+ | 4GB+ | Good | 32K |
| 13B | 16GB+ | 8GB+ | Better | 32K |

## 🛠️ Troubleshooting

### Common Issues

#### 1. Ollama Connection Errors

```bash

# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start it
ollama serve
```

#### 2. GPU Not Detected

```bash

# Install GPU dependencies
uv sync --extra gpu

# Verify CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### 3. Model Download Issues

```bash

# Pull models manually
ollama pull qwen3-4b-instruct-2507  # For 128K context
ollama pull qwen2:7b  # Alternative
ollama list  # Verify installation
```

#### 4. Memory Issues

- Reduce context size in UI (262144 → 32768 → 4096)

- Use smaller models (7B instead of 4B for lower VRAM)

- Enable document chunking for large files

- Close other applications to free RAM

#### 5. Document Processing Errors

```bash

# Check supported formats
echo "Supported: PDF, DOCX, TXT, XLSX, CSV, JSON, XML, MD, RTF, MSG, PPTX, ODT, EPUB"

# For unsupported formats, convert to PDF first
```

#### 6. vLLM FlashInfer Installation Issues

```bash
# Check CUDA compatibility
nvcc --version  # Should show CUDA 12.8+
nvidia-smi     # Should show RTX 4090 and compatible driver

# Clean installation if issues occur
uv pip uninstall torch torchvision torchaudio vllm flashinfer-python -y
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install "vllm[flashinfer]>=0.10.1" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Test FlashInfer availability
python -c "import vllm; print('vLLM with FlashInfer imported successfully')"
```

#### 7. PyTorch 2.7.1 Compatibility Issues

**RESOLVED**: PyTorch 2.7.1 compatibility was confirmed in vLLM v0.10.0+ (July 2025). Current project uses vLLM>=0.10.1.

```bash
# Verify versions
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# If using older vLLM, upgrade:
uv pip install --upgrade "vllm[flashinfer]>=0.10.1"
```

#### 8. GPU Memory Issues (16GB RTX 4090)

```bash
# Reduce GPU memory utilization in .env
export VLLM_GPU_MEMORY_UTILIZATION=0.75  # Reduce from 0.85

# Monitor GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1

# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 9. Performance Validation

```bash
# Run performance validation script
uv run python scripts/performance_monitor.py --run-tests --check-regressions

# Expected results for RTX 4090:
# - Decode: 120-180 tokens/second
# - Prefill: 900-1400 tokens/second  
# - VRAM: 12-14GB usage
# - Context: 128K tokens supported
```

### Performance Optimization

1. **Enable GPU acceleration** in the UI sidebar
2. **Use appropriate model sizes** for your hardware
3. **Enable caching** to speed up repeat analysis
4. **Adjust chunk sizes** based on document complexity
5. **Use hybrid search** for better retrieval quality

### Getting Help

- Check logs in `logs/` directory for detailed errors

- Review [troubleshooting guide](docs/user/troubleshooting-reference.md)

- Search existing [GitHub Issues](https://github.com/BjornMelin/docmind-ai-llm/issues)

- Open a new issue with: steps to reproduce, error logs, system info

## 📖 How to Cite

If you use DocMind AI in your research or work, please cite it as follows:

```bibtex
@software{melin_docmind_ai_2025,
  author = {Melin, Bjorn},
  title = {DocMind AI: Local LLM for AI-Powered Document Analysis},
  url = {https://github.com/BjornMelin/docmind-ai-llm},
  version = {0.1.0},
  year = {2025}
}
```

## 🙌 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository** and create a feature branch
2. **Set up development environment:**

   ```bash
   git clone https://github.com/your-username/docmind-ai-llm.git
   cd docmind-ai-llm
   uv sync --group dev
   ```

3. **Make your changes** following the established patterns
4. **Run tests and linting:**

   ```bash
   # Lint & format
   ruff format . && ruff check .
   uv run pylint -j 0 -sn --rcfile=pyproject.toml src tests/unit

   # Unit tests with coverage
   uv run pytest tests/unit -m unit --cov=src -q

   # Integration tests (offline; no coverage gating)
   uv run pytest tests/integration -m integration --no-cov -q
   ```

5. **Submit a pull request** with clear description of changes

### Development Guidelines

- Follow PEP 8 style guide (enforced by Ruff)

- Add type hints for all functions

- Include docstrings for public APIs

- Write tests for new functionality

- Update documentation as needed

#### 🧪 Tests and CI

We use a tiered test strategy and keep everything offline by default:

- Unit (fast, offline): mocks only; no network/GPU.
- Integration (offline): component interactions; router uses a session‑autouse MockLLM fixture in `tests/integration/conftest.py`, preventing any Ollama/remote calls.
- System/E2E (optional): heavier flows beyond the PR quality gates.

Quick local commands:

```bash
# Fast unit + integration sweep (offline)
uv run python scripts/run_tests.py --fast

# Full coverage gate (unit + integration)
uv run python scripts/run_tests.py --coverage

# Targeted module or pattern
uv run python scripts/run_tests.py tests/unit/persistence/test_snapshot_manager.py
```

CI pipeline mirrors this flow using `uv run python scripts/run_tests.py --fast` as a quick gate followed by `--coverage` for the full report. This keeps coverage thresholds stable while still surfacing integration regressions early. See ADR‑014 for quality gates/validation and ADR‑029 for the boundary‑first testing strategy.

See the [Developer Handbook](docs/developers/developer-handbook.md) for detailed guidelines. For an overview of the unit test layout and fixture strategy, see tests/README.md.

## 📃 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## 📡 Observability

DocMind AI configures OpenTelemetry tracing and metrics via `configure_observability` (see SPEC-012).

- Install the optional extras when you need OTLP exporters + `portalocker`: `uv sync --extra observability`.
- Default mode uses console exporters to remain offline-first.
- Set `DOCMIND_OBSERVABILITY__ENDPOINT` (or OTEL env vars) to forward spans and metrics to an OTLP collector.
- Core spans cover ingestion pipeline runs, snapshot promotion, GraphRAG exports, router selection, and Streamlit UI actions.
- Telemetry events (`router_selected`, `export_performed`, `lock_takeover`, `snapshot_stale_detected`) are persisted as JSONL for local audits.

```bash
uv run python -m src.telemetry.opentelemetry --dry-run
```

Use `tests/unit/telemetry/test_observability_config.py` as a reference for wiring custom exporters in extensions.

---

<div align="center">

Built with ❤️ by [Bjorn Melin](https://bjornmelin.io)

</div>
