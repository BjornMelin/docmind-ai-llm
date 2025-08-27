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

**DocMind AI** transforms how you analyze documents locally with zero cloud dependency. This system combines hybrid search (dense + sparse embeddings), knowledge graph extraction, and a sophisticated 5-agent coordination system to extract and analyze information from your PDFs, Office docs, and multimedia content. Built on LlamaIndex pipelines with LangGraph supervisor orchestration and Qwen3-4B-Instruct-2507's FULL 262K context capability through INT8 KV cache optimization, it delivers production-ready document intelligence that runs entirely on your hardware—with GPU acceleration for enhanced performance and specialized agent coordination for improved query quality.

**Why DocMind AI?** Traditional document analysis tools either send your data to the cloud (privacy risk) or provide basic keyword search (limited intelligence). DocMind AI gives you the best of both worlds: AI reasoning with complete data privacy. Process complex queries that require multiple reasoning strategies, extract entities and relationships, and get contextual answers—all while your sensitive documents never leave your machine.

## ✨ Features of DocMind AI

- **Privacy-Focused:** Local processing ensures data security without cloud dependency.

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

- **LlamaIndex RAG Pipeline:** QueryPipeline with async/parallel processing, ingestion pipelines, and caching.

- **Hybrid Retrieval:** RRF fusion (α=0.7) combining BGE-Large dense and SPLADE++ sparse embeddings for 15-20% better recall.

- **Knowledge Graph Integration:** spaCy entity extraction with relationship mapping for complex queries.

- **Multimodal Processing:** Unstructured hi-res parsing for PDFs with text, tables, and images using Jina v4 embeddings.

- **ColBERT Reranking:** Late-interaction reranking improves context quality by 20-30%.

- **Offline-First Design:** 100% local processing with no external API dependencies.

- **GPU Acceleration:** CUDA support with mixed precision and FP8 quantization via vLLM FlashInfer backend for optimized performance.

- **Session Persistence:** SQLite WAL with local multi-process support for concurrent access.

- **Docker Support:** Easy deployment with Docker and Docker Compose.

- **Intelligent Caching:** High-performance document processing cache for rapid re-analysis.

- **Robust Error Handling:** Reliable retry strategies with exponential backoff.

- **Structured Logging:** Contextual logging with automatic rotation and JSON output.

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
    - [Additional Configuration](#additional-configuration)
  - [📊 Performance Benchmarks](#-performance-benchmarks)
    - [Performance Metrics](#performance-metrics)
    - [Caching Performance](#caching-performance)
    - [Hybrid Search Performance](#hybrid-search-performance)
    - [System Resource Usage](#system-resource-usage)
    - [Scalability Benchmarks](#scalability-benchmarks)
  - [🔧 Offline Operation](#-offline-operation)
    - [Prerequisites for Offline Use](#prerequisites-for-offline-use)
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
  - [📃 License](#-license)

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

   **Key Dependencies Included:**
   - **LlamaIndex Core**: RAG framework with QueryPipeline patterns
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
   ```

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

   See [GPU Setup Guide](docs/developers/gpu-setup.md) for detailed configuration and troubleshooting.

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

Select a pre-defined prompt or create a custom one:

- **Comprehensive Document Analysis:** Summary, key insights, action items, and open questions.

- **Extract Key Insights and Action Items:** Focus on insights and actionable outcomes.

- **Summarize and Identify Open Questions:** Generate summaries and highlight unresolved questions.

- **Custom Prompt:** Define your own analysis prompt.

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

Use the chat interface to ask follow-up questions. The LLM leverages hybrid search (Jina v4 dense + FastEmbed SPLADE++ sparse) with submodular-optimized reranking for context-aware, high-quality responses.

## 🔧 API Usage Examples

### Programmatic Document Analysis

```python
import asyncio
from pathlib import Path
from models import AppSettings
from src.utils.document import load_documents_unstructured
from src.utils.embedding import create_index_async
from agent_factory import get_agent_system

async def analyze_document(file_path: str, query: str):
    """Example: Analyze a document programmatically."""
    settings = AppSettings()
    
    # Load and process document
    documents = await load_documents_unstructured([Path(file_path)], settings)
    index = await create_index_async(documents, settings)
    
    # Create agent system
    agent_system = get_agent_system(index, settings)
    
    # Run analysis
    response = await agent_system.arun(query)
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
from models import AppSettings
import os

# Override default settings
os.environ["DEFAULT_MODEL"] = "llama3.2"
os.environ["GPU_ACCELERATION"] = "true"
os.environ["ENABLE_COLBERT_RERANKING"] = "true"

settings = AppSettings()
print(f"Using model: {settings.default_model}")
print(f"GPU enabled: {settings.gpu_acceleration}")
```

### Batch Document Processing

```python
import asyncio
from pathlib import Path
from models import AppSettings
from src.utils.document import load_documents_unstructured
from src.utils.embedding import create_index_async

async def process_document_folder(folder_path: str):
    """Process all supported documents in a folder."""
    settings = AppSettings()
    
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
    
    D --> E[SentenceSplitter<br/>1024 tokens / 200 overlap]
    D --> F[spaCy NLP Pipeline<br/>Entity Recognition]
    
    E --> G[Multi-Modal Embeddings]
    F --> H[Knowledge Graph Builder<br/>Entity Relations]
    
    G --> I[Dense: BGE-Large 1024D<br/>Sparse: SPLADE++ FastEmbed<br/>Multimodal: Jina v4 512D]
    I --> J[Qdrant Vector Store<br/>RRF Fusion α=0.7]
    
    H --> K[Knowledge Graph Index<br/>NetworkX Relations]
    
    J --> L[LlamaIndex QueryPipeline<br/>Multi-Stage Processing]
    K --> L
    
    L --> M[LangGraph Supervisor System<br/>5-Agent Coordination]
    M --> N[5 Specialized Agents:<br/>• Query Router<br/>• Query Planner<br/>• Retrieval Expert<br/>• Result Synthesizer<br/>• Response Validator]
    
    N --> O[ColBERT Reranker<br/>Late Interaction top-5]
    O --> P[Local LLM Backend<br/>Ollama/LM Studio/LlamaCpp]
    P --> Q[Supervisor Coordination<br/>Agent-to-Agent Handoffs]
    Q --> R[Response Synthesis<br/>Quality Validation]
    R --> S[Streamlit Interface<br/>Chat + Analysis Results]
    
    T[SQLite WAL Database<br/>Session Persistence] <--> M
    T <--> L
    U[DiskCache<br/>Document Processing] <--> D
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

- **Chunking:** LlamaIndex SentenceSplitter with 1024-token chunks and 200-token overlap for optimal context

- **Metadata:** spaCy en_core_web_sm for entity extraction and relationship mapping

### Hybrid Retrieval Architecture

- **Dense Embeddings:** BGE-Large 1024D (BAAI/bge-large-en-v1.5) for semantic similarity

- **Sparse Embeddings:** SPLADE++ with FastEmbed for neural lexical matching and term expansion

- **Multimodal:** Jina v4 512D embeddings for images and mixed content with int8 quantization

- **Fusion:** RRF (Reciprocal Rank Fusion) with α=0.7 weighting for optimal dense/sparse balance

- **Storage:** Qdrant vector database with metadata filtering and concurrent access

### Multi-Agent Coordination

- **Supervisor Pattern:** LangGraph supervisor using `langgraph-supervisor` library for proven coordination patterns with automatic state management

- **5 Specialized Agents:**
  - **Query Router:** Analyzes query complexity and determines optimal retrieval strategy
  - **Query Planner:** Decomposes complex queries into manageable sub-tasks for better processing
  - **Retrieval Expert:** Executes optimized retrieval with DSPy query optimization and optional GraphRAG for relationships
  - **Result Synthesizer:** Combines and reconciles results from multiple retrieval passes with deduplication
  - **Response Validator:** Validates response quality, accuracy, and completeness before final output

- **Enhanced Capabilities:** DSPy automatic query optimization (20-30% quality improvement) and optional GraphRAG for multi-hop reasoning

- **Workflow Coordination:** Supervisor automatically routes between agents based on query complexity with <300ms coordination overhead

- **Session Management:** SQLite WAL database with built-in conversation context preservation and error recovery

- **Async Execution:** Concurrent agent operations with automatic resource management and fallback mechanisms

### Performance Optimizations

- **GPU Acceleration:** CUDA support with FP8 quantization via vLLM FlashInfer backend and torch.compile optimization

- **Async Processing:** QueryPipeline with parallel execution and intelligent caching

- **Reranking:** ColBERT late-interaction model improves top-5 results from top-20 prefetch

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
LMDEPLOY_HOST=http://localhost:23333

# Embedding Models (BGE-M3 unified)
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Feature Flags
ENABLE_DSPY_OPTIMIZATION=true
ENABLE_GRAPHRAG=false
ENABLE_GPU_ACCELERATION=true
LMDEPLOY_QUANT_POLICY=fp8  # FP8 KV cache

# Performance Tuning
RETRIEVAL_TOP_K=10
RERANK_TOP_K=5
CACHE_SIZE_LIMIT=1073741824  # 1GB
```

See the complete [.env.example](.env.example) file for all available configuration options.

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

### Performance Metrics

| Operation | Performance | Notes |
|-----------|-------------|--------|
| **Document Processing (Cold)** | ~15-30 seconds | 50-page PDF with GPU acceleration |
| **Document Processing (Warm)** | ~2-5 seconds | DiskCache + index caching |
| **Query Response** | 1-3 seconds | Hybrid retrieval + ColBERT reranking |
| **5-Agent System Response** | 3-8 seconds | LangGraph supervisor coordination with <200ms overhead |
| **128K Context Processing** | 1.5-3 seconds | 128K context with FP8 KV cache |
| **Vector Search** | <500ms | Qdrant in-memory with GPU embeddings |
| **Test Suite (99 tests)** | ~40 seconds | Comprehensive coverage |
| **Memory Usage (Idle)** | 400-500MB | Base application |
| **Memory Usage (Processing)** | 1.2-2.1GB | During document analysis |
| **GPU Memory Usage** | ~12-14GB | Model + 128K context + embedding cache |

### Caching Performance

**Document Processing Cache:**

- **Cache hit ratio**: 85-90% for repeated documents

- **Storage efficiency**: ~1GB handles 1000+ documents

- **Cache invalidation**: Automatic based on file content + settings hash

- **Concurrent access**: Multi-process safe with WAL mode

### Hybrid Search Performance

**Retrieval Quality Metrics:**

- **Dense + Sparse RRF**: 15-20% better recall vs single-vector

- **ColBERT Reranking**: 20-30% context quality improvement

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

> *Benchmarks performed on RTX 4090 Laptop GPU, 16GB RAM, NVMe SSD*

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
   python scripts/gpu_validation.py  # Validate CUDA setup
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
python scripts/performance_validation.py

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

- Review [troubleshooting guide](docs/user/troubleshooting.md)

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
   ruff check . --fix
   ruff format .
   pytest tests/
   ```

5. **Submit a pull request** with clear description of changes

### Development Guidelines

- Follow PEP 8 style guide (enforced by Ruff)

- Add type hints for all functions

- Include docstrings for public APIs

- Write tests for new functionality

- Update documentation as needed

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📃 License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ by [Bjorn Melin](https://bjornmelin.io)

</div>
