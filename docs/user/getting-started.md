# Getting Started with DocMind AI

DocMind AI is a local-first AI document analysis system that runs entirely on your machine with zero cloud dependencies. This guide will get you from installation to analyzing your first document in under 15 minutes.

## What You'll Accomplish

By the end of this guide, you'll have:

- ✅ DocMind AI running on your system
- ✅ Your first documents uploaded and analyzed  
- ✅ Understanding of the 5-agent coordination system
- ✅ Confidence to explore advanced capabilities

## Hardware Requirements

### Minimum Setup (CPU-Only)

- **CPU**: Any modern processor
- **RAM**: 8GB system memory
- **Storage**: 20GB free space for models
- **OS**: Linux, macOS, or Windows 10/11

### Recommended Setup  

- **CPU**: Modern multi-core processor
- **RAM**: 16GB system memory
- **GPU**: RTX 4060 (12GB VRAM) or better
- **Storage**: 50GB SSD space

### Optimal Performance

- **CPU**: High-end desktop processor
- **RAM**: 32GB+ system memory
- **GPU**: RTX 4090 (16GB+ VRAM)
- **Storage**: 100GB+ NVMe SSD

> **Hardware Check**: Run `nvidia-smi` (if you have NVIDIA) and `uv run python --version` to verify GPU and Python availability.

## Step 1: Installation (5 minutes)

### Clone and Install Dependencies

```bash
# Get the code
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm

# Install Python dependencies
uv sync

# (Recommended) Install a spaCy language model for NLP enrichment (entities/sentences).
# The app will run without a model (blank fallback), but entity extraction will be limited.
uv run python -m spacy download en_core_web_sm

# Optional: disable enrichment (CPU-only and fastest ingestion)
# DOCMIND_SPACY__ENABLED=false

# Optional: device selection (cpu|cuda|apple|auto)
# DOCMIND_SPACY__DEVICE=auto
```

### Install Ollama (Local AI Backend)

**Linux/macOS:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:** Download from [ollama.com](https://ollama.com/download)

**Start Ollama and download the recommended model:**

```bash
# Start the Ollama service
ollama serve

# In a new terminal, download the model (this may take 10-15 minutes)
ollama pull qwen3-4b-instruct-2507
```

### GPU Support (Optional but Recommended)

**For NVIDIA GPUs with 12GB+ VRAM:**

```bash
# Verify CUDA installation
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA 12.8+

# Install GPU-optimized dependencies
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install "vllm>=0.10.1,<0.11.0" \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install "flashinfer-python>=0.5.3,<0.6.0"
uv sync --extra gpu --index https://download.pytorch.org/whl/cu128 --index-strategy=unsafe-best-match
```

**Verify GPU setup:**

```bash
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

> **spaCy GPU note**: spaCy acceleration is configured via `SPACY_DEVICE=auto|cuda` and
> installed via `uv sync --extra gpu`. See `docs/developers/gpu-setup.md`.

## Step 2: Configuration (2 minutes)

### Basic Configuration

```bash
# Create configuration file
cp .env.example .env

# For CPU-only setups
echo 'DOCMIND_ENABLE_GPU_ACCELERATION=false' >> .env
echo 'DOCMIND_CONTEXT_WINDOW_SIZE=32768' >> .env

# For GPU setups (RTX 4060/4070)
echo 'DOCMIND_ENABLE_GPU_ACCELERATION=true' >> .env
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.85' >> .env
echo 'DOCMIND_CONTEXT_WINDOW_SIZE=65536' >> .env

# For high-end GPUs (RTX 4090)
echo 'DOCMIND_ENABLE_GPU_ACCELERATION=true' >> .env
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.85' >> .env
echo 'DOCMIND_CONTEXT_WINDOW_SIZE=131072' >> .env
echo 'VLLM_ATTENTION_BACKEND=FLASHINFER' >> .env
```

### Enable DSPy Optimization (Optional)

DSPy can refine queries before retrieval to improve result quality.

1) Install DSPy:

    ```bash
    uv pip install dspy-ai
    ```

2) Enable the feature flag in your `.env`:

    ```bash
    echo 'DOCMIND_ENABLE_DSPY_OPTIMIZATION=true' >> .env
    ```

#### Optional tuning

```bash
echo 'DOCMIND_DSPY_OPTIMIZATION_ITERATIONS=10' >> .env
echo 'DOCMIND_DSPY_OPTIMIZATION_SAMPLES=20' >> .env
echo 'DOCMIND_DSPY_MAX_RETRIES=3' >> .env
echo 'DOCMIND_DSPY_TEMPERATURE=0.1' >> .env
echo 'DOCMIND_DSPY_METRIC_THRESHOLD=0.8' >> .env
echo 'DOCMIND_ENABLE_DSPY_BOOTSTRAPPING=true' >> .env
```

#### Notes

- DSPy runs in the agents layer and augments retrieval by refining the query; retrieval remains library‑first (server‑side hybrid via Qdrant + reranking).
- If DSPy isn’t installed or the flag is false, the system falls back to standard retrieval automatically.

### Knowledge Graph (Optional)

- When enabled and built, DocMind adds a `knowledge_graph` tool to the Router. Selection prefers `PydanticSingleSelector` then `LLMSingleSelector` and falls back to vector/hybrid when graph is absent or unhealthy.
- Disable GraphRAG with:

```bash
echo 'DOCMIND_ENABLE_GRAPHRAG=false' >> .env
```

### Snapshots & Staleness (Quick Note)

- The app persists vector and graph indices with a manifest (schema/persist versions, versions map, hashes). Chat auto‑loads the latest non‑stale snapshot and shows a staleness badge when current corpus/config differ; rebuild from the Documents page.

### Enable Multi-Agent System

```bash
# Enable the 5-agent coordination system
echo 'ENABLE_MULTI_AGENT=true' >> .env
echo 'AGENT_TIMEOUT_SECONDS=30' >> .env
```

## Step 3: Launch DocMind AI (1 minute)

```bash
# Start the application
uv run streamlit run src/app.py
```

**You should see:**

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Step 4: Verify System Status (1 minute)

Check the sidebar for system indicators:

- ✅ **GPU Status**: Shows your GPU model (or "CPU Only")
- ✅ **Model Status**: Shows "Model: Ready"
- ✅ **Multi-Agent**: Shows "5 Agents Active"
- ✅ **Ollama**: Shows "Connected" with model name

If you see any red error indicators, check the [Common Issues](#common-issues) section below.

## Step 5: Analyze Your First Document (5 minutes)

### Upload a Document

1. Click **"Upload Documents"** in the sidebar
2. Select a PDF, Word doc, or text file (start with something under 10 pages)
3. Wait for the processing indicator to complete
4. Verify the document appears in the document list

### Ask Your First Question

Try this example query:

```text
What are the main topics and key insights in this document?
```

**Watch the multi-agent system work:**

- 🎯 **Query Router** analyzes your question  
- 📋 **Query Planner** breaks down the task
- 🔍 **Retrieval Expert** searches your documents
- 🔧 **Result Synthesizer** combines findings
- ✅ **Response Validator** ensures quality

### Interpret Your Results

Your response will include:

- **Main Answer**: Direct response to your question
- **Sources**: Referenced document sections  
- **Confidence Score**: Quality indicator (aim for >0.8)
- **Processing Time**: Should be 1-3 seconds

## Understanding DocMind AI

### The 5-Agent System

DocMind AI uses specialized AI agents that coordinate automatically:

1. **Query Router**: Analyzes query complexity and routes to appropriate agents
2. **Query Planner**: Breaks complex questions into manageable sub-tasks  
3. **Retrieval Expert**: Finds relevant content using hybrid search (semantic + keyword)
4. **Result Synthesizer**: Combines information from multiple sources
5. **Response Validator**: Ensures response quality and accuracy

### Document Processing

- **Supported Formats**: PDF, DOCX, TXT, XLSX, CSV, JSON, XML, MD, RTF, MSG, PPTX, ODT, EPUB, and code files
- **Processing**: Unstructured.io hi-res parsing extracts text, tables, and images
- **NLP enrichment (optional)**: spaCy sentence segmentation + entity extraction during ingestion
- **Search**: BGE-M3 unified embeddings with hybrid dense + sparse retrieval

### Privacy and Local Operation

- **100% Local**: No data ever leaves your machine
- **Offline Capable**: Works without internet after initial setup
- **No API Keys**: Uses local models through Ollama
- **Secure**: All processing happens on your hardware

## Next Steps

### Try Different Query Types

**Simple Information Lookup:**

```text
What is the budget mentioned in the financial report?
Who are the key stakeholders identified?
```

**Comparative Analysis:**

```text
Compare the Q1 and Q2 performance metrics.
How do the proposed solutions differ in approach?
```

**Multi-Step Reasoning:**

```text
What risks are identified, how severe are they, and what mitigation strategies are proposed?
```

### Explore Advanced Features

- **Large Context**: Process entire documents (up to 128K tokens) without chunking
- **Multiple Documents**: Upload related files for cross-document analysis
- **Conversation History**: Build on previous questions for deeper analysis
- **Export Results**: Save responses as JSON or Markdown

## Common Issues

### DocMind AI Won't Start

**Problem**: Application crashes or won't launch

**Quick Fix:**

```bash
# Check Python version (should be 3.10-3.11)
python --version

# Reinstall dependencies
uv sync

# Check for port conflicts
lsof -i :8501  # Kill any process using port 8501
```

### GPU Not Detected

**Problem**: Shows "CPU Only" despite having GPU

**Quick Fix:**

```bash
# Verify GPU detection
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"

# Force GPU in configuration
echo 'DOCMIND_DEVICE=cuda' >> .env
echo 'DOCMIND_ENABLE_GPU_ACCELERATION=true' >> .env
```

### Out of Memory Errors

**Problem**: CUDA out of memory or system crashes

**Quick Fix:**

```bash
# Reduce GPU memory usage
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.75' >> .env
echo 'DOCMIND_CONTEXT_WINDOW_SIZE=32768' >> .env

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

### Slow Performance

**Problem**: Queries take >10 seconds

**Quick Fix:**

```bash
# Enable performance optimizations
echo 'VLLM_ATTENTION_BACKEND=FLASHINFER' >> .env
echo 'VLLM_USE_CUDNN_PREFILL=1' >> .env

# Check system resources
nvidia-smi  # GPU utilization should be <90%
htop        # Monitor CPU and RAM usage
```

### Ollama Connection Failed

**Problem**: Can't connect to Ollama service

**Quick Fix:**

```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start it
ollama serve

# Verify model is installed
ollama list
ollama pull qwen3-4b-instruct-2507  # If missing
```

### Document Upload Fails

**Problem**: Documents won't upload or process

**Quick Fix:**

```bash
# Check file format support
echo "Supported: PDF, DOCX, TXT, XLSX, CSV, JSON, XML, MD, RTF, MSG, PPTX, ODT, EPUB"

# Check file size (should be <100MB)
ls -lh your_file.pdf

# Try with a simple text file first
echo "Test document content" > test.txt
```

## Configuration Reference

### Performance Profiles

**Speed Optimized (Fast responses):**

```bash
DOCMIND_CONTEXT_WINDOW_SIZE=32768
AGENT_TIMEOUT_SECONDS=15
MIN_VALIDATION_SCORE=0.6
RETRIEVAL_TOP_K=10
```

**Balanced (Default):**

```bash
DOCMIND_CONTEXT_WINDOW_SIZE=65536
AGENT_TIMEOUT_SECONDS=30
MIN_VALIDATION_SCORE=0.7
RETRIEVAL_TOP_K=20
```

**Quality Optimized (Best accuracy):**

```bash
DOCMIND_CONTEXT_WINDOW_SIZE=131072
AGENT_TIMEOUT_SECONDS=60
MIN_VALIDATION_SCORE=0.85
RETRIEVAL_TOP_K=30
ENABLE_DSPY_OPTIMIZATION=true
```

### Hardware-Specific Settings

**RTX 4060 (12GB VRAM):**

```bash
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_CONTEXT_WINDOW_SIZE=65536
DOCMIND_MAX_DOCUMENT_SIZE_MB=50
```

**RTX 4080 (16GB VRAM):**

```bash
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_MAX_DOCUMENT_SIZE_MB=100
VLLM_ATTENTION_BACKEND=FLASHINFER
```

**RTX 4090 (24GB VRAM):**

```bash
DOCMIND_GPU_MEMORY_UTILIZATION=0.90
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_MAX_DOCUMENT_SIZE_MB=200
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
```

## Getting Help

### Built-in Diagnostics

```bash
# Check system health
python scripts/system_validation.py

# Run performance tests
uv run python scripts/performance_monitor.py --run-tests --check-regressions

# Analyze logs
tail -f logs/app.log | grep "ERROR\|WARNING"
```

### Documentation Resources

- **Configuration Options**: [configuration.md](configuration.md)
- **Common Problems & FAQ**: [troubleshooting-faq.md](troubleshooting-faq.md)

### Support Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Community Q&A and best practices
- **Log Analysis**: Check `logs/app.log` for detailed error information

### When Reporting Issues

Include in your report:

- System specifications (GPU, RAM, OS)
- Complete error messages and logs
- Steps to reproduce the problem
- Configuration settings (`.env` file, remove any sensitive data)

---

**Success!** You now have DocMind AI running locally with complete privacy and powerful AI-driven document analysis capabilities. Your sensitive documents never leave your machine, and you have access to advanced multi-agent coordination for complex queries.

Ready to explore more? Check out [configuration.md](configuration.md) for advanced settings and [troubleshooting-faq.md](troubleshooting-faq.md) for detailed problem resolution.
