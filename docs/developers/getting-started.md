# Getting Started with DocMind AI Development

## Overview

This guide helps you set up a working development environment and run DocMind AI locally using the unified configuration and local-first architecture.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Environment Configuration](#environment-configuration)
5. [First Run Validation](#first-run-validation)
6. [Development Environment Setup](#development-environment-setup)
7. [Common Issues & Troubleshooting](#common-issues--troubleshooting)

## Quick Start (5 minutes)

### Prerequisites - Quick Start

- Python 3.10+ (tested with 3.11)
- CUDA-compatible GPU (RTX 4060+ recommended)
- Docker for Qdrant vector database
- Git

### Setup Commands

```bash
# 1. Clone and enter directory
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm

# 2. Install dependencies
uv sync

# 3. Start services
docker-compose up -d qdrant

# 4. Copy environment template
cp .env.example .env

# 5. Run application
streamlit run src/app.py
```

### Validate Setup

```bash
# Check configuration loads correctly
python3 -c "from src.config import settings; print(f'✅ {settings.app_name} v{settings.app_version}')"

# Verify 2026 Reference Alignment
python3 -c "
from src.config import settings
print(f'Model: {settings.model or settings.vllm.model}')
print(f'Embedding: {settings.embedding.model_name}')
print(f'Timeout: {settings.agents.decision_timeout}ms')
assert settings.embedding.model_name == 'BAAI/bge-m3', 'Should use BGE-M3'
"

# Test system health
uv run python scripts/performance_monitor.py --run-tests --check-regressions
```

## Prerequisites

### Required Software

- **Python**: 3.11+ (tested with 3.11)
- **uv**: For package management (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Git**: Version control
- **Docker**: Optional, for Qdrant database
- **NVIDIA CUDA**: 12.8+ for GPU testing with RTX 4090

### Hardware Requirements

#### Minimum Requirements

- **GPU**: NVIDIA GPU with CUDA Compute Capability 6.0+
- **VRAM**: 16GB minimum (RTX 4090 recommended)
- **System RAM**: 16GB minimum
- **Storage**: 50GB available space

#### Optimal Configuration (RTX 4090)

- **VRAM Usage**: 12-14GB with **FP8 KV Cache** optimization
- **System Memory**: 32GB+ recommended
- **CUDA Toolkit**: 12.8+ (required for PyTorch 2.7.0 support)
- **Drive**: NVMe SSD (required for low-latency model loading)
- **Attention**: FlashInfer backend recommended

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm
```

### 2. Install Dependencies

#### Standard Installation

```bash
# Install core dependencies
uv sync

# Install with GPU support (recommended)
uv sync --extra gpu --index https://download.pytorch.org/whl/cu128 --index-strategy=unsafe-best-match

# Install with test dependencies
uv sync --group test
```

Note: development and test tooling live in dependency groups (`[dependency-groups]`, PEP 735) and
are installed with `--group`. Optional runtime features are published as extras
(`project.optional-dependencies`, PEP 621) and are installed with `--extra`.

#### GPU-Optimized Installation (Recommended)

For optimal performance with RTX 4090:

```bash
# Phase 1: Install PyTorch 2.7.1 with CUDA 12.8
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Phase 2: Install vLLM and FlashInfer
uv pip install "vllm>=0.10.1,<0.11.0" \
    --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install "flashinfer-python>=0.5.3,<0.6.0"

# Phase 3: Install remaining dependencies
uv sync --extra gpu --index https://download.pytorch.org/whl/cu128 --index-strategy=unsafe-best-match
```

### 3. Start Required Services

```bash
# Start Qdrant vector database
docker-compose up -d qdrant

# Verify service is running
curl -f http://localhost:6333/health || echo "Qdrant not ready"
```

## Environment Configuration

### 1. Basic Configuration Setup

```bash
# Copy the example configuration
cp .env.example .env
```

### 2. Essential Environment Variables

Start from `.env.example` and override only what you need. For a full list of all 100+ variables, see the **[Configuration Reference](configuration.md)**.

```bash
# --- 1. Global & Core ---
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=INFO
DOCMIND_DATA_DIR=./data
DOCMIND_CACHE_DIR=./cache

# --- 2. LLM Backend (Bridge Vars) ---
DOCMIND_LLM_BACKEND=ollama
DOCMIND_OLLAMA_BASE_URL=http://localhost:11434
# DOCMIND_MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8  # Top-level Alias

# --- 3. Vector Storage (Nested Schema) ---
DOCMIND_DATABASE__QDRANT_URL=http://localhost:6333
DOCMIND_DATABASE__QDRANT_COLLECTION=docmind_docs

# --- 4. Security & Privacy ---
DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=false
DOCMIND_HASHING__HMAC_SECRET=your-32-character-secret-key-here
DOCMIND_PROCESSING__ENCRYPT_PAGE_IMAGES=false
```

> **Note:** If enabling encryption, generate a secure 256-bit AES key:
>
> ```bash
> python -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"
> ```

### 3. Key Configuration Concepts

- **Single Truth**: All configuration is managed via `src.config.settings`. See the **[Configuration Reference](configuration.md)** for exhaustive details.
- **Convention-Over-Configuration**: Double underscores map to nested Pydantic models (e.g., `DOCMIND_VLLM__MODEL`).
- **Path Relocation**: Bare filenames in DB paths are automatically moved under `DOCMIND_DATA_DIR` at startup.

## First Run Validation

### 1. Configuration Validation

```bash
# Test configuration loads correctly
python3 -c "
from src.config import settings
print(f'✅ App: {settings.app_name} v{settings.app_version}')
print(f'✅ Model: {settings.model or settings.vllm.model}')
print(f'✅ SQLite: {settings.database.sqlite_db_path}')
print(f'✅ Qdrant: {settings.database.qdrant_url}')
"
```

### 2. System Health Check

```bash
# Comprehensive validation
uv run python scripts/performance_monitor.py --run-tests --check-regressions --report

# Quick environment check (no full test suite)
uv run python scripts/performance_monitor.py --collection-only --report
```

### 3. Start the Application

```bash
# Start development server
streamlit run src/app.py

# Application will be available at http://localhost:8501
```

### 4. Verify Core Functionality

1. **Upload a document** (PDF, DOCX, TXT)
2. **Verify document processing** - document should be parsed and chunked
3. **Test query functionality** - ask a question about the document
4. **Check multi-agent coordination** - verify agents are working correctly

## Development Environment Setup

### 1. Essential Development Commands

```bash
# Format and lint code (run before commits)
ruff format . && ruff check . --fix

# Run tests
pytest tests/unit/ -v                    # Fast unit tests
pytest tests/integration/ -v             # Cross-component tests
python scripts/run_tests.py              # Full test suite

# Performance testing
uv run python scripts/performance_monitor.py --run-tests --report
```

### 2. IDE Setup

#### VS Code (Recommended)

1. Install Python extension
2. Install Pylance for type checking
3. Configure workspace settings:

   ```json
   {
     "python.defaultInterpreterPath": "./.venv/bin/python",
     "python.linting.ruffEnabled": true,
     "python.formatting.provider": "ruff"
   }
   ```

#### PyCharm

1. Set interpreter to project venv
2. Enable Ruff for code quality
3. Configure run configurations for streamlit

### 3. Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/your-feature-name

# 2. Make changes following coding standards

# 3. Test changes
pytest tests/ -v
ruff format . && ruff check . --fix

# 4. Commit changes
git add .
git commit -m "feat: your feature description"

# 5. Push and create PR
git push origin feature/your-feature-name
```

## Common Issues & Troubleshooting

### GPU Setup Issues

**Issue**: CUDA not detected

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Solution**:

```bash
# Reinstall PyTorch with correct CUDA version
uv pip install torch==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128
```

### vLLM Installation Issues

**Issue**: vLLM compilation fails

```bash
# Check environment
echo $CUDA_HOME
echo $PATH | grep cuda
```

**Solution**:

```bash
# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Reinstall vLLM + FlashInfer (runtime pins)
uv pip install --force-reinstall "vllm>=0.10.1,<0.11.0" --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --force-reinstall "flashinfer-python>=0.5.3,<0.6.0"
```

### Configuration Issues

**Issue**: Configuration not loading

```bash
# Debug configuration loading and prefix detection
python3 -c "
from src.config import settings
print('Config loaded successfully:', hasattr(settings, 'app_name'))
"
```

**Solution**:

1. Verify `.env` file exists and has correct format
2. Check `DOCMIND_` prefix on all variables
3. Validate nested delimiter usage (`__`)

### Service Connection Issues

**Issue**: Cannot connect to Qdrant

```bash
# Check Qdrant status
curl -f http://localhost:6333/health
docker-compose ps
```

**Solution**:

```bash
# Restart Qdrant service
docker-compose restart qdrant
docker-compose logs qdrant
```

### Qdrant Dimension Mismatch (SigLIP image collection)

**Issue**: Image indexing fails with an error like `expected dim: 768, got 1024`.

This usually means your existing Qdrant image collection was created with a
different vector size than the SigLIP embedder expects (768D).

**Solution**:

1. **Inspect the collection config**:

   ```bash
   curl http://localhost:6333/collections/docmind_images
   ```

2. **Recreate the image collection** if the `siglip` vector size is not `768`:

   ```bash
   curl -X DELETE http://localhost:6333/collections/docmind_images
   ```

   The collection is recreated automatically on the next ingestion run.

3. **Reindex images** by re-running ingestion for the affected documents.

See `docs/developers/adrs/ADR-058-final-multimodal-pipeline-and-persistence.md`
for the full multimodal indexing workflow.

### Performance Issues

**Issue**: Slow model loading or inference

```bash
# Check GPU memory usage
nvidia-smi

# Check model configuration
python -c "
from src.config import settings
print(f'Model: {settings.vllm.model}')
print(f'GPU memory: {settings.vllm.gpu_memory_utilization}')
print(f'Attention backend: {settings.vllm.attention_backend}')
"
```

**Solution**:

1. Verify FP8 quantization is enabled
2. Check FlashInfer backend configuration
3. Optimize GPU memory utilization settings

### Import Errors

**Issue**: Module not found errors

```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Verify installation
uv pip list | grep -E "(vllm|torch|streamlit)"
```

**Solution**:

```bash
# Reinstall dependencies
uv sync --force-reinstall

# Verify correct Python version
python --version  # Should be 3.11+
```

## Next Steps

Once you have the system running:

1. **Read the System Architecture Guide** - Understand how the multi-agent system works
2. **Review the Developer Handbook** - Learn implementation patterns and best practices
3. **Check the Configuration Guide** - Understand advanced configuration options
4. **Explore the Operations Guide** - Learn about production deployment and optimization

## Getting Help

1. **Setup Issues**: Review troubleshooting section above
2. **Architecture Questions**: See system-architecture.md
3. **Implementation Help**: See developer-handbook.md
4. **Performance Issues**: See operations-guide.md and configuration.md
5. **ADR References**: Check `../adrs/` for architectural decisions

---

**Welcome to DocMind AI development!**

This documentation represents a production-ready system with 95% ADR compliance and excellent code quality. The unified architecture ensures you only need to learn one pattern: `from src.config import settings`.

You're now ready to contribute to a world-class AI document analysis system.
