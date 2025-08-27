# Developer Setup Guide

## Overview

This guide will get you productive with DocMind AI's unified architecture in under 30 minutes. The system has been refactored to achieve 95% complexity reduction while maintaining full functionality and ADR compliance.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Validation](#validation)
6. [Development Environment](#development-environment)
7. [Troubleshooting](#troubleshooting)

## Quick Start (5 minutes)

### Prerequisites

- Python 3.10+ (tested with 3.11, 3.12)
- CUDA-compatible GPU (RTX 4060+ recommended)
- Docker for Qdrant vector database
- Git

### Setup

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
python -c "from src.config import settings; print(f'✅ {settings.app_name} v{settings.app_version}')"

# Verify ADR compliance
python -c "
from src.config import settings
assert settings.embedding.model_name == 'BAAI/bge-m3', 'Should use BGE-M3'
assert settings.agents.decision_timeout <= 200, 'Agent timeout must be ≤200ms'
print('✅ ADR compliance verified')
"

# Test system health
python scripts/performance_validation.py
```

## Prerequisites

- **Python**: 3.11+ (tested with 3.11, 3.12)
- **uv**: For package management
- **Git**: Version control
- **Docker**: Optional, for Qdrant database
- **NVIDIA CUDA**: 12.8+ for GPU testing with RTX 4090

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm
```

### 2. Create Virtual Environment

```bash
uv venv --python 3.12
source .venv/bin/activate
uv sync
```

### 3. GPU Development Setup (RTX 4090 with vLLM FlashInfer)

```bash
# Install PyTorch 2.7.1 with CUDA 12.8
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install vLLM with FlashInfer
uv pip install "vllm[flashinfer]>=0.10.1" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install GPU extras
uv sync --extra gpu
```

### 4. Testing Dependencies

```bash
uv sync --extra test
```

### 5. spaCy Model Installation

DocMind AI uses spaCy for Named Entity Recognition (NER), linguistic analysis, knowledge graph entity extraction, and document preprocessing.

#### Quick Setup (Recommended)

```bash
# Install the small English model (recommended)
uv run python -m spacy download en_core_web_sm
```

#### Available Models

| Model | Size | Description | Use Case |
|-------|------|-------------|----------|
| `en_core_web_sm` | ~15MB | Small English model | General purpose, fast |
| `en_core_web_md` | ~50MB | Medium English model | Better accuracy |
| `en_core_web_lg` | ~560MB | Large English model | Highest accuracy |

#### Installation Commands

```bash
# Small model (default for DocMind AI)
uv run python -m spacy download en_core_web_sm

# Medium model (better accuracy)
uv run python -m spacy download en_core_web_md

# Large model (best accuracy, slower)
uv run python -m spacy download en_core_web_lg
```

## Configuration

### Understanding the Architecture

The system follows key architectural principles:

1. **Single Source of Truth**: All configuration through `from src.config import settings`
2. **Flattened Structure**: No unnecessary nested directories or complex hierarchies
3. **Library-First**: Use existing solutions (LlamaIndex, Pydantic) over custom code
4. **KISS Principle**: Simplicity over clever abstractions

### Configuration Patterns

**ALWAYS use this import pattern:**

```python
# ✅ CORRECT - Single import for everything
from src.config import settings

# Access any configuration
model_name = settings.vllm.model
embedding_model = settings.embedding.model_name
chunk_size = settings.processing.chunk_size
agent_timeout = settings.agents.decision_timeout
```

**NEVER do this:**

```python
# ❌ WRONG - Don't import nested models
from src.config.settings import VLLMConfig
from src.config import VLLMConfig

# ❌ WRONG - Don't access internal structure  
from src.config.settings import settings as internal_settings
```

### Environment Variables

Follow the **DOCMIND_** prefix with double underscore for nesting:

```bash
# Top-level settings
DOCMIND_DEBUG=true
DOCMIND_LOG_LEVEL=DEBUG

# Nested settings (use double underscore)
DOCMIND_VLLM__MODEL=custom-model
DOCMIND_VLLM__CONTEXT_WINDOW=65536
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.9

DOCMIND_EMBEDDING__MODEL_NAME=custom-embedding
DOCMIND_EMBEDDING__BATCH_SIZE_GPU=16

DOCMIND_AGENTS__DECISION_TIMEOUT=150
DOCMIND_AGENTS__MAX_RETRIES=3
```

Copy `.env.example` to `.env` and configure as needed.

## Validation

### spaCy Setup Verification

```bash
# Test model loading and basic functionality
uv run python -c "
from src.utils.core import ensure_spacy_model

# Load and test the model
nlp = ensure_spacy_model('en_core_web_sm')
if nlp:
    doc = nlp('Apple Inc. is a technology company.')
    print('✅ spaCy is working correctly')
    print(f'Entities found: {[(ent.text, ent.label_) for ent in doc.ents]}')
else:
    print('❌ spaCy model loading failed')
"
```

Expected output:
```text
✅ spaCy is working correctly
Entities found: [('Apple Inc.', 'ORG')]
```

### Complete Setup Verification

```bash
# 1. Check Python environment
python --version
which python

# 2. Check package installation
uv run python -c "import streamlit, spacy, torch; print('✅ All packages imported successfully')"

# 3. Check GPU support (if applicable)
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Run basic tests
pytest tests/ -v --tb=short
```

## Development Environment

### Running Locally

```bash
streamlit run src/app.py
```

### Testing

```bash
pytest
```

### Code Quality

```bash
ruff check .
ruff format .
```

### IDE Setup (VS Code recommended)

Install extensions:
- Python
- Pylance  
- Ruff

## Troubleshooting

### Common Issues

**Import Errors**

```bash
# Problem: "ModuleNotFoundError: No module named 'src'"
# Solution: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or run from project root
cd /path/to/docmind-ai-llm
python -c "from src.config import settings"
```

**Configuration Issues**

```bash
# Problem: Settings not loading
# Check environment variables
env | grep DOCMIND_ | sort

# Validate .env file syntax
grep -n "=" .env | grep -v "^#"

# Test configuration loading
python -c "
from src.config.settings import DocMindSettings
settings = DocMindSettings()
print('✅ Settings loaded')
print(f'App: {settings.app_name}')
print(f'Model: {settings.vllm.model}')
"
```

**Performance Issues**

```bash
# Check GPU availability
nvidia-smi

# Validate CUDA setup
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
"

# Test model loading
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
print('✅ BGE-M3 loaded')
"
```

### spaCy Troubleshooting

#### Model Not Found Error

If you encounter `OSError: [E050] Can't find model 'en_core_web_sm'`:

1. **Check Installation**: Verify the model is installed

   ```bash
   uv run python -m spacy info en_core_web_sm
   ```

2. **Reinstall Model**: Download and install again

   ```bash
   uv run python -m spacy download en_core_web_sm --force-reinstall
   ```

3. **Check Virtual Environment**: Ensure you're using the correct environment

   ```bash
   which python
   uv run python -c "import spacy; print(spacy.__file__)"
   ```

#### Download Failures

If model download fails:

1. **Check Network**: Ensure internet connectivity
2. **Use Alternative Installation**:

   ```bash
   # Alternative download method
   pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl
   ```

3. **Manual Installation**: Download wheel from GitHub releases

### Offline Installation

For environments without internet access:

#### Step 1: Download Model Archive

On a machine with internet access:

```bash
# Download the model wheel file
uv run python -m spacy download en_core_web_sm --user

# Find the downloaded model location
uv run python -m spacy info en_core_web_sm
```

#### Step 2: Locate Model Files

```bash
# In your virtual environment
find .venv -name "en_core_web_sm*" -type d
```

#### Step 3: Transfer for Offline Installation

1. Copy the entire model directory to your offline environment
2. Place it in the same relative path in your offline environment's `.venv`
3. Verify installation:

```bash
uv run python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('Model loaded successfully')"
```

## Getting Help

If you encounter issues during setup:

1. **spaCy issues**: Check the [spaCy documentation](https://spacy.io/usage/models)
2. **GPU issues**: See [GPU and Performance Guide](gpu-and-performance.md)
3. **General setup**: Review the DocMind AI logs for detailed error messages
4. **Contributing**: Follow [Development Guide](development-guide.md)
5. **Project issues**: Create an issue on GitHub with setup details

## Next Steps

After successful setup:

1. **Explore the architecture**: Review [Architecture Guide](architecture.md)
2. **Understand development practices**: Read [Development Guide](development-guide.md)
3. **Learn about performance**: Check [GPU and Performance](gpu-and-performance.md)
4. **Configure models**: See [Model Configuration](model-configuration.md)

---

**Welcome to the team!** 🎉

This architecture is designed to be **simple, powerful, and maintainable**. The unified configuration approach means you only need to remember one pattern: `from src.config import settings`.

When in doubt, follow the **KISS principle** and look for existing patterns in the codebase.