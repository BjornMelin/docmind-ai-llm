# Getting Started with DocMind AI

## System Requirements

### Hardware Requirements
- **GPU**: RTX 4090 (24GB) or RTX 4090 Laptop (16GB VRAM minimum)
- **CUDA**: Version 12.8 or higher
- **Driver**: NVIDIA Driver 550.54.14+
- **RAM**: 32GB system memory recommended
- **Storage**: 50GB+ free space for models and vector database

### Software Requirements
- **Python**: 3.10-3.12 (3.13 not yet supported)
- **Operating System**: Linux (Ubuntu 20.04+) or Windows 10/11 with WSL2
- **uv**: Python package manager (recommended)

For detailed GPU setup, see [gpu-requirements.md](gpu-requirements.md).

## Quick Installation

### 1. Clone Repository
```bash
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm
```

### 2. Install Dependencies
```bash
# Install with GPU support (recommended)
uv sync --extra gpu --index-strategy=unsafe-best-match

# Or install basic dependencies
uv sync
```

### 3. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (set paths and options)
nano .env
```

### 4. Verify GPU Setup
```bash
# Test CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run GPU validation script
uv run python scripts/gpu_validation.py
```

## First-Time Setup

### Model Configuration

DocMind AI uses the Qwen3-4B-Instruct-2507-FP8 model with 128K context:

```bash
# Environment variables for optimal performance
export VLLM_ATTENTION_BACKEND=FLASHINFER
export VLLM_USE_CUDNN_PREFILL=1
export DOCMIND_MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507-FP8"
```

### Vector Database Setup

```bash
# Start Qdrant vector database
docker run -p 6333:6333 qdrant/qdrant:latest

# Or install locally
curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
./qdrant
```

## Running DocMind AI

### Start the Application

```bash
# Run with Streamlit interface
uv run streamlit run src/app.py

# Or run with custom configuration
DOCMIND_GPU_MEMORY_UTILIZATION=0.85 uv run streamlit run src/app.py
```

### Access the Interface

Open your browser and navigate to:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501 (if configured for network access)

## Initial Setup Steps

### 1. System Validation
- Check that GPU is detected and properly configured
- Verify VRAM usage is within target range (12-14GB)
- Confirm model loading completes successfully

### 2. Upload Documents
- Click "Upload Documents" in the sidebar
- Select PDF, DOCX, or TXT files
- Wait for document processing to complete
- Verify documents appear in the document list

### 3. Test Multi-Agent System
- Enter a test query: "Summarize the key points from the uploaded documents"
- Observe the 5-agent coordination process:
  - Query Router: Determines search strategy
  - Query Planner: Breaks down complex queries
  - Retrieval Expert: Finds relevant content
  - Result Synthesizer: Combines information
  - Response Validator: Ensures quality

### 4. Monitor Performance
- Check the performance metrics in the sidebar
- Verify decode speed (target: 100-160 tokens/second)
- Monitor VRAM usage (target: <16GB)
- Observe agent coordination efficiency

## Configuration Options

### Performance Settings

```bash
# .env configuration for optimal performance
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_MAX_CONTEXT_LENGTH=131072
DOCMIND_ENABLE_FP8_KV_CACHE=true
DOCMIND_MAX_PARALLEL_AGENTS=3
```

### Agent Configuration

```bash
# Agent-specific settings
QUERY_ROUTER_TEMPERATURE=0.1
RETRIEVAL_EXPERT_TOP_K=20
RESPONSE_VALIDATOR_THRESHOLD=0.85
```

### Advanced Options

```bash
# Enable experimental features
DOCMIND_ENABLE_GRAPHRAG=false
DOCMIND_ENABLE_DSPY_OPTIMIZATION=true
DOCMIND_PARALLEL_TOOL_EXECUTION=true
```

## Troubleshooting Quick Start

### Common Issues

**GPU Not Detected**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"
```

**Out of Memory Errors**:
```bash
# Reduce memory utilization
export DOCMIND_GPU_MEMORY_UTILIZATION=0.75

# Enable FP8 optimization
export DOCMIND_ENABLE_FP8_KV_CACHE=true
```

**Slow Performance**:
```bash
# Verify FlashInfer backend
export VLLM_ATTENTION_BACKEND=FLASHINFER

# Enable cuDNN prefill
export VLLM_USE_CUDNN_PREFILL=1
```

**Model Loading Failures**:
```bash
# Check model path
echo $DOCMIND_MODEL_PATH

# Verify model availability
python -c "from transformers import AutoTokenizer; print('Model accessible')"
```

## Next Steps

After successful setup:

1. **Read the User Guide**: [usage-guide.md](usage-guide.md) for detailed feature documentation
2. **Explore Multi-Agent Features**: [multi-agent-coordination-guide.md](multi-agent-coordination-guide.md) for advanced usage
3. **Performance Tuning**: [../developers/multi-agent-performance-tuning.md](../developers/multi-agent-performance-tuning.md) for optimization
4. **Troubleshooting**: [troubleshooting.md](troubleshooting.md) for detailed problem resolution

## Support and Community

- **Documentation**: Complete documentation in `/docs/`
- **Issues**: Report bugs and request features on GitHub
- **Performance**: Monitor system performance through built-in metrics
- **Updates**: Check for model and system updates regularly

For advanced configuration and development setup, see [../developers/setup.md](../developers/setup.md).
