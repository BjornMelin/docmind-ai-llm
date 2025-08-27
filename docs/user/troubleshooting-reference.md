# Troubleshooting Reference

Comprehensive problem resolution guide organized by issue type and severity. Find solutions quickly with our systematic troubleshooting approach.

## Quick Problem Resolver

### 🚨 Critical Issues (System Won't Start)

#### DocMind AI Won't Launch

**Symptoms:**

- Application crashes on startup
- "Module not found" errors
- Server won't start

**Immediate Actions:**

```bash
# 1. Check Python environment
python --version  # Should be 3.10-3.12
which python

# 2. Verify installation
uv sync --extra gpu
uv run python -c "import streamlit; print('Streamlit OK')"

# 3. Check for conflicting processes
lsof -i :8501  # Kill any process using port 8501
```

**Common Solutions:**

```bash
# Reinstall dependencies
rm -rf .venv
uv sync --extra gpu

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Reset configuration
cp .env.example .env
```

#### GPU Not Detected or Initialized

**Symptoms:**

- "GPU: CPU Only" in sidebar
- CUDA errors on startup
- Poor performance

**Immediate Actions:**

```bash
# 1. Verify GPU hardware
nvidia-smi
# Should show: GPU name, VRAM, driver version

# 2. Check CUDA installation  
nvcc --version
# Should show: CUDA 12.8+

# 3. Test PyTorch CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Common Solutions:**

```bash
# Install/update NVIDIA driver
sudo ubuntu-drivers autoinstall
sudo reboot

# Install CUDA 12.8
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_550.54.14_linux.run
sudo sh cuda_12.8.0_550.54.14_linux.run

# Reinstall PyTorch with CUDA
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --extra-index-url https://download.pytorch.org/whl/cu128
```

### ⚠️ Performance Issues (System Runs Slowly)

#### Queries Taking >10 Seconds

**Symptoms:**

- Very slow response times
- High GPU/CPU usage
- System appears frozen

**Immediate Actions:**

```bash
# 1. Check system resources
nvidia-smi  # GPU utilization should be <90%
htop        # CPU and memory usage

# 2. Enable performance optimizations
echo 'VLLM_ATTENTION_BACKEND=FLASHINFER' >> .env
echo 'VLLM_USE_CUDNN_PREFILL=1' >> .env

# 3. Reduce memory pressure
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.75' >> .env
```

#### Out of Memory Errors

**Symptoms:**

- "CUDA out of memory" errors
- System crashes during processing
- GPU memory warnings

**Immediate Actions:**

```bash
# 1. Clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"

# 2. Reduce memory usage
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.75' >> .env
echo 'DOCMIND_MAX_CONTEXT_LENGTH=65536' >> .env

# 3. Enable FP8 optimization
echo 'DOCMIND_ENABLE_FP8_KV_CACHE=true' >> .env
echo 'VLLM_KV_CACHE_DTYPE=fp8_e5m2' >> .env
```

### 🔧 Functionality Issues (Features Don't Work)

#### Multi-Agent System Not Working

**Symptoms:**

- Only getting basic responses
- No agent coordination visible
- "Fallback mode" messages

**Immediate Actions:**

```bash
# 1. Enable multi-agent system
echo 'ENABLE_MULTI_AGENT=true' >> .env
echo 'AGENT_TIMEOUT_SECONDS=30' >> .env

# 2. Check agent initialization
python -c "from src.agents import MultiAgentCoordinator; print('Agents OK')"

# 3. Verify LangGraph dependencies
uv pip install langgraph
```

#### Documents Won't Upload/Process

**Symptoms:**

- Upload fails silently
- "Unsupported file type" errors
- Processing never completes

**Immediate Actions:**

```bash
# 1. Check file format support
echo "Supported: PDF, DOCX, TXT, XLSX, CSV, JSON, XML, MD, RTF, MSG, PPTX, ODT, EPUB"

# 2. Verify file size limits
ls -lh your_file.pdf  # Should be <100MB

# 3. Test with simple file
echo "Test document content" > test.txt
# Try uploading test.txt
```

## Systematic Troubleshooting

### Step-by-Step Diagnosis

#### Level 1: Basic System Check

**Environment Validation:**

```bash
# Run comprehensive system check
python scripts/system_validation.py

# Expected green checkmarks:
# ✅ Python 3.10+ detected
# ✅ GPU hardware available  
# ✅ CUDA installation valid
# ✅ Dependencies installed
# ✅ Configuration files present
```

**Configuration Validation:**

```bash
# Check critical settings
python -c "
from src.config import settings
print(f'GPU Enabled: {settings.enable_gpu_acceleration}')
print(f'Multi-Agent: {settings.enable_multi_agent}')
print(f'Model: {settings.default_model}')
"
```

#### Level 2: Component Testing

**GPU Performance Test:**

```bash
# Test GPU functionality
python -c "
import torch
if torch.cuda.is_available():
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x)
    print(f'GPU Test: PASSED - {torch.cuda.get_device_name(0)}')
else:
    print('GPU Test: FAILED - CUDA not available')
"
```

**Model Loading Test:**

```bash
# Test model accessibility
python -c "
try:
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-4B-Instruct-2507-FP8')
    print('Model Test: PASSED - Model accessible')
except Exception as e:
    print(f'Model Test: FAILED - {str(e)}')
"
```

**Agent System Test:**

```bash
# Test agent coordination
python scripts/test_agents.py --quick
# Should show all 5 agents responding within timeout
```

#### Level 3: Performance Benchmarking

**Complete Performance Validation:**

```bash
# Run full benchmark suite
python scripts/performance_validation.py --comprehensive

# Expected results:
# Decode Speed: 100-160 tok/s
# Prefill Speed: 800-1300 tok/s  
# VRAM Usage: 12-14GB
# Agent Coordination: <200ms overhead
```

## Issue-Specific Solutions

### Installation and Dependencies

#### Python Environment Issues

**Problem:** Wrong Python version or virtual environment conflicts

```bash
# Solution: Clean Python environment
pyenv install 3.11.5  # Install specific Python version
pyenv global 3.11.5   # Set as default

# Or use uv for environment management
uv python install 3.11
uv venv --python 3.11
source .venv/bin/activate
```

**Problem:** Package conflicts or missing dependencies

```bash
# Solution: Complete dependency refresh
uv sync --extra gpu --index-strategy=unsafe-best-match

# If issues persist, clean installation
rm -rf .venv uv.lock
uv sync --extra gpu
```

#### CUDA and GPU Setup

**Problem:** CUDA version mismatch

```bash
# Check current CUDA version
nvcc --version
nvidia-smi | grep "CUDA Version"

# Install compatible CUDA (12.8+ required)
# Ubuntu/Debian:
sudo apt install cuda-toolkit-12-8

# CentOS/RHEL:
sudo yum install cuda-toolkit-12-8
```

**Problem:** GPU drivers outdated or incompatible

```bash
# Update NVIDIA drivers (Ubuntu)
sudo ubuntu-drivers devices
sudo ubuntu-drivers autoinstall
sudo reboot

# Manual driver installation
sudo apt purge nvidia* libnvidia*
sudo apt install nvidia-driver-550
sudo reboot
```

**Problem:** vLLM installation issues

```bash
# Reinstall vLLM with FlashInfer support
uv pip uninstall vllm
uv pip install "vllm[flashinfer]>=0.10.1" --extra-index-url https://download.pytorch.org/whl/cu128

# Verify FlashInfer availability
python -c "import vllm; print('FlashInfer:', 'Available' if 'flashinfer' in vllm.__version__ else 'Not Available')"
```

### Multi-Agent System Troubleshooting

#### Agent Timeout and Coordination Issues

**Problem:** Frequent agent timeouts

```bash
# Increase timeout values
echo 'AGENT_TIMEOUT_SECONDS=60' >> .env
echo 'AGENT_DECISION_TIMEOUT=600' >> .env
echo 'MAX_AGENT_RETRIES=3' >> .env

# Monitor agent performance
tail -f logs/app.log | grep "agent_timeout\|coordination"
```

**Problem:** Agents not communicating properly

```bash
# Enable debug logging for agents
echo 'DOCMIND_LOG_LEVEL=DEBUG' >> .env
echo 'LANGGRAPH_DEBUG=1' >> .env

# Test agent connectivity
python -c "
from src.agents.coordinator import test_agent_coordination
results = test_agent_coordination()
print('Agent connectivity:', 'OK' if results['success'] else 'FAILED')
"
```

**Problem:** Poor response quality from agents

```bash
# Adjust quality thresholds
echo 'MIN_VALIDATION_SCORE=0.6' >> .env  # Lower for testing
echo 'ENABLE_DSPY_OPTIMIZATION=true' >> .env
echo 'ENABLE_CROSS_VALIDATION=true' >> .env

# Monitor validation scores
grep "validation_score" logs/app.log | tail -10
```

#### Fallback and Recovery Issues

**Problem:** System frequently falls back to basic RAG

```bash
# Check fallback triggers
grep "fallback_triggered" logs/app.log

# Adjust fallback settings
echo 'FALLBACK_THRESHOLD_MS=5000' >> .env  # Increase threshold
echo 'ENABLE_FALLBACK_RAG=true' >> .env
echo 'FALLBACK_STRATEGY=basic_rag' >> .env
```

**Problem:** Context preservation not working

```bash
# Enable context preservation
echo 'CONTEXT_PRESERVATION=true' >> .env
echo 'MAX_CONTEXT_TOKENS=65000' >> .env

# Test context continuity
python scripts/test_context_preservation.py
```

### Performance Optimization

#### Memory Management

**Problem:** GPU memory leaks

```bash
# Enable automatic cleanup
echo 'DOCMIND_AUTO_GPU_CLEANUP=true' >> .env
echo 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' >> .env

# Manual memory cleanup
python -c "
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
print('Memory cleared')
"
```

**Problem:** System memory exhaustion

```bash
# Monitor memory usage
free -h
ps aux --sort=-%mem | head

# Optimize memory settings
echo 'DOCMIND_MAX_DOCUMENT_SIZE_MB=50' >> .env
echo 'AGENT_CONCURRENCY_LIMIT=3' >> .env
echo 'CACHE_TTL_SECONDS=180' >> .env
```

#### CPU and Processing Optimization

**Problem:** High CPU usage

```bash
# Limit CPU-intensive operations
echo 'TOKENIZERS_PARALLELISM=false' >> .env
echo 'OMP_NUM_THREADS=4' >> .env

# Monitor CPU usage patterns
top -p $(pgrep -f streamlit)
```

**Problem:** Slow document processing

```bash
# Optimize document processing
echo 'DOCMIND_PARALLEL_PROCESSING=true' >> .env
echo 'DOCMIND_CHUNK_SIZE=256' >> .env  # Smaller chunks for speed
echo 'DOCMIND_CHUNK_OVERLAP=25' >> .env
```

### Model and Inference Issues

#### Qwen3-4B-Instruct-2507-FP8 Specific Problems

**Problem:** Model won't load or crashes

```bash
# Verify model download
python -c "
from huggingface_hub import hf_hub_download
try:
    hf_hub_download(repo_id='Qwen/Qwen3-4B-Instruct-2507-FP8', filename='config.json')
    print('Model download: OK')
except Exception as e:
    print(f'Model download: FAILED - {e}')
"

# Clear model cache and redownload
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507-FP8
```

**Problem:** FP8 quantization not working

```bash
# Check FP8 hardware support
python -c "
import torch
if torch.cuda.is_available():
    compute_cap = torch.cuda.get_device_capability()
    print(f'Compute Capability: {compute_cap[0]}.{compute_cap[1]}')
    print(f'FP8 Support: {\"Optimal\" if compute_cap[0] >= 8 else \"Limited\"}')
"

# Enable FP8 optimizations
echo 'VLLM_QUANTIZATION=fp8' >> .env
echo 'VLLM_KV_CACHE_DTYPE=fp8_e5m2' >> .env
```

**Problem:** 128K context window issues

```bash
# Validate context configuration
echo 'DOCMIND_CONTEXT_LENGTH=131072' >> .env
echo 'DOCMIND_MAX_CONTEXT_LENGTH=131072' >> .env

# Test long context processing
python -c "
text = 'Test context. ' * 10000  # ~100K tokens
print(f'Test context length: ~{len(text.split())} tokens')
"
```

#### vLLM Backend Issues

**Problem:** vLLM server connection failures

```bash
# Check vLLM server status
curl -f http://localhost:8000/health || echo "vLLM server not responding"

# Start vLLM server manually for debugging
vllm serve Qwen/Qwen3-4B-Instruct-2507-FP8 \
  --port 8000 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e5m2
```

**Problem:** FlashInfer backend not active

```bash
# Verify FlashInfer installation
pip show vllm | grep flashinfer

# Reinstall with FlashInfer
uv pip install "vllm[flashinfer]>=0.10.1" --extra-index-url https://download.pytorch.org/whl/cu128

# Test FlashInfer activation
python -c "
import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASHINFER'
import vllm
print('FlashInfer backend configured')
"
```

### Document Processing Issues

#### File Format and Upload Problems

**Problem:** Unsupported file formats

```bash
# Check supported formats
echo "Supported formats:"
echo "Documents: PDF, DOCX, TXT, RTF, MD"
echo "Spreadsheets: XLSX, CSV, TSV"  
echo "Presentations: PPTX"
echo "Web: HTML, XML, JSON"
echo "Archives: ZIP (extracts contents)"
echo "Email: MSG"
echo "eBooks: EPUB"
echo "Code: PY, JS, Java, C++, etc."
```

**Problem:** Large file processing failures

```bash
# Check file size limits
ls -lh your_large_file.pdf
echo "Current limit: $(grep MAX_DOCUMENT_SIZE .env || echo '100MB default')"

# Increase limits if needed
echo 'DOCMIND_MAX_DOCUMENT_SIZE_MB=200' >> .env
echo 'DOCMIND_CHUNK_SIZE=1024' >> .env  # Larger chunks for big files
```

**Problem:** Text extraction quality issues

```bash
# Test text extraction
python -c "
from src.utils.document import load_documents_unstructured
import asyncio

async def test_extraction():
    docs = await load_documents_unstructured(['your_file.pdf'], settings)
    print(f'Extracted {len(docs[0].text)} characters')
    print('Sample:', docs[0].text[:200])

asyncio.run(test_extraction())
"
```

#### Vector Database and Search Issues

**Problem:** Qdrant connection failures

```bash
# Check Qdrant status
curl -f http://localhost:6333/collections || echo "Qdrant not responding"

# Start Qdrant manually
docker run -p 6333:6333 qdrant/qdrant:latest

# Test Qdrant connectivity
python -c "
from qdrant_client import QdrantClient
try:
    client = QdrantClient(host='localhost', port=6333)
    print('Qdrant: Connected')
except:
    print('Qdrant: Connection failed')
"
```

**Problem:** Search results irrelevant or missing

```bash
# Check embedding model status
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-m3')
test_embedding = model.encode('test text')
print(f'Embedding model: OK, dimension: {len(test_embedding)}')
"

# Test search functionality
python scripts/test_search.py --query "test query" --debug
```

## Advanced Diagnostics

### Comprehensive System Analysis

#### Full System Health Check

```bash
# Run complete diagnostic suite
python scripts/comprehensive_diagnostics.py --full-report

# This checks:
# - Hardware compatibility and performance
# - Software dependencies and versions
# - Configuration validity and optimization
# - Agent system functionality
# - Model loading and inference
# - Search and retrieval accuracy
# - Memory and resource utilization
```

#### Performance Profiling

```bash
# Profile system performance
python scripts/performance_profiler.py --detailed

# Monitor during typical usage
python scripts/continuous_monitor.py --duration 300  # 5 minutes
```

#### Log Analysis Tools

```bash
# Automated log analysis
python scripts/log_analyzer.py --last-24h --severity error,warning

# Real-time issue monitoring
tail -f logs/app.log | python scripts/log_filter.py --issues-only
```

### Resource Management Diagnostics

#### GPU Resource Analysis

```bash
# Detailed GPU analysis
python scripts/gpu_analysis.py --memory-fragmentation --utilization-patterns

# GPU memory leak detection
python scripts/gpu_memory_monitor.py --leak-detection --duration 600
```

#### Agent Performance Analysis

```bash
# Agent coordination profiling
python scripts/agent_profiler.py --coordination-analysis --response-times

# Agent resource usage
python scripts/agent_resource_monitor.py --memory --cpu --gpu
```

## Environment Configuration Reference

### Complete Configuration Template

```bash
# Core system settings
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_MAX_CONTEXT_LENGTH=131072

# Multi-agent system
ENABLE_MULTI_AGENT=true
AGENT_TIMEOUT_SECONDS=30
MAX_CONTEXT_TOKENS=65000
ENABLE_DSPY_OPTIMIZATION=true
FALLBACK_STRATEGY=basic_rag

# Performance optimization
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_CUDNN_PREFILL=1
VLLM_QUANTIZATION=fp8
VLLM_KV_CACHE_DTYPE=fp8_e5m2

# Model configuration
DOCMIND_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_CONTEXT_WINDOW_SIZE=131072

# Search and retrieval
DOCMIND_RETRIEVAL_STRATEGY=hybrid
DOCMIND_EMBEDDING_MODEL=BAAI/bge-m3
DOCMIND_USE_RERANKING=true
RRF_ALPHA=0.7

# Document processing
DOCMIND_CHUNK_SIZE=512
DOCMIND_CHUNK_OVERLAP=50
DOCMIND_MAX_DOCUMENT_SIZE_MB=100
DOCMIND_PARALLEL_PROCESSING=true

# Quality and reliability
MIN_VALIDATION_SCORE=0.7
ENABLE_HALLUCINATION_CHECK=true
SOURCE_ATTRIBUTION_REQUIRED=true

# Resource management
DOCMIND_AUTO_GPU_CLEANUP=true
DOCMIND_SAFE_FALLBACKS=true
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
TOKENIZERS_PARALLELISM=false

# Logging and debugging
DOCMIND_LOG_LEVEL=INFO
DOCMIND_LOG_GPU_ERRORS=true
```

### Performance Targets Reference

#### Expected Performance Metrics

**RTX 4090 Laptop (16GB VRAM):**

- **Decode Speed**: 100-160 tokens/second
- **Prefill Speed**: 800-1300 tokens/second
- **VRAM Usage**: 12-14GB typical, <16GB maximum
- **Agent Coordination**: <200ms overhead
- **Response Time**: 1-3 seconds for complex queries

**RTX 4090 (24GB VRAM):**

- **Decode Speed**: 120-180 tokens/second
- **Prefill Speed**: 1000-1500 tokens/second
- **VRAM Usage**: 12-16GB typical
- **Batch Processing**: Small batches supported
- **Response Time**: 0.8-2.5 seconds for complex queries

#### Quality Benchmarks

**Multi-Agent System:**

- **Validation Scores**: >0.8 for most queries
- **Fallback Rate**: <5% under normal conditions
- **Agent Success Rate**: >95% individual agent success
- **Context Preservation**: 90%+ across conversation turns

## Getting Additional Help

### Community Resources

**Documentation:**

- **Getting Started**: [getting-started.md](getting-started.md)
- **User Guide**: [user-guide.md](user-guide.md)
- **Advanced Features**: [advanced-features.md](advanced-features.md)
- **Developer Documentation**: [../developers/](../developers/)

**Support Channels:**

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Community Q&A and best practices
- **Documentation Contributions**: Improve these guides

### When to Report Issues

**Report bugs when you encounter:**

- Consistent crashes or startup failures
- Performance significantly below targets
- Multi-agent system not functioning
- Data loss or corruption issues
- Security or privacy concerns

**Include in bug reports:**

- System specifications (GPU, RAM, OS)
- Complete error messages and stack traces
- Steps to reproduce the issue
- Log files (`logs/app.log`)
- Configuration settings (`.env` file, sensitive data removed)

**Performance optimization requests:**

- Specific use case and requirements
- Current vs desired performance metrics
- Hardware constraints or limitations
- Document types and sizes typically processed

---

**Need basic setup help?** Start with the [Getting Started Guide](getting-started.md) for initial installation and configuration.
