# Troubleshooting Guide

Quick solutions to the most common issues encountered when running DocMind AI locally.

## Quick Diagnostic Commands

Before troubleshooting specific issues, run these diagnostic commands:

```bash
# System health check
nvidia-smi                    # Check GPU status
python --version              # Verify Python 3.10-3.12
curl http://localhost:11434/api/version  # Check Ollama service

# DocMind AI status
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from src.config import settings; print('Config loaded OK')"
```

## Common Issues and Solutions

### 1. DocMind AI Won't Start

**Symptoms:**
- Application crashes on startup
- "Module not found" errors
- Streamlit server won't start

**Quick Fix:**
```bash
# Reinstall dependencies
uv sync

# Check Python version (should be 3.10-3.12)
python --version

# Clear Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Check port availability
lsof -i :8501  # Kill any process using port 8501
```

**Alternative Solution:**
```bash
# Complete clean installation
rm -rf .venv
uv sync
uv run streamlit run src/app.py
```

### 2. GPU Not Detected

**Symptoms:**
- Shows "CPU Only" in sidebar
- CUDA errors in logs
- Very slow performance

**Quick Fix:**
```bash
# Verify GPU hardware
nvidia-smi  # Should show your GPU and VRAM

# Check CUDA installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Force GPU in configuration
echo 'DOCMIND_DEVICE=cuda' >> .env
echo 'DOCMIND_ENABLE_GPU_ACCELERATION=true' >> .env
```

**If GPU still not detected:**
```bash
# Install/update NVIDIA drivers
sudo ubuntu-drivers autoinstall
sudo reboot

# Reinstall PyTorch with CUDA support
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

### 3. Out of Memory Errors

**Symptoms:**
- "CUDA out of memory" errors
- System crashes during processing
- Application becomes unresponsive

**Quick Fix:**
```bash
# Reduce GPU memory usage
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.75' >> .env
echo 'DOCMIND_CONTEXT_WINDOW_SIZE=32768' >> .env

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Reduce document processing limits
echo 'DOCMIND_MAX_DOCUMENT_SIZE_MB=50' >> .env
echo 'DOCMIND_CHUNK_SIZE=256' >> .env
```

**Monitor memory usage:**
```bash
# Watch GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv --loop=1

# Check system memory
free -h
```

### 4. Ollama Connection Failed

**Symptoms:**
- "Connection refused" errors
- "Ollama not responding" messages
- No model responses

**Quick Fix:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/version

# If not running, start Ollama
ollama serve

# In a new terminal, verify model is installed
ollama list
ollama pull qwen3-4b-instruct-2507  # Download if missing
```

**Alternative solutions:**
```bash
# Restart Ollama service
pkill ollama
ollama serve

# Check model accessibility
ollama run qwen3-4b-instruct-2507 "Hello, are you working?"
```

### 5. Slow Performance

**Symptoms:**
- Queries taking >10 seconds
- High CPU/GPU usage
- System appears frozen

**Quick Fix:**
```bash
# Enable performance optimizations
echo 'VLLM_ATTENTION_BACKEND=FLASHINFER' >> .env
echo 'VLLM_USE_CUDNN_PREFILL=1' >> .env

# Check system resources
nvidia-smi  # GPU utilization should be <90%
htop        # Monitor CPU and memory

# Optimize for your hardware
echo 'DOCMIND_PERFORMANCE_TIER=medium' >> .env
echo 'AGENT_TIMEOUT_SECONDS=30' >> .env
```

**Performance monitoring:**
```bash
# Run performance validation
python scripts/performance_validation.py

# Expected performance (RTX 4090):
# - Decode: 100-160 tokens/second
# - Response time: 1-3 seconds
# - VRAM usage: 12-14GB
```

### 6. Documents Won't Upload

**Symptoms:**
- Upload fails silently
- "Unsupported file type" errors
- Processing never completes

**Quick Fix:**
```bash
# Check supported formats
echo "Supported: PDF, DOCX, TXT, XLSX, CSV, JSON, XML, MD, RTF, MSG, PPTX, ODT, EPUB"

# Check file size
ls -lh your_file.pdf  # Should be <100MB by default

# Test with simple file
echo "Test document content for troubleshooting" > test.txt
# Try uploading test.txt through the UI
```

**Alternative solutions:**
```bash
# Increase file size limits
echo 'DOCMIND_MAX_DOCUMENT_SIZE_MB=200' >> .env

# Check document processing
tail -f logs/app.log | grep -i "document\|processing\|error"
```

### 7. Multi-Agent System Not Working

**Symptoms:**
- Only getting basic responses
- No agent coordination visible
- "Fallback mode" messages

**Quick Fix:**
```bash
# Enable multi-agent system
echo 'ENABLE_MULTI_AGENT=true' >> .env
echo 'AGENT_TIMEOUT_SECONDS=30' >> .env

# Check agent initialization
python -c "
try:
    from src.agents.coordinator import MultiAgentCoordinator
    print('Multi-agent system: OK')
except Exception as e:
    print(f'Multi-agent system: ERROR - {e}')
"
```

**Verify agent coordination:**
```bash
# Check logs for agent activity
tail -f logs/app.log | grep -i "agent\|coordination"

# Test with a complex query that should trigger multiple agents
# Example: "Compare the budget allocations between Q1 and Q2, identify trends, and recommend actions"
```

## Advanced Diagnostics

### System Health Check

```bash
# Run comprehensive system check
python -c "
import torch
import sys
from src.config import settings

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
print(f'Multi-Agent Enabled: {settings.agents.enable_multi_agent}')
print(f'GPU Acceleration: {settings.enable_gpu_acceleration}')
print(f'Context Window: {settings.context_window_size}')
"
```

### Log Analysis

```bash
# Check for common errors
grep -i "error\|failed\|exception" logs/app.log | tail -10

# Monitor real-time logs
tail -f logs/app.log | grep -v "DEBUG"

# Check agent coordination issues
grep -i "agent_timeout\|coordination_failed\|fallback" logs/app.log
```

### Performance Profiling

```bash
# Monitor GPU usage during operation
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
    --format=csv --loop=1 > gpu_usage.log &

# Kill background monitoring
pkill nvidia-smi
```

## Hardware-Specific Solutions

### RTX 4060/4070 (12-16GB VRAM)

```bash
# Optimized settings for mid-range GPUs
cat >> .env << 'EOF'
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
DOCMIND_CONTEXT_WINDOW_SIZE=65536
DOCMIND_MAX_DOCUMENT_SIZE_MB=50
VLLM_ATTENTION_BACKEND=FLASHINFER
EOF
```

### RTX 4090 (24GB VRAM)

```bash
# High-performance settings
cat >> .env << 'EOF'
DOCMIND_GPU_MEMORY_UTILIZATION=0.90
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_MAX_DOCUMENT_SIZE_MB=100
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_KV_CACHE_DTYPE=fp8_e5m2
EOF
```

### CPU-Only Systems

```bash
# CPU-optimized settings
cat >> .env << 'EOF'
DOCMIND_ENABLE_GPU_ACCELERATION=false
DOCMIND_CONTEXT_WINDOW_SIZE=32768
DOCMIND_MAX_MEMORY_GB=4.0
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=4
EOF
```

## When to Seek Additional Help

### Issues Requiring Further Investigation

- Persistent crashes after following these solutions
- Performance significantly below expected targets
- Data corruption or loss
- Security or privacy concerns

### Information to Include When Reporting Issues

1. **System Information:**
   ```bash
   nvidia-smi
   python --version
   uv --version
   cat /etc/os-release  # Linux
   ```

2. **DocMind AI Configuration:**
   ```bash
   cat .env | grep -v "^#" | grep -v "^$"  # Remove comments and empty lines
   ```

3. **Error Logs:**
   ```bash
   tail -50 logs/app.log  # Last 50 lines of logs
   ```

4. **Steps to Reproduce:**
   - Detailed description of what you were doing
   - Specific documents or queries that trigger the issue
   - Screenshots of error messages if applicable

### Support Resources

- **GitHub Issues**: [Report bugs and request features](https://github.com/BjornMelin/docmind-ai-llm/issues)
- **Discussions**: [Community Q&A](https://github.com/BjornMelin/docmind-ai-llm/discussions)
- **Documentation**: Check [getting-started.md](getting-started.md) and [configuration.md](configuration.md)

---

**Quick Reference**: Most issues are resolved by: 1) Restarting Ollama service, 2) Adjusting GPU memory settings, 3) Verifying Python dependencies, 4) Checking file formats and sizes. Start with these solutions before deeper troubleshooting.