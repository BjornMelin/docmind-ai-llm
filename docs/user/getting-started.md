# Getting Started with DocMind AI

Welcome to DocMind AI - a local-first AI document analysis system. This guide will help you go from zero to analyzing your first document in under 15 minutes.

## What You'll Accomplish

By the end of this guide, you'll have:

- ✅ DocMind AI running on your system
- ✅ Your first documents uploaded and analyzed
- ✅ A working understanding of the multi-agent system
- ✅ Confidence to explore advanced features

## Before You Start

### Minimum System Requirements

**Essential Hardware:**

- **GPU**: NVIDIA RTX 4090 (16GB+ VRAM)
- **RAM**: 32GB system memory
- **Storage**: 50GB+ free space
- **OS**: Linux (Ubuntu 20.04+) or Windows 10/11 with WSL2

**Software Prerequisites:**

- **Python**: 3.10-3.12
- **CUDA**: Version 12.8+
- **NVIDIA Driver**: 550.54.14+

> **💡 New to GPU setup?** Don't worry - we'll guide you through the essentials below. For detailed GPU configuration, see [Advanced Features](advanced-features.md#gpu-optimization).

### Quick Hardware Check

```bash
# Check if you have the minimum requirements
nvidia-smi  # Should show your GPU and VRAM
python --version  # Should show 3.10-3.12
```

If these commands work, you're ready to proceed!

## Step 1: Installation (5 minutes)

### Clone and Setup

```bash
# Get the code
git clone https://github.com/BjornMelin/docmind-ai-llm.git
cd docmind-ai-llm

# Install dependencies (this will take a few minutes)
uv sync --extra gpu
```

### Essential Configuration

```bash
# Create your configuration file
cp .env.example .env

# Set minimal required variables
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.85' >> .env
echo 'DOCMIND_ENABLE_MULTI_AGENT=true' >> .env
```

## Step 2: Start DocMind AI (2 minutes)

### Launch the Application

```bash
# Start DocMind AI
uv run streamlit run src/app.py
```

You should see:

```text
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
```

### Verify System Status

1. Open <http://localhost:8501> in your browser
2. Check the sidebar for:
   - ✅ **GPU Status**: Should show "GPU: NVIDIA GeForce RTX 4090"
   - ✅ **Model Status**: Should show "Model: Ready"
   - ✅ **Multi-Agent**: Should show "5 Agents Active"

> **⚠️ Issues?** If you see red error messages, check our [Troubleshooting Quick Fixes](#troubleshooting-quick-fixes) below.

## Step 3: Your First Document Analysis (5 minutes)

### Upload Documents

1. **Click "Upload Documents"** in the sidebar
2. **Select files** - try a PDF, Word doc, or text file
3. **Wait for processing** - you'll see a progress bar
4. **Confirm success** - uploaded files appear in the document list

### Ask Your First Question

**Try this example query:**

```text
What are the main topics covered in my documents?
```

**Watch the multi-agent system work:**

- 🔄 **Query Router** analyzes your question
- 🔍 **Retrieval Agent** searches your documents  
- 🔧 **Synthesis Agent** combines findings
- ✅ **Validator** ensures response quality

### Interpret Your Results

Your response will include:

- **Main Answer**: Direct response to your question
- **Sources**: Which documents were referenced
- **Confidence Score**: Quality indicator (aim for >0.8)
- **Processing Time**: Should be 1-3 seconds

## Step 4: Explore Key Features (3 minutes)

### Try Different Query Types

**Simple Lookup:**

```text
What is the budget mentioned in the financial report?
```

**Comparison Analysis:**

```text
Compare the methodologies between document A and document B.
```

**Multi-step Question:**

```text
What are the risks identified, how severe are they, and what mitigation strategies are proposed?
```

### Understanding System Responses

Pay attention to:

- **Response Time**: Simple queries ~1s, complex ~3s
- **Agent Coordination**: Which agents were activated
- **Source Attribution**: Clear document references
- **Quality Scores**: Higher scores indicate better responses

## Essential Configuration Options

### Performance Tuning (For Your Hardware)

```bash
# In your .env file - adjust based on your GPU
DOCMIND_GPU_MEMORY_UTILIZATION=0.85  # Lower to 0.75 if memory issues
DOCMIND_MAX_CONTEXT_LENGTH=131072    # Full 128K context window
VLLM_ATTENTION_BACKEND=FLASHINFER    # Fastest attention mechanism
```

### Multi-Agent Settings

```bash
# Agent coordination settings
ENABLE_MULTI_AGENT=true              # Use 5-agent coordination
AGENT_TIMEOUT_SECONDS=30             # Agent response timeout
MAX_CONTEXT_TOKENS=65000             # Context preservation limit
```

## Troubleshooting Quick Fixes

### GPU Not Detected

**Symptoms:** "GPU: CPU Only" in the sidebar

**Quick Fix:**

```bash
# Verify CUDA installation
nvidia-smi && nvcc --version

# If missing, install CUDA 12.8
# Ubuntu: sudo apt install cuda-toolkit-12-8
# Then restart DocMind AI
```

### Out of Memory Errors

**Symptoms:** CUDA out of memory errors

**Quick Fix:**

```bash
# Reduce GPU memory usage
echo 'DOCMIND_GPU_MEMORY_UTILIZATION=0.75' >> .env

# Clear GPU cache and restart
python -c "import torch; torch.cuda.empty_cache()"
```

### Slow Performance

**Symptoms:** Queries taking >10 seconds

**Quick Fix:**

```bash
# Enable performance optimizations
echo 'VLLM_ATTENTION_BACKEND=FLASHINFER' >> .env
echo 'VLLM_USE_CUDNN_PREFILL=1' >> .env

# Restart the application
```

### Model Won't Load

**Symptoms:** "Model: Error" in sidebar

**Quick Fix:**

```bash
# Check model accessibility
python -c "from transformers import AutoTokenizer; print('Model available')"

# If this fails, check your internet connection
# Models download automatically on first use
```

## What's Next?

Congratulations! You now have DocMind AI running successfully. Here's your learning path:

### Immediate Next Steps

1. **📖 [User Guide](user-guide.md)** - Master daily workflows and core features
2. **🔧 [Advanced Features](advanced-features.md)** - Unlock power-user capabilities
3. **🆘 [Troubleshooting Reference](troubleshooting-reference.md)** - Solve any issues you encounter

### Explore Advanced Capabilities

- **Multi-Agent Coordination**: Let 5 specialized agents handle complex queries
- **Hybrid Search**: Semantic + keyword search for better results
- **128K Context**: Analyze very large documents in full
- **Multi-Language**: Work with documents in 100+ languages

## Success Indicators

You've successfully set up DocMind AI when you see:

- ✅ **System Status**: All green indicators in sidebar
- ✅ **Document Processing**: Files upload and process without errors
- ✅ **Query Responses**: Receive relevant answers in 1-3 seconds
- ✅ **Agent Coordination**: See multi-agent workflow in action
- ✅ **Performance**: Decode speed >100 tokens/second

## Getting Help

If you encounter issues:

1. **Check Logs**: Look at `logs/app.log` for detailed error information
2. **Try Quick Fixes**: Use the troubleshooting section above
3. **Consult References**: Check [Troubleshooting Reference](troubleshooting-reference.md)
4. **Report Issues**: GitHub issues with logs and system details

---

**Ready to dive deeper?** Continue to the [User Guide](user-guide.md) to master DocMind AI's powerful document analysis capabilities.
