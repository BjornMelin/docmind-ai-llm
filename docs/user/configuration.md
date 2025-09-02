# Configuration Guide

This guide shows basic, practical configuration examples for common setups. Edit `.env` based on your hardware and preferences.

## Quick Profiles

### CPU-only (simple, privacy-first)

```bash
cp .env.example .env
cat >> .env << 'EOF'
DOCMIND_ENABLE_GPU_ACCELERATION=false
DOCMIND_CONTEXT_WINDOW_SIZE=32768
EOF
```

### Mid-range GPU (RTX 4060/4070)

```bash
cp .env.example .env
cat >> .env << 'EOF'
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CONTEXT_WINDOW_SIZE=65536
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
VLLM_ATTENTION_BACKEND=FLASHINFER
EOF
```

### High-end GPU (RTX 4090)

```bash
cp .env.example .env
cat >> .env << 'EOF'
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_GPU_MEMORY_UTILIZATION=0.85
VLLM_ATTENTION_BACKEND=FLASHINFER
EOF
```

## Common Options

- Cache directory

```bash
DOCMIND_CACHE_DIR=./cache
```

- LLM backend (Ollama)

```bash
DOCMIND_LLM__BASE_URL=http://localhost:11434
DOCMIND_LLM__MODEL=qwen3-4b-instruct-2507
```

- Agent coordination

```bash
DOCMIND_AGENTS__ENABLE_MULTI_AGENT=true
DOCMIND_AGENTS__DECISION_TIMEOUT=200
```

## Tips

- Start with a profile, then adjust context and memory settings to fit your hardware.
- Keep GPU memory utilization ≤0.85 to avoid OOM errors.
- For cache operations and details, see developers/cache.md.
