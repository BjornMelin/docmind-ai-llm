# Environment Variable Configuration Guide

## Overview

This guide explains how to configure DocMind AI using environment variables. DocMind AI uses a unified configuration system with 30 essential environment variables organized into logical categories, replacing the previous 55+ variable system for improved maintainability and clarity.

## Purpose

- Configure all aspects of DocMind AI through environment variables
- Understand the unified `DOCMIND_*` prefix pattern with nested configuration
- Set up proper integration with Docker and local development environments
- Migrate from legacy variable names to the new unified system

## Audience

Developers setting up DocMind AI for local development, staging, or production environments.

## Prerequisites

- Python 3.10+ with DocMind AI dependencies installed
- Basic understanding of environment variables and `.env` files
- Familiarity with Docker if using containerized deployment
- RTX 4090 (or similar 16GB VRAM GPU) for full system functionality

## Quick Start

1. **Copy the example configuration:**
   ```bash
   cp .env.example .env
   ```

2. **Edit essential variables in `.env`:**
   ```bash
   # Core Application  
   DOCMIND_DEBUG=false
   DOCMIND_LOG_LEVEL=INFO
   DOCMIND_BASE_PATH=./

   # LLM Configuration
   DOCMIND_LLM__BASE_URL=http://localhost:11434
   DOCMIND_LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8

   # Vector Storage
   DOCMIND_QDRANT__URL=http://localhost:6333
   ```

3. **Start the application:**
   ```bash
   streamlit run src/app.py
   ```

## Configuration System Overview

DocMind AI uses a **unified prefix pattern** with **nested configuration support**:

- **Prefix**: All variables start with `DOCMIND_`
- **Nesting**: Use double underscores (`__`) for nested sections
- **Example**: `DOCMIND_VLLM__ATTENTION_BACKEND=FLASHINFER`

### Pydantic Settings v2 Integration

The configuration system uses Pydantic Settings v2 with these features:

```python
class DocMindSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DOCMIND_", 
        env_nested_delimiter="__",  # Enable DOCMIND_VLLM__ATTENTION_BACKEND
        case_sensitive=False,
        extra="forbid"
    )
    
    # Nested vLLM configuration
    class VLLMConfig(BaseModel):
        attention_backend: str = "FLASHINFER"
        gpu_memory_utilization: float = 0.85
        kv_cache_dtype: str = "fp8_e5m2"
```

### Variable Naming Conventions

| Pattern | Example | Purpose |
|---------|---------|---------|
| `DOCMIND_*` | `DOCMIND_DEBUG=true` | Core application settings |
| `DOCMIND_LLM__*` | `DOCMIND_LLM__MODEL=...` | LLM configuration |
| `DOCMIND_VLLM__*` | `DOCMIND_VLLM__ATTENTION_BACKEND=...` | vLLM optimization |
| `DOCMIND_AGENTS__*` | `DOCMIND_AGENTS__MAX_RETRIES=2` | Multi-agent system |
| `DOCMIND_PROCESSING__*` | `DOCMIND_PROCESSING__CHUNK_SIZE=1024` | Document processing |
| `DOCMIND_QDRANT__*` | `DOCMIND_QDRANT__URL=...` | Vector database |

## Configuration Categories

DocMind AI organizes its 30 essential environment variables into 6 logical categories:

### 1. Core Application (8 variables)

Essential application configuration for debugging, logging, and performance limits.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_DEBUG` | boolean | `false` | Enable debug mode for detailed logging |
| `DOCMIND_LOG_LEVEL` | string | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `DOCMIND_BASE_PATH` | path | `./` | Base path for data, cache, and logs directories |
| `DOCMIND_CONTEXT_SIZE` | integer | `131072` | LLM context window size in tokens (128K) |
| `DOCMIND_ENABLE_GPU_ACCELERATION` | boolean | `true` | Enable GPU acceleration |
| `DOCMIND_ENABLE_PERFORMANCE_LOGGING` | boolean | `true` | Enable performance monitoring |
| `DOCMIND_MAX_MEMORY_GB` | float | `4.0` | Maximum system memory usage in GB |
| `DOCMIND_MAX_VRAM_GB` | float | `14.0` | Maximum VRAM usage in GB |

### 2. LLM Configuration (4 variables)

Configuration for the primary language model endpoint and generation parameters.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_LLM__BASE_URL` | string | `http://localhost:11434` | LLM service base URL (Ollama/vLLM) |
| `DOCMIND_LLM__MODEL` | string | `Qwen/Qwen3-4B-Instruct-2507-FP8` | LLM model name |
| `DOCMIND_LLM__TEMPERATURE` | float | `0.1` | LLM temperature setting (0.0-2.0) |
| `DOCMIND_LLM__MAX_TOKENS` | integer | `2048` | Maximum output tokens per generation |

### 3. vLLM Optimization (6 variables)

Performance optimization settings for vLLM backend with FP8 quantization.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_VLLM__ATTENTION_BACKEND` | string | `FLASHINFER` | vLLM attention backend (FLASHINFER, FLASH_ATTN) |
| `DOCMIND_VLLM__GPU_MEMORY_UTILIZATION` | float | `0.85` | GPU memory utilization ratio (0.1-0.95) |
| `DOCMIND_VLLM__KV_CACHE_DTYPE` | string | `fp8_e5m2` | KV cache data type for memory optimization |
| `DOCMIND_VLLM__MAX_MODEL_LEN` | integer | `131072` | Maximum model length (context window) |
| `DOCMIND_VLLM__ENABLE_CHUNKED_PREFILL` | boolean | `true` | Enable chunked prefill optimization |
| `DOCMIND_VLLM__CALCULATE_KV_SCALES` | boolean | `true` | Calculate KV scales for FP8 quantization |

### 4. Multi-Agent System (4 variables)

Configuration for the 5-agent coordination system with LangGraph supervision.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_AGENTS__ENABLE_MULTI_AGENT` | boolean | `true` | Enable multi-agent system |
| `DOCMIND_AGENTS__DECISION_TIMEOUT` | integer | `200` | Agent decision timeout in seconds |
| `DOCMIND_AGENTS__MAX_RETRIES` | integer | `2` | Maximum agent retry attempts |
| `DOCMIND_AGENTS__MAX_CONCURRENT` | integer | `3` | Maximum concurrent agents |

### 5. Document Processing (4 variables)

Document parsing and chunking configuration optimized for BGE-M3 embeddings.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_PROCESSING__CHUNK_SIZE` | integer | `1024` | Document chunk size in characters |
| `DOCMIND_PROCESSING__OVERLAP` | integer | `100` | Chunk overlap size in characters |
| `DOCMIND_PROCESSING__MAX_SIZE_MB` | integer | `100` | Maximum document size in MB |
| `DOCMIND_PROCESSING__STRATEGY` | string | `hi_res` | Document parsing strategy (hi_res, fast) |

### 6. Vector Storage (4 variables)

Qdrant vector database and retrieval configuration with hybrid search.

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DOCMIND_QDRANT__URL` | string | `http://localhost:6333` | Qdrant vector database URL |
| `DOCMIND_QDRANT__COLLECTION` | string | `docmind_docs` | Qdrant collection name |
| `DOCMIND_RETRIEVAL__TOP_K` | integer | `10` | Number of top results to retrieve |
| `DOCMIND_RETRIEVAL__USE_RERANKING` | boolean | `true` | Enable BGE-reranker-v2-m3 result reranking |

## Common Configuration Examples

### Development Environment

Optimized for fast development cycles with reduced resource usage:

```bash
# Core settings
DOCMIND_DEBUG=true
DOCMIND_LOG_LEVEL=DEBUG
DOCMIND_BASE_PATH=./dev

# Reduced resource usage
DOCMIND_CONTEXT_SIZE=8192
DOCMIND_MAX_VRAM_GB=8.0

# Faster timeouts for development
DOCMIND_AGENTS__DECISION_TIMEOUT=100

# Smaller chunks for faster processing
DOCMIND_PROCESSING__CHUNK_SIZE=512
DOCMIND_PROCESSING__OVERLAP=50
```

### Production Environment

Optimized for maximum performance and reliability:

```bash
# Core settings
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=WARNING
DOCMIND_ENABLE_PERFORMANCE_LOGGING=true

# Maximum performance
DOCMIND_CONTEXT_SIZE=131072
DOCMIND_MAX_VRAM_GB=14.0
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.90

# Increased concurrency
DOCMIND_AGENTS__MAX_CONCURRENT=5
DOCMIND_AGENTS__DECISION_TIMEOUT=300

# Optimized processing
DOCMIND_PROCESSING__CHUNK_SIZE=1024
DOCMIND_PROCESSING__MAX_SIZE_MB=200
```

### Minimal Resource Setup

Configuration for systems with limited GPU memory (8GB):

```bash
# Reduced resource limits
DOCMIND_MAX_VRAM_GB=6.0
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.75

# Smaller context window
DOCMIND_CONTEXT_SIZE=32768

# Reduced batch sizes
DOCMIND_PROCESSING__CHUNK_SIZE=512
DOCMIND_AGENTS__MAX_CONCURRENT=2
```

## Docker Integration

### Docker Compose Configuration

DocMind AI supports Docker deployment with environment variable propagation:

```yaml
# docker-compose.yml
services:
  docmind:
    image: docmind-ai:latest
    environment:
      # Core application
      - DOCMIND_DEBUG=false
      - DOCMIND_LOG_LEVEL=INFO
      - DOCMIND_BASE_PATH=/app/data
      
      # LLM configuration
      - DOCMIND_LLM__BASE_URL=http://vllm:8000
      - DOCMIND_LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
      
      # vLLM optimization - Docker containers need both formats
      - DOCMIND_VLLM__ATTENTION_BACKEND=FLASHINFER
      - VLLM_ATTENTION_BACKEND=${DOCMIND_VLLM__ATTENTION_BACKEND}
      - DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.85
      - VLLM_GPU_MEMORY_UTILIZATION=${DOCMIND_VLLM__GPU_MEMORY_UTILIZATION}
      - DOCMIND_VLLM__KV_CACHE_DTYPE=fp8_e5m2
      - VLLM_KV_CACHE_DTYPE=${DOCMIND_VLLM__KV_CACHE_DTYPE}
      
      # Vector database
      - DOCMIND_QDRANT__URL=http://qdrant:6333
    volumes:
      - ./data:/app/data
      - ./cache:/app/cache
    ports:
      - "8501:8501"
    depends_on:
      - vllm
      - qdrant

  vllm:
    image: vllm/vllm-openai:latest
    environment:
      - VLLM_ATTENTION_BACKEND=FLASHINFER
      - VLLM_GPU_MEMORY_UTILIZATION=0.85
      - VLLM_KV_CACHE_DTYPE=fp8_e5m2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

### Variable Propagation Strategy

Docker containers often expect specific variable names, so DocMind AI uses a **dual propagation approach**:

1. **Python Application**: Uses unified `DOCMIND_*` variables
2. **Docker Containers**: Receive both `DOCMIND_*` and legacy formats (e.g., `VLLM_*`)
3. **Environment Variable Substitution**: Use Docker's `${VAR}` syntax for propagation

## Migration Guide

### Upgrading from Legacy Variables

If you have an existing DocMind AI installation, you may need to update your environment variables. The system provides **backward compatibility** for one release cycle with deprecation warnings.

### Critical Variable Changes

These variable names have changed and should be updated:

| Old Variable | New Variable | Notes |
|-------------|--------------|-------|
| `OLLAMA_BASE_URL` | `DOCMIND_LLM__BASE_URL` | Unified LLM endpoint |
| `CONTEXT_SIZE` | `DOCMIND_CONTEXT_SIZE` | Consistent naming |
| `DOCMIND_CONTEXT_WINDOW_SIZE` | `DOCMIND_CONTEXT_SIZE` | Simplified naming |
| `VLLM_ATTENTION_BACKEND` | `DOCMIND_VLLM__ATTENTION_BACKEND` | Nested structure |
| `DOCMIND_ENABLE_MULTI_AGENT` | `DOCMIND_AGENTS__ENABLE_MULTI_AGENT` | Grouped under agents |
| `DOCMIND_CHUNK_SIZE` | `DOCMIND_PROCESSING__CHUNK_SIZE` | Grouped under processing |

### Migration Script

Run the migration script to automatically update your `.env` file:

```bash
# Check for deprecated variables
uv run python scripts/validate_environment_variables.py

# Migrate variables (creates .env.backup)
uv run python scripts/migrate_environment_variables.py

# Verify migration
uv run python scripts/validate_environment_variables.py --check-new
```

### Manual Migration Steps

1. **Backup your current `.env` file:**
   ```bash
   cp .env .env.backup
   ```

2. **Update variable names** using the mapping table above

3. **Add missing variables** from the 30 essential variables list

4. **Test the configuration:**
   ```bash
   uv run python -c "from src.config.settings import settings; print(settings.model_dump())"
   ```

## Troubleshooting

### Common Issues

#### Variable Not Recognized

**Error**: `ValidationError: Environment variable DOCMIND_XYZ not found`

**Solution**:
1. Check variable name spelling and prefix
2. Verify the variable is in the 30 essential variables list
3. Restart the application after changing `.env`

```bash
# Validate your configuration
uv run python -c "from src.config.settings import settings; print(settings)"
```

#### GPU Memory Issues

**Error**: `CUDA out of memory` or high VRAM usage

**Solution**:
1. Reduce `DOCMIND_VLLM__GPU_MEMORY_UTILIZATION` (try 0.75 or 0.65)
2. Lower `DOCMIND_MAX_VRAM_GB` to match your GPU
3. Reduce `DOCMIND_CONTEXT_SIZE` for smaller memory footprint

```bash
# Check GPU memory usage
nvidia-smi

# Adjust memory settings
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.65
DOCMIND_MAX_VRAM_GB=8.0
DOCMIND_CONTEXT_SIZE=65536
```

#### LLM Connection Failed

**Error**: `Connection refused` or `Service unavailable`

**Solution**:
1. Verify `DOCMIND_LLM__BASE_URL` is correct
2. Ensure Ollama or vLLM is running
3. Check firewall and port accessibility

```bash
# Test LLM endpoint
curl http://localhost:11434/api/version

# Check if Ollama is running
ollama list
```

#### Agent Timeout Issues

**Error**: `Agent decision timeout` or slow responses

**Solution**:
1. Increase `DOCMIND_AGENTS__DECISION_TIMEOUT`
2. Reduce `DOCMIND_AGENTS__MAX_CONCURRENT`
3. Enable fallback RAG mode

```bash
# Adjust agent settings
DOCMIND_AGENTS__DECISION_TIMEOUT=300
DOCMIND_AGENTS__MAX_CONCURRENT=2
DOCMIND_ENABLE_FALLBACK_RAG=true
```

### Debug Mode

Enable comprehensive logging for troubleshooting:

```bash
# Enable debug mode
DOCMIND_DEBUG=true
DOCMIND_LOG_LEVEL=DEBUG
DOCMIND_ENABLE_PERFORMANCE_LOGGING=true

# Run with debug output
streamlit run src/app.py
```

## Best Practices

### Security

- **Never commit `.env` files** to version control
- **Use separate `.env` files** for different environments
- **Restrict file permissions**: `chmod 600 .env`
- **Use environment-specific values** for URLs and paths

### Performance

- **Match VRAM settings to your GPU**: RTX 4090 = 14GB, RTX 4060 = 16GB
- **Use appropriate context sizes**: Larger = more memory, slower inference
- **Enable GPU acceleration**: Always set `DOCMIND_ENABLE_GPU_ACCELERATION=true`
- **Optimize for your use case**: Development vs production settings

### Maintainability

- **Group related variables**: Use the 6-category structure
- **Document custom values**: Add comments in `.env` files
- **Use consistent naming**: Follow the `DOCMIND_CATEGORY__SETTING` pattern
- **Validate after changes**: Test configuration with validation scripts

### Development Workflow

1. **Start with `.env.example`**: Copy and customize for your environment
2. **Use development settings**: Enable debug mode and reduce resource usage
3. **Test incrementally**: Change one category of variables at a time
4. **Monitor performance**: Use performance logging during development
5. **Validate before production**: Run validation scripts before deployment

### Production Deployment

- **Use environment variables**: Don't rely on `.env` files in production
- **Set resource limits**: Configure memory and VRAM limits appropriately
- **Enable monitoring**: Use performance logging and error tracking
- **Test thoroughly**: Validate all functionality with production settings
- **Plan for scaling**: Consider concurrent user load when setting limits

## Related Documentation

- **[Architecture Guide](./architecture.md)**: System architecture overview
- **[Development Guide](./development-guide.md)**: Development setup and workflows
- **[GPU and Performance Guide](./gpu-and-performance.md)**: GPU optimization and troubleshooting
- **[Integration Guide](../INTEGRATION_GUIDE.md)**: Integration with external systems
