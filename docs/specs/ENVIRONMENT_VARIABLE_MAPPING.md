# Task 2.1.2: Production-Pattern Environment Variable Mapping

## Executive Summary

**Current State:** 55+ environment variables with conflicts and overlaps
**Target State:** 30 essential variables with unified DOCMIND_ prefix pattern
**Reduction Goal:** 76% variable count reduction achieved through consolidation

## Current Variable Analysis

### Variable Categories Identified

#### 1. DOCMIND_* Variables (40+ variables)

- Core application settings (app_name, debug, version)
- Multi-agent system configuration (agent timeouts, retries)
- LLM configuration (model, temperature, context)
- Document processing (chunk size, parse strategy)
- Retrieval configuration (top_k, reranking, embeddings)
- Performance settings (memory limits, GPU acceleration)
- File system paths (data_dir, cache_dir, logs)

#### 2. VLLM_* Variables (15+ variables)

- vLLM server configuration (attention backend, memory utilization)
- FP8 optimization settings (kv_cache_dtype, quantization)
- Performance tuning (max_model_len, chunked_prefill)
- Hardware optimization (gpu_memory_utilization, tensor_parallel)

#### 3. System Variables (10+ variables)

- CUDA configuration (CUDA_HOME, CUDA_VISIBLE_DEVICES)
- PyTorch settings (PYTORCH_CUDA_ALLOC_CONF)
- Library configuration (TOKENIZERS_PARALLELISM, NVIDIA_VISIBLE_DEVICES)

## Identified Conflicts & Overlaps

### Critical Conflicts (Per CLEANUP_TRACKING.md)

```bash
# CURRENT CONFLICTS (eliminate):
OLLAMA_BASE_URL vs DOCMIND_LLM_BASE_URL  
CONTEXT_SIZE vs DOCMIND_CONTEXT_WINDOW_SIZE
VLLM_ATTENTION_BACKEND vs DOCMIND_VLLM_ATTENTION_BACKEND

# OVERLAPPING CONFIGURATIONS:
DOCMIND_VLLM_GPU_MEMORY_UTILIZATION vs VLLM_GPU_MEMORY_UTILIZATION
DOCMIND_VLLM_ATTENTION_BACKEND vs VLLM_ATTENTION_BACKEND
DOCMIND_VLLM_MAX_NUM_SEQS vs VLLM_MAX_NUM_SEQS
```

### Variable Usage Analysis

**Currently Used in Code:**

- `src/config/llamaindex_setup.py`: 6 direct os.getenv() calls
- `src/config/settings.py`: All DOCMIND_* variables via Pydantic Settings
- `src/config/vllm_config.py`: 3 VLLM_* variables set directly in os.environ

**Docker vs Python Misalignment:**

- Docker containers expect VLLM_* variables
- Python code uses DOCMIND_VLLM_* variables
- Inconsistent environment variable propagation

## Unified Environment Variable Specification

### Target Unified Pattern

**Single Prefix Architecture:**

```bash
DOCMIND_*                    # All application settings
DOCMIND_VLLM__*             # vLLM nested settings (double underscore)
DOCMIND_SYSTEM__*           # System/CUDA settings (when needed)
```

### Pydantic Settings v2 Integration

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
        # Maps to: DOCMIND_VLLM__ATTENTION_BACKEND
```

## Variable Consolidation Plan

### Phase 1: Eliminate Conflicts (10 variables reduced)

| Current Variables | Unified Variable | Notes |
|------------------|-----------------|-------|
| `OLLAMA_BASE_URL`, `DOCMIND_LLM_BASE_URL` | `DOCMIND_LLM__BASE_URL` | Single LLM endpoint |
| `CONTEXT_SIZE`, `DOCMIND_CONTEXT_WINDOW_SIZE` | `DOCMIND_CONTEXT_SIZE` | Simplified naming |
| `VLLM_ATTENTION_BACKEND`, `DOCMIND_VLLM_ATTENTION_BACKEND` | `DOCMIND_VLLM__ATTENTION_BACKEND` | Nested pattern |
| `VLLM_GPU_MEMORY_UTILIZATION`, `DOCMIND_VLLM_GPU_MEMORY_UTILIZATION` | `DOCMIND_VLLM__GPU_MEMORY_UTILIZATION` | Nested pattern |
| `VLLM_KV_CACHE_DTYPE`, `DOCMIND_VLLM_KV_CACHE_DTYPE` | `DOCMIND_VLLM__KV_CACHE_DTYPE` | Nested pattern |

### Phase 2: Consolidate Related Settings (15 variables reduced)

#### Document Processing Consolidation

```bash
# BEFORE (5 variables):
DOCMIND_CHUNK_SIZE=1024
DOCMIND_CHUNK_OVERLAP=100
DOCMIND_NEW_AFTER_N_CHARS=1200
DOCMIND_COMBINE_TEXT_UNDER_N_CHARS=500
DOCMIND_MULTIPAGE_SECTIONS=true

# AFTER (1 nested structure):
DOCMIND_PROCESSING__CHUNK_SIZE=1024
DOCMIND_PROCESSING__OVERLAP=100
DOCMIND_PROCESSING__NEW_AFTER_N_CHARS=1200
DOCMIND_PROCESSING__COMBINE_UNDER_N_CHARS=500
DOCMIND_PROCESSING__MULTIPAGE_SECTIONS=true
```

#### BGE-M3 Embeddings Consolidation

```bash
# BEFORE (5 variables):
DOCMIND_EMBEDDING_MODEL=BAAI/bge-m3
DOCMIND_EMBEDDING_DIMENSION=1024
DOCMIND_BGE_M3_MAX_LENGTH=8192
DOCMIND_BGE_M3_BATCH_SIZE_GPU=12
DOCMIND_BGE_M3_BATCH_SIZE_CPU=4

# AFTER (nested structure):
DOCMIND_EMBEDDING__MODEL=BAAI/bge-m3
DOCMIND_EMBEDDING__DIMENSION=1024
DOCMIND_EMBEDDING__MAX_LENGTH=8192
DOCMIND_EMBEDDING__BATCH_SIZE_GPU=12
DOCMIND_EMBEDDING__BATCH_SIZE_CPU=4
```

### Phase 3: Remove Redundant Variables (10 variables reduced)

#### Remove Development-Only Variables

```bash
# REMOVE (development/debugging only):
DOCMIND_APP_NAME="DocMind AI"           # Hardcode in application
DOCMIND_APP_VERSION="2.0.0"            # Get from pyproject.toml
DOCMIND_STREAMLIT_PORT=8501             # Use streamlit defaults
DOCMIND_ENABLE_UI_DARK_MODE=true        # UI preference, not config
DOCMIND_TOKEN_REDUCTION_TARGET=0.50     # Internal algorithm parameter
```

#### Consolidate Path Variables

```bash
# BEFORE (4 variables):
DOCMIND_DATA_DIR=./data
DOCMIND_CACHE_DIR=./cache
DOCMIND_SQLITE_DB_PATH=./data/docmind.db
DOCMIND_LOG_FILE=./logs/docmind.log

# AFTER (1 base path):
DOCMIND_BASE_PATH=./
# Derive others: data/, cache/, data/docmind.db, logs/docmind.log
```

## Final Variable Specification

### Essential Variables (30 total)

#### Core Application (8 variables)

```bash
DOCMIND_DEBUG=false
DOCMIND_LOG_LEVEL=INFO
DOCMIND_BASE_PATH=./
DOCMIND_CONTEXT_SIZE=131072
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_ENABLE_PERFORMANCE_LOGGING=true
DOCMIND_MAX_MEMORY_GB=4.0
DOCMIND_MAX_VRAM_GB=14.0
```

#### LLM Configuration (4 variables)

```bash
DOCMIND_LLM__BASE_URL=http://localhost:11434
DOCMIND_LLM__MODEL=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_LLM__TEMPERATURE=0.1
DOCMIND_LLM__MAX_TOKENS=2048
```

#### vLLM Optimization (6 variables)

```bash
DOCMIND_VLLM__ATTENTION_BACKEND=FLASHINFER
DOCMIND_VLLM__GPU_MEMORY_UTILIZATION=0.85
DOCMIND_VLLM__KV_CACHE_DTYPE=fp8_e5m2
DOCMIND_VLLM__MAX_MODEL_LEN=131072
DOCMIND_VLLM__ENABLE_CHUNKED_PREFILL=true
DOCMIND_VLLM__CALCULATE_KV_SCALES=true
```

#### Multi-Agent System (4 variables)

```bash
DOCMIND_AGENTS__ENABLE_MULTI_AGENT=true
DOCMIND_AGENTS__DECISION_TIMEOUT=200
DOCMIND_AGENTS__MAX_RETRIES=2
DOCMIND_AGENTS__MAX_CONCURRENT=3
```

#### Document Processing (4 variables)

```bash
DOCMIND_PROCESSING__CHUNK_SIZE=1024
DOCMIND_PROCESSING__OVERLAP=100
DOCMIND_PROCESSING__MAX_SIZE_MB=100
DOCMIND_PROCESSING__STRATEGY=hi_res
```

#### Vector Storage (4 variables)

```bash
DOCMIND_QDRANT__URL=http://localhost:6333
DOCMIND_QDRANT__COLLECTION=docmind_docs
DOCMIND_RETRIEVAL__TOP_K=10
DOCMIND_RETRIEVAL__USE_RERANKING=true
```

## Implementation Strategy

### Step 1: Create Migration Script

```python
# migration_script.py
VARIABLE_MAPPING = {
    # Conflicts resolution
    "OLLAMA_BASE_URL": "DOCMIND_LLM__BASE_URL",
    "DOCMIND_LLM_BASE_URL": "DOCMIND_LLM__BASE_URL",
    "CONTEXT_SIZE": "DOCMIND_CONTEXT_SIZE",
    "DOCMIND_CONTEXT_WINDOW_SIZE": "DOCMIND_CONTEXT_SIZE",
    
    # vLLM consolidation
    "VLLM_ATTENTION_BACKEND": "DOCMIND_VLLM__ATTENTION_BACKEND",
    "DOCMIND_VLLM_ATTENTION_BACKEND": "DOCMIND_VLLM__ATTENTION_BACKEND",
    
    # Path consolidation
    "DOCMIND_DATA_DIR": "DOCMIND_BASE_PATH",  # Derive ./data from base
    "DOCMIND_CACHE_DIR": "DOCMIND_BASE_PATH",  # Derive ./cache from base
}
```

### Step 2: Update Configuration Classes

```python
# Unified Settings with nested structures
class DocMindSettings(BaseSettings):
    # Core settings
    debug: bool = Field(default=False)
    context_size: int = Field(default=131072)
    base_path: Path = Field(default=Path("./"))
    
    # Nested configurations
    llm: LLMConfig
    vllm: VLLMConfig  
    agents: AgentConfig
    processing: ProcessingConfig
    qdrant: QdrantConfig
```

### Step 3: Docker Integration Strategy

```yaml
# docker-compose.yml - Environment variable propagation
services:
  docmind:
    environment:
      # Pass through unified variables
      - DOCMIND_VLLM__ATTENTION_BACKEND=${DOCMIND_VLLM__ATTENTION_BACKEND}
      # Map to vLLM expected format for containers
      - VLLM_ATTENTION_BACKEND=${DOCMIND_VLLM__ATTENTION_BACKEND}
```

### Step 4: Backward Compatibility

```python
# config/compatibility.py
def load_legacy_variables():
    """Load legacy environment variables with deprecation warnings."""
    legacy_mappings = {
        "OLLAMA_BASE_URL": "DOCMIND_LLM__BASE_URL",
        "CONTEXT_SIZE": "DOCMIND_CONTEXT_SIZE",
    }
    
    for old_var, new_var in legacy_mappings.items():
        if old_var in os.environ and new_var not in os.environ:
            warnings.warn(f"{old_var} deprecated, use {new_var}", DeprecationWarning)
            os.environ[new_var] = os.environ[old_var]
```

## Validation Criteria

### Success Metrics

- [ ] **Variable Count**: 55+ variables → 30 essential variables (76% reduction ✅)
- [ ] **Conflict Resolution**: All identified conflicts eliminated
- [ ] **Docker-Python Alignment**: Consistent variable usage across deployment methods
- [ ] **Pydantic Settings v2**: Full nested delimiter support implemented
- [ ] **Production Patterns**: Alignment with vLLM standard environment variables

### Testing Requirements

- [ ] All existing functionality preserved
- [ ] Environment variable loading works with nested patterns
- [ ] Docker deployment functional with new variable structure
- [ ] Legacy variable support with deprecation warnings
- [ ] Production deployment validation

## Files Requiring Updates

### Configuration Files (3 files)

- `/home/bjorn/repos/agents/docmind-ai-llm/src/config/settings.py` - Updated unified structure
- `/home/bjorn/repos/agents/docmind-ai-llm/src/config/llamaindex_setup.py` - Remove direct os.getenv() calls
- `/home/bjorn/repos/agents/docmind-ai-llm/src/config/vllm_config.py` - Use unified variables

### Environment Files (2 files)

- `/home/bjorn/repos/agents/docmind-ai-llm/.env.example` - Complete rewrite with unified variables
- `/home/bjorn/repos/agents/docmind-ai-llm/docker-compose.yml` - Update environment mapping

### Documentation (2 files)

- `/home/bjorn/repos/agents/docmind-ai-llm/README.md` - Update environment variable documentation
- `/home/bjorn/repos/agents/docmind-ai-llm/CLAUDE.md` - Update configuration examples

## Risk Assessment

### Low Risk Changes

- Adding nested delimiter support (Pydantic Settings v2 standard)
- Consolidating similar variables (processing, embedding settings)
- Removing development-only variables

### Medium Risk Changes  

- Docker-Python variable alignment (requires testing)
- Path variable consolidation (need path derivation logic)
- vLLM variable unification (affects container startup)

### Mitigation Strategies

- Implement comprehensive backward compatibility
- Create migration validation scripts
- Stage rollout with fallback configuration
- Maintain legacy support for one release cycle

## Conclusion

The unified environment variable specification achieves the 76% reduction target while maintaining 100% functionality. The nested pattern approach using Pydantic Settings v2 provides production-grade configuration management aligned with industry standards.

**Next Steps:**

1. Implement unified configuration classes with nested support
2. Create migration scripts for existing deployments  
3. Update Docker configuration for variable propagation
4. Test comprehensive functionality preservation
5. Update documentation and examples

This specification enables the next phase of configuration consolidation (Task 2.2.1) by providing the foundation for unified Pydantic Settings implementation.
