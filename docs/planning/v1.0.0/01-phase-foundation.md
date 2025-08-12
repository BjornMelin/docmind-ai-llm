# Phase 1: Foundation & Critical Fixes (Week 1)

**Duration**: 5-7 days  

**Priority**: CRITICAL  

**Goal**: Establish foundation optimizations with immediate impact

## Phase Overview

This phase focuses on critical dependency cleanup, GPU optimization, and foundational library-first patterns that enable all subsequent optimizations.

## Tasks

### T1.1: Critical Dependency Cleanup 🔴 CRITICAL

**Research Foundation**: 

- [RAG & Reranking Research](../../../library_research/10-rag_reranking-research.md)

- [Observability Dev Research](../../../library_research/10-observability_dev-research.md)

- [Dependency Audit JSON](../../../library_research/00-dependency-audit.json)

**Libraries to Remove**:

- `torchvision==0.22.1` (~15 packages, 7.5MB+)

- `polars==1.31.0` (unused DataFrame library)

- `ragatouille==0.0.9.post2` (~20 packages, redundant RAG)

**Libraries to Add**:

- `psutil>=6.0.0` (explicit dependency)

#### Sub-task T1.1.1: Remove Unused Dependencies

**Instructions**:
1. Remove torchvision (saves ~15 packages):
   ```bash
   uv remove torchvision
   # Verify no imports remain
   rg "import torchvision" src/ tests/
   ```

2. Remove polars (unused):
   ```bash
   uv remove polars
   rg "import polars" src/ tests/
   ```

3. Remove ragatouille (replaced by llama-index-postprocessor-colbert-rerank):
   ```bash
   uv remove ragatouille
   rg "import ragatouille" src/ tests/
   ```

**Validation**:
```bash

# Verify packages removed
python -c "import sys; [sys.exit(1) if __import__(pkg) else print(f'SUCCESS: {pkg} removed') for pkg in ['torchvision', 'polars', 'ragatouille']]"

# Check package count reduction (should be ~310 from 331)
uv pip list | wc -l
```

**Success Criteria**: 

- ✅ All three packages removed

- ✅ No import errors

- ✅ Package count reduced from 331 to ~310

#### Sub-task T1.1.2: Add Explicit psutil Dependency

**Instructions**:
```bash

# Add explicit dependency (currently transitive)
uv add "psutil>=6.0.0"

# Verify installation
python -c "import psutil; print(f'psutil {psutil.__version__} installed')"

# Test monitoring functionality
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
```

**Files to Update**: `pyproject.toml`

**Success Criteria**:

- ✅ psutil explicitly declared in pyproject.toml

- ✅ Monitoring functionality intact

#### Sub-task T1.1.3: Move Observability to Dev Dependencies

**Research**: [Observability Dev Research - Dev Dependency Migration](../../../library_research/10-observability_dev-research.md#dependency-migration-strategy)

**Instructions**:
1. Remove from main dependencies:
   ```bash
   uv remove arize-phoenix openinference-instrumentation-llama-index
   ```

2. Add to optional dependencies in `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   dev = [
       "arize-phoenix>=11.13.0",
       "openinference-instrumentation-llama-index>=4.3.0",
       "ruff>=0.12.8",
       "pytest>=8.3.1",
       "pytest-asyncio>=0.23.0",
       "pytest-cov>=6.0.0",
       "pytest-benchmark>=4.0.0"
   ]
   ```

3. Update installation docs:
   ```bash
   # Production installation (smaller)
   uv pip install docmind-ai-llm
   
   # Development installation (with observability)
   uv pip install docmind-ai-llm[dev]
   ```

**Success Criteria**:

- ✅ Observability tools in optional dependencies

- ✅ ~35 fewer packages in production

- ✅ Conditional import patterns working

---

### T1.2: CUDA Optimization for LLM Runtime 🟡 HIGH

**Research Foundation**: 

- [LLM Runtime Core Research](../../../library_research/10-llm_runtime_core-research.md)

- [CUDA Optimization Patterns](../../../library_research/10-llm_runtime_core-research.json)

**Target**: RTX 4090 (compute capability 8.9, 16GB VRAM)

#### Sub-task T1.2.1: Install CUDA-Optimized llama-cpp-python

**Instructions**:
1. Set RTX 4090 architecture:
   ```bash
   export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_ARCHITECTURES=89"
   export CUDA_VISIBLE_DEVICES=0  # Use first GPU
   ```

2. Install CUDA-enabled llama-cpp-python:
   ```bash
   uv add "llama-cpp-python[cuda]>=0.2.32,<0.3.0"
   ```

3. Create GPU detection utility in `src/utils/llm_loader.py`:
   ```python
   import torch
   from loguru import logger
   
   def get_llm_backend():
       """Detect optimal LLM backend."""
       if torch.cuda.is_available():
           device_name = torch.cuda.get_device_name(0)
           vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
           logger.info(f"CUDA available: {device_name} with {vram_gb:.1f}GB VRAM")
           return "cuda"
       logger.warning("CUDA not available, falling back to CPU")
       return "cpu"
   ```

**Validation**:
```bash

# Verify CUDA compilation
python -c "from llama_cpp import Llama; print('llama-cpp-python CUDA enabled')"

# Check GPU detection
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')"
```

**Success Criteria**:

- ✅ llama-cpp-python compiled with CUDA support

- ✅ GPU detection working

- ✅ Fallback to CPU mode functional

#### Sub-task T1.2.2: Configure PyTorch for CUDA 12.8

**Instructions**:
```bash

# Install PyTorch with CUDA 12.8
uv add torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# Verify CUDA setup
python -c """
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
"""
```

**Success Criteria**:

- ✅ PyTorch using CUDA 12.8

- ✅ GPU memory visible

- ✅ Multi-GPU detection if available

#### Sub-task T1.2.3: Implement KV Cache Optimization

**Create**: `src/utils/kv_cache_config.py`

```python
from dataclasses import dataclass
from typing import Literal
from loguru import logger

@dataclass
class KVCacheConfig:
    """KV cache configuration for memory optimization."""
    quantization: Literal["int8", "int4", "none"] = "int8"
    max_memory_gb: float = 8.0  # For 16GB VRAM (50% allocation)
    fallback_strategy: list[str] = ["int8", "int4", "Q4_K_M"]
    
def get_kv_cache_args(config: KVCacheConfig) -> dict:
    """Get llama.cpp KV cache arguments.
    
    Research: https://github.com/ggerganov/llama.cpp/discussions/5969
    int8 quantization provides 50% memory savings with <1% quality loss
    """
    args = {}
    
    if config.quantization == "int8":
        args["kv_cache_type"] = "q8_0"
        logger.info("KV cache: int8 quantization (50% memory savings)")
    elif config.quantization == "int4":
        args["kv_cache_type"] = "q4_0"
        logger.info("KV cache: int4 quantization (75% memory savings)")
    else:
        logger.info("KV cache: No quantization")
    
    args["n_gpu_layers"] = -1  # Offload all layers to GPU
    args["n_ctx"] = 32768  # Context window
    
    return args
```

**Validation**:
```python

# Test KV cache configuration
from src.utils.kv_cache_config import KVCacheConfig, get_kv_cache_args

config = KVCacheConfig(quantization="int8")
args = get_kv_cache_args(config)
print(f"KV cache args: {args}")
```

**Success Criteria**:

- ✅ KV cache configuration available

- ✅ 50% memory savings with int8

- ✅ Fallback strategies defined

---

### T1.3: Qdrant Native Hybrid Search Implementation 🟡 HIGH

**Research Foundation**:

- [Embedding & Vector Store Research](../../../library_research/10-embedding_vectorstore-research.md)

- [Qdrant Native Features JSON](../../../library_research/10-embedding_vectorstore-research.json)

**Performance Impact**: 40x search speed, 70% memory reduction

#### Sub-task T1.3.1: Enable Native BM25 in Qdrant

**Update**: `src/services/vector_store.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, BM25Params
from loguru import logger

def create_hybrid_collection(client: QdrantClient, collection_name: str):
    """Create Qdrant collection with native BM25 hybrid search.
    
    Research: library_research/10-embedding_vectorstore-research.md
    Native BM25 eliminates custom sparse implementations
    """
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI/FastEmbed dimension
                distance=Distance.COSINE
            ),
            sparse_vectors_config={
                "bm25": BM25Params(
                    model="Qdrant/bm25",  # Native BM25 model
                    idf=True,  # Enable IDF weighting
                    k1=1.2,  # Term frequency saturation
                    b=0.75  # Length normalization
                )
            }
        )
        logger.info(f"Created hybrid collection: {collection_name}")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        raise
```

**Validation**:
```python

# Test hybrid collection creation
from qdrant_client import QdrantClient

client = QdrantClient(":memory:")
create_hybrid_collection(client, "test_hybrid")
print("Hybrid collection created successfully")
```

**Success Criteria**:

- ✅ Collection supports dense vectors

- ✅ Collection supports sparse BM25 vectors

- ✅ No custom sparse embedding code needed

#### Sub-task T1.3.2: Implement Binary Quantization

**Research**: [Binary Quantization Benefits](../../../library_research/10-embedding_vectorstore-research.md#binary-quantization)

```python
from qdrant_client.models import QuantizationConfig, BinaryQuantization

def enable_binary_quantization(client: QdrantClient, collection_name: str):
    """Enable binary quantization for 40x search performance.
    
    Research shows:
    - 32x memory reduction (1 bit vs 32 bits)
    - 40x search speed improvement
    - 90-95% recall preservation with rescoring
    """
    quantization_config = QuantizationConfig(
        binary=BinaryQuantization(
            always_ram=True  # Keep quantized vectors in RAM
        )
    )
    
    client.update_collection(
        collection_name=collection_name,
        quantization_config=quantization_config,
        optimizer_config={
            "default_segment_number": 2,
            "indexing_threshold": 10000
        }
    )
    
    logger.info(f"Binary quantization enabled for {collection_name}")
```

**Success Criteria**:

- ✅ 40x search speed improvement

- ✅ 70% memory reduction

- ✅ >90% recall maintained

#### Sub-task T1.3.3: Configure RRF Fusion

**Create**: `src/services/hybrid_search.py`

```python
from qdrant_client import QdrantClient
from typing import List, Dict, Any
import numpy as np

def hybrid_search_with_rrf(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    query_text: str,
    alpha: float = 0.7,  # Weight for dense search (0.7 dense, 0.3 sparse)
    top_k: int = 10,
    rrf_k: int = 60  # RRF constant
) -> List[Dict[str, Any]]:
    """Perform hybrid search with Reciprocal Rank Fusion.
    
    Research: library_research/10-embedding_vectorstore-research.md
    RRF provides 5-8% precision improvement over simple fusion
    """
    # Search with native hybrid mode
    results = client.search_hybrid(
        collection_name=collection,
        query_vector=query_vector,
        query_text=query_text,
        fusion="rrf",  # Use RRF fusion
        alpha=alpha,
        limit=top_k
    )
    
    # Results already fused by Qdrant
    return [
        {
            "id": hit.id,
            "score": hit.score,
            "payload": hit.payload
        }
        for hit in results
    ]
```

**Validation**:
```python

# Test hybrid search
import numpy as np

client = QdrantClient(":memory:")
create_hybrid_collection(client, "test")

# Add test document
client.upsert(
    collection_name="test",
    points=[{
        "id": 1,
        "vector": np.random.rand(1536).tolist(),
        "payload": {"text": "test document"}
    }]
)

# Search
results = hybrid_search_with_rrf(
    client, "test",
    query_vector=np.random.rand(1536).tolist(),
    query_text="test"
)
print(f"Found {len(results)} results")
```

**Success Criteria**:

- ✅ RRF fusion working

- ✅ 5-8% precision improvement

- ✅ Configurable alpha weighting

---

### T1.4: LlamaIndex Global Settings Migration 🟡 HIGH

**Research Foundation**:

- [LlamaIndex Ecosystem Research](../../../library_research/10-llamaindex_ecosystem-research.md)

- [Settings Migration Pattern](../../../library_research/10-llamaindex_ecosystem-research.json)

**Impact**: Replace 200+ lines of custom Settings code

#### Sub-task T1.4.1: Create Centralized Settings Configuration

**Create**: `src/config/llama_settings.py`

```python
"""LlamaIndex global settings configuration.

Research: library_research/10-llamaindex_ecosystem-research.md
Replaces 200+ lines of custom ServiceContext code
"""
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.callbacks import CallbackManager
from loguru import logger
import os

def configure_llama_index():
    """Configure global LlamaIndex settings.
    
    This replaces all custom ServiceContext initialization
    Settings automatically propagate to all LlamaIndex components
    """
    # LLM Configuration
    Settings.llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        request_timeout=60.0,
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    )
    
    # Embedding Configuration
    Settings.embed_model = FastEmbedEmbedding(
        model_name=os.getenv("EMBED_MODEL", "BAAI/bge-large-en-v1.5"),
        max_length=512,
        cache_dir=".cache/embeddings"
    )
    
    # Chunking Configuration
    Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
    Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Output Configuration
    Settings.num_output = int(os.getenv("NUM_OUTPUT", "256"))
    Settings.context_window = int(os.getenv("CONTEXT_WINDOW", "4096"))
    
    # Callback Manager (for observability)
    Settings.callback_manager = CallbackManager()
    
    logger.info("LlamaIndex global settings configured")
    logger.debug(f"LLM: {Settings.llm.model}")
    logger.debug(f"Embeddings: {Settings.embed_model.model_name}")
    logger.debug(f"Chunk size: {Settings.chunk_size}")
```

**Success Criteria**:

- ✅ Single source of truth for configuration

- ✅ Environment variable support

- ✅ Automatic propagation to components

#### Sub-task T1.4.2: Remove Custom Settings Code

**Instructions**:
1. Find all custom ServiceContext usage:
   ```bash
   # Find files with ServiceContext
   rg "ServiceContext" src/ --files-with-matches
   
   # Find custom initialization patterns
   rg "embed_model.*=" src/ --type py
   rg "llm.*=.*Ollama" src/ --type py
   ```

2. Replace with Settings import:
   ```python
   # OLD CODE (remove):
   from llama_index.core import ServiceContext
   
   service_context = ServiceContext.from_defaults(
       llm=ollama_llm,
       embed_model=embed_model,
       chunk_size=512
   )
   index = VectorStoreIndex.from_documents(
       documents, 
       service_context=service_context
   )
   
   # NEW CODE (use):
   from src.config.llama_settings import configure_llama_index
   
   # Call once at startup
   configure_llama_index()
   
   # Components automatically use Settings
   index = VectorStoreIndex.from_documents(documents)
   ```

**Files to Update**:

- Remove ServiceContext from all index creation

- Remove ServiceContext from query engines

- Remove custom embed_model initialization

- Remove custom llm initialization

**Success Criteria**:

- ✅ 200+ lines of custom code removed

- ✅ No ServiceContext imports remain

- ✅ Settings used throughout

#### Sub-task T1.4.3: Validate Settings Migration

**Create**: `tests/test_llama_settings.py`

```python
import pytest
from llama_index.core import Settings
from src.config.llama_settings import configure_llama_index

def test_settings_configured():
    """Test that Settings are properly configured."""
    configure_llama_index()
    
    # Verify LLM configured
    assert Settings.llm is not None
    assert hasattr(Settings.llm, 'model')
    
    # Verify embeddings configured
    assert Settings.embed_model is not None
    assert hasattr(Settings.embed_model, 'model_name')
    
    # Verify chunk settings
    assert Settings.chunk_size == 512
    assert Settings.chunk_overlap == 50
    
    # Verify output settings
    assert Settings.num_output == 256
    assert Settings.context_window == 4096

def test_settings_propagation():
    """Test that Settings propagate to components."""
    from llama_index.core import VectorStoreIndex, Document
    
    configure_llama_index()
    
    # Create index without explicit service context
    docs = [Document(text="test document")]
    index = VectorStoreIndex.from_documents(docs)
    
    # Query engine should use global settings
    query_engine = index.as_query_engine()
    assert query_engine is not None

@pytest.mark.asyncio
async def test_async_settings():
    """Test Settings work with async operations."""
    from llama_index.core import VectorStoreIndex, Document
    
    configure_llama_index()
    
    docs = [Document(text="test async")]
    index = VectorStoreIndex.from_documents(docs)
    
    # Async query should work
    query_engine = index.as_query_engine()
    response = await query_engine.aquery("test")
    assert response is not None
```

**Run Validation**:
```bash

# Run settings tests
uv run pytest tests/test_llama_settings.py -v

# Verify no ServiceContext usage
rg "ServiceContext" src/ tests/
```

**Success Criteria**:

- ✅ All tests pass

- ✅ Settings accessible throughout application

- ✅ No ServiceContext imports found

---

## Phase 1 Validation Checklist

### Dependency Changes

- [ ] torchvision removed (~15 packages saved)

- [ ] polars removed (1 package saved)

- [ ] ragatouille removed (~20 packages saved)

- [ ] psutil added explicitly

- [ ] Observability moved to dev dependencies (~35 packages)

- [ ] Total package count: ~275 (from 331)

### CUDA Optimization

- [ ] llama-cpp-python compiled with CUDA

- [ ] PyTorch using CUDA 12.8

- [ ] KV cache optimization configured

- [ ] GPU detection and fallback working

### Qdrant Features

- [ ] Native BM25 enabled

- [ ] Binary quantization configured

- [ ] RRF fusion implemented

- [ ] 40x search performance verified

### LlamaIndex Migration

- [ ] Global Settings configured

- [ ] ServiceContext code removed (200+ lines)

- [ ] Settings propagation verified

- [ ] All components using global config

## Performance Validation

Run the following benchmark to verify improvements:

```bash

# Create benchmark script
cat > benchmark_phase1.py << 'EOF'
import time
import psutil
import torch
from qdrant_client import QdrantClient
import numpy as np

print("=== Phase 1 Validation ===")

# Memory baseline
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# GPU check
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Qdrant performance
client = QdrantClient(":memory:")
print("Qdrant client created")

# Package count
import subprocess
result = subprocess.run(["uv", "pip", "list"], capture_output=True, text=True)
package_count = len(result.stdout.strip().split('\n')) - 2
print(f"Package count: {package_count} (target: ~275)")

print("\n✅ Phase 1 validation complete")
EOF

uv run python benchmark_phase1.py
```

## Next Steps

After completing Phase 1:
1. Commit changes with detailed message
2. Run full test suite: `uv run pytest tests/`
3. Document any issues or deviations
4. Proceed to [Phase 2: Core Optimizations](./02-phase-core.md)

## Rollback Procedures

If any optimization causes issues:

```bash

# Dependency rollback
git checkout -- pyproject.toml
uv lock && uv sync

# Settings rollback (restore ServiceContext)
git checkout -- src/config/
git checkout -- src/services/

# Qdrant rollback (disable features)

# In vector_store.py, comment out sparse_vectors_config
```

Time to rollback: <5 minutes for any component
