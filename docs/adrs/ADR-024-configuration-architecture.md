# ADR-024: Unified Settings Architecture

## Metadata

**Status:** Approved  
**Version/Date:** v2.0 / 2025-08-24

## Title

Unified Pydantic BaseSettings Configuration with LlamaIndex Native Integration

## Description

Adopts a unified configuration approach using simple Pydantic BaseSettings for app-specific settings and LlamaIndex's native Settings for LLM/embedding configuration, achieving 95% complexity reduction while maintaining full functionality and industry standard compliance.

## Context

Research has definitively identified massive over-engineering in DocMind AI's configuration management. Current monolithic approach uses 737 lines of complex configuration where production systems use 27-80 lines total. Analysis of real-world LlamaIndex projects, Pydantic best practices, and successful LLM frameworks reveals a path to 95% reduction in configuration complexity while retaining all functionality.

**Current Issues**:

- Monolithic 737-line Settings class with unnecessary complexity
- Reinventing functionality already provided by LlamaIndex Settings
- Over-engineered nested models and facade patterns
- Poor framework integration requiring custom implementations
- Maintenance burden from custom validators and complex inheritance

**Research Evidence**:

- **Real-world validation**: Production LlamaIndex systems (sec-insights) handle complete RAG configuration in 27 lines
- **Framework analysis**: LlamaIndex Settings natively handles 90%+ of our configuration needs
- **Industry patterns**: Successful systems use simple BaseSettings + framework integration
- **Complexity analysis**: Current system is 15-27x more complex than production equivalents

**Key Research Finding**: *"Why didn't we just do this from the start?"* - The classic simplification that experienced developers recognize as obviously correct.

## Decision Drivers

- **Library-First Principle**: Leverage LlamaIndex Settings for 90% of configuration needs
- **Complexity Reduction**: Eliminate unnecessary abstraction and custom implementations
- **Industry Standards**: Follow 12-factor app methodology and proven patterns
- **Maintainability**: Reduce cognitive load through simplicity and framework alignment
- **Backward Compatibility**: Maintain existing functionality with simpler implementation

## Alternatives

- **A: Status Quo (Complex Modular)** — Pros: Already implemented / Cons: 737 lines of over-engineering, poor framework integration
- **B: Complete Rewrite** — Pros: Clean slate / Cons: High risk, unnecessary when simple refactor works
- **C: Gradual Simplification** — Pros: Lower risk / Cons: Maintains complexity longer than needed

### Decision Framework

| Option | Solution Leverage (40%) | Application Value (25%) | Maintenance Load (25%) | Adaptability (10%) | Total Score | Decision |
| ---------------------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------- | ----------- | ------------- |
| **Unified Settings** | 0.95 | 0.90 | 0.95 | 0.85 | **0.93** | ✅ **Selected** |
| Status Quo (Complex) | 0.20 | 0.70 | 0.10 | 0.60 | 0.35 | Rejected |
| Complete Rewrite | 0.60 | 0.80 | 0.30 | 0.80 | 0.58 | Rejected |
| Gradual Simplification | 0.70 | 0.75 | 0.60 | 0.70 | 0.69 | Rejected |

## Decision

We will adopt **Unified Settings Architecture** to replace the over-engineered configuration system. This involves using **simple Pydantic BaseSettings** for app-specific configuration and **LlamaIndex native Settings** for LLM/embedding configuration. This decision supersedes the complex modular approach and achieves 95% complexity reduction.

## High-Level Architecture

```mermaid
graph TD
    A[DocMindSettings] --> B[App-specific Config]
    A --> C[Environment Variables]
    
    D[setup_llamaindex()] --> E[LlamaIndex Settings]
    E --> F[LLM Configuration]
    E --> G[Embedding Configuration] 
    E --> H[Document Processing]
    E --> I[Chunking Parameters]
    
    J[Application] --> A
    J --> E
    
    K[.env File] --> C
    K --> L[Framework Integration]
    L --> E
```

## Related Requirements

### Functional Requirements

- **FR-1:** Maintain all existing configuration functionality with simplified implementation
- **FR-2:** Enable environment variable configuration following 12-factor app principles
- **FR-3:** Support multi-agent coordination settings without complex nested structures

### Non-Functional Requirements

- **NFR-1:** **(Maintainability)** Reduce configuration complexity by 95% (737 lines → ~80 lines)
- **NFR-2:** **(Framework Integration)** Leverage LlamaIndex Settings for native compatibility
- **NFR-3:** **(Standards Compliance)** Follow industry standard configuration patterns

### Performance Requirements

- **PR-1:** Configuration loading time must remain negligible (< 1ms at startup)
- **PR-2:** Memory usage reduction through elimination of complex object hierarchies

### Integration Requirements

- **IR-1:** Direct integration with LlamaIndex Settings singleton
- **IR-2:** Seamless environment variable loading via Pydantic BaseSettings
- **IR-3:** Maintain compatibility with existing multi-agent coordination

## Related Decisions

- **ADR-001** (Modern Agentic RAG Architecture): Multi-agent system requiring streamlined configuration
- **ADR-010** (Performance Optimization Strategy): Performance benefits from simplified configuration
- **ADR-011** (Agent Orchestration Framework): Supervisor pattern with simplified settings management

## Design

### Architecture Overview

The unified configuration system provides minimal app-specific settings via Pydantic BaseSettings while leveraging LlamaIndex's native Settings for all LLM/embedding configuration.

### Implementation Details

**In `src/config/settings.py`:**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class DocMindSettings(BaseSettings):
    """Minimal app-specific settings. LlamaIndex Settings handles LLM/embedding config."""
    
    # Core Application Settings
    app_name: str = "DocMind AI"
    version: str = "0.1.0"
    debug: bool = False
    
    # Multi-Agent Configuration
    enable_multi_agent: bool = True
    max_agent_retries: int = 2
    agent_decision_timeout: int = 300
    
    # Storage Paths
    data_dir: Path = Path("./data")
    cache_dir: Path = Path("./cache")
    qdrant_storage_path: Path = Path("./qdrant_storage")
    
    # UI Settings
    streamlit_port: int = 8501
    enable_gpu_acceleration: bool = True
    
    # Performance Settings
    max_concurrent_agents: int = 5
    chunk_batch_size: int = 100
    
    model_config = SettingsConfigDict(
        env_prefix="DOCMIND_",
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

# Global settings instance
settings = DocMindSettings()
```

**In `src/config/llamaindex_setup.py`:**

```python
import os
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

def setup_llamaindex() -> None:
    """Configure LlamaIndex Settings. Call once at app startup."""
    
    # LLM Configuration
    Settings.llm = Ollama(
        model=os.getenv("DOCMIND_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507-FP8"),
        base_url=os.getenv("DOCMIND_LLM_BASE_URL", "http://localhost:11434"),
        temperature=float(os.getenv("DOCMIND_TEMPERATURE", "0.1")),
        request_timeout=float(os.getenv("DOCMIND_REQUEST_TIMEOUT", "120.0"))
    )
    
    # Embedding Configuration
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("DOCMIND_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        cache_folder=str(Path("./embeddings_cache").resolve())
    )
    
    # Document Processing Configuration
    Settings.chunk_size = int(os.getenv("DOCMIND_CHUNK_SIZE", "1024"))
    Settings.chunk_overlap = int(os.getenv("DOCMIND_CHUNK_OVERLAP", "200"))
    
    Settings.node_parser = SentenceSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap
    )
    
    # Context Window Configuration
    Settings.context_window = int(os.getenv("DOCMIND_CONTEXT_WINDOW_SIZE", "131072"))
    Settings.num_output = int(os.getenv("DOCMIND_MAX_TOKENS", "2048"))

def get_settings():
    """Get current DocMind settings instance."""
    return settings
```

### Configuration

**Environment variable schema (.env):**

```env
# Core Application
DOCMIND_APP_NAME="DocMind AI"
DOCMIND_DEBUG=false
DOCMIND_STREAMLIT_PORT=8501

# Multi-Agent System
DOCMIND_ENABLE_MULTI_AGENT=true
DOCMIND_MAX_AGENT_RETRIES=2
DOCMIND_AGENT_DECISION_TIMEOUT=300
DOCMIND_MAX_CONCURRENT_AGENTS=5

# Storage Paths
DOCMIND_DATA_DIR=./data
DOCMIND_CACHE_DIR=./cache
DOCMIND_QDRANT_STORAGE_PATH=./qdrant_storage

# LLM Configuration (read by LlamaIndex setup)
DOCMIND_MODEL_NAME=Qwen/Qwen3-4B-Instruct-2507-FP8
DOCMIND_LLM_BASE_URL=http://localhost:11434
DOCMIND_TEMPERATURE=0.1
DOCMIND_CONTEXT_WINDOW_SIZE=131072
DOCMIND_MAX_TOKENS=2048

# Embedding Configuration
DOCMIND_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
DOCMIND_CHUNK_SIZE=1024
DOCMIND_CHUNK_OVERLAP=200

# Performance Settings
DOCMIND_ENABLE_GPU_ACCELERATION=true
DOCMIND_CHUNK_BATCH_SIZE=100
DOCMIND_REQUEST_TIMEOUT=120.0
```

**Application Integration:**

```python
# src/app.py startup
from src.config.settings import settings
from src.config.llamaindex_setup import setup_llamaindex

def initialize_app():
    """Initialize application with unified configuration."""
    # Setup LlamaIndex Settings
    setup_llamaindex()
    
    # App-specific configuration available via settings
    if settings.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    return settings
```

## Testing

**In `tests/config/test_unified_config.py`:**

```python
import pytest
import os
from src.config.settings import DocMindSettings, settings
from src.config.llamaindex_setup import setup_llamaindex
from llama_index.core import Settings

def test_docmind_settings_loading():
    """Verify DocMind settings load correctly with environment variables."""
    test_settings = DocMindSettings()
    
    assert test_settings.app_name == "DocMind AI"
    assert test_settings.enable_multi_agent is True
    assert test_settings.agent_decision_timeout == 300
    assert test_settings.streamlit_port == 8501

def test_environment_variable_override():
    """Verify environment variables override default settings."""
    os.environ["DOCMIND_DEBUG"] = "true"
    os.environ["DOCMIND_MAX_AGENT_RETRIES"] = "5"
    
    test_settings = DocMindSettings()
    
    assert test_settings.debug is True
    assert test_settings.max_agent_retries == 5
    
    # Cleanup
    del os.environ["DOCMIND_DEBUG"]
    del os.environ["DOCMIND_MAX_AGENT_RETRIES"]

def test_llamaindex_settings_integration():
    """Verify LlamaIndex Settings are properly configured."""
    setup_llamaindex()
    
    assert Settings.llm is not None
    assert Settings.embed_model is not None
    assert Settings.chunk_size > 0
    assert Settings.chunk_overlap >= 0

@pytest.mark.asyncio
async def test_agent_configuration_access():
    """Verify multi-agent settings are accessible."""
    assert settings.enable_multi_agent is True
    assert settings.max_concurrent_agents > 0
    assert settings.agent_decision_timeout > 0

def test_configuration_simplicity():
    """Verify configuration complexity is minimized."""
    # Simple validation that we don't have over-engineered patterns
    assert hasattr(settings, 'enable_multi_agent')
    assert not hasattr(settings, 'agents')  # No nested config objects
    assert not hasattr(settings, 'llm')     # LLM config handled by LlamaIndex
```

## Consequences

### Positive Outcomes

- **Massive Complexity Reduction**: 95% reduction from 737 lines to ~80 lines total configuration
- **Framework Alignment**: Native integration with LlamaIndex Settings eliminates custom implementations
- **Industry Standard Compliance**: Follows 12-factor app methodology and proven patterns
- **Maintainability**: Simple, flat configuration structure that any developer can understand
- **Performance**: Eliminated object hierarchy overhead and complex validation chains
- **Backward Compatibility**: All existing functionality preserved with simpler implementation

### Negative Consequences / Trade-offs

- **Migration Effort**: Requires updating imports and configuration access patterns across codebase
- **Learning Curve**: Team needs to understand LlamaIndex Settings patterns (minimal learning required)
- **Framework Dependency**: Relies on LlamaIndex Settings singleton (standard practice in framework)

### Ongoing Maintenance & Considerations

- **Environment Variables**: Monitor usage patterns and add new variables as needed
- **LlamaIndex Updates**: Track LlamaIndex releases for Settings API changes
- **Configuration Validation**: Ensure new settings follow simple, flat pattern
- **Documentation Updates**: Maintain clear examples of configuration patterns

### Dependencies

- **System**: Python 3.10+, existing LlamaIndex installation
- **Python**: `pydantic-settings>=2.0.0`, `llama-index-core>=0.12.0`
- **Removed**: Complex nested configuration models, custom validation logic

## References

- [Streamlined Settings Research Report](../../ai-research/2025-08-23/006-streamlined-settings-research.md) - Comprehensive analysis showing 95% complexity reduction opportunity
- [Production System Analysis](https://github.com/run-llama/sec-insights) - Real-world LlamaIndex configuration patterns (27 lines)
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) - Industry standard configuration management
- [12-Factor App Methodology](https://12factor.net/config) - Configuration best practices followed in this implementation
- [LlamaIndex Settings Documentation](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings/) - Framework-native configuration patterns
- [ADR-001: Modern Agentic RAG Architecture](ADR-001-modern-agentic-rag-architecture.md) - Multi-agent system requiring streamlined configuration
- [ADR-011: Agent Orchestration Framework](ADR-011-agent-orchestration-framework.md) - Supervisor pattern with simplified settings

## Changelog

- **v2.0 (2025-08-24)**: Complete replacement of over-engineered approach with unified architecture. Based on comprehensive research showing 95% complexity reduction opportunity. Adopted simple Pydantic BaseSettings + LlamaIndex Settings integration following industry standards and real-world production patterns.
