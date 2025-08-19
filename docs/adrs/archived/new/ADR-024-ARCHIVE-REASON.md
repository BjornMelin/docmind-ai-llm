# ADR-024 Configuration Management - Archive Reason

## Archived Date
2025-08-18

## Reason for Archiving
**Architectural Conflicts and Over-Engineering**

ADR-024 was archived due to significant conflicts with established architectural decisions and violation of core design principles.

## Specific Conflicts

### 1. Violates ADR-016 (UI State Management)
- ADR-016 explicitly chose "Streamlit native st.session_state and st.cache_data directly without custom abstraction layers"
- ADR-016 removed 500+ lines of custom state management as "over-engineering"
- ADR-024 reintroduced 1000+ lines of custom configuration management

### 2. Conflicts with ADR-015 (Deployment Strategy)
- ADR-015 successfully uses simple environment variables: `DOCMIND_MODEL`, `DOCMIND_CONTEXT_LENGTH`, etc.
- ADR-024 wrapped these in unnecessary Pydantic BaseSettings abstractions

### 3. Violates ADR-013 (UI Architecture)
- ADR-013 chose "Native Streamlit Components without unnecessary libraries"
- ADR-024 added new dependencies (pydantic, python-decouple, watchdog) for functionality Streamlit already provides

## False Premises

### "Configuration Scattered Across ADRs"
**Reality**: Configuration is appropriately distributed:
- Most ADRs use library defaults (LlamaIndex, LangGraph, Qdrant)
- Simple environment variables handle runtime settings
- Streamlit native config handles UI settings
- Only 2 experimental features need simple boolean flags

### "Type Safety and Validation Needed"
**Reality**: For a local single-user desktop application:
- Environment variables are sufficient
- Library defaults provide appropriate validation
- Complex startup validation adds overhead without benefit

### "Hot Reloading Required"
**Reality**: 
- Streamlit auto-reloads on file changes
- Local app restarts in seconds
- Hot reloading adds unnecessary complexity

## Correct Approach

DocMind AI uses **distributed, simple configuration by design**:

```bash
# Runtime Configuration (ADR-015)
DOCMIND_MODEL=Qwen/Qwen3-14B
DOCMIND_CONTEXT_LENGTH=131072
DOCMIND_DEVICE=cuda

# Feature Flags
ENABLE_DSPY=false
ENABLE_GRAPHRAG=false

# Database
DATABASE_URL=sqlite:///data/docmind.db
QDRANT_URL=http://localhost:6333
```

Plus:
- `.streamlit/config.toml` for UI configuration (ADR-013)
- `st.session_state` for runtime state (ADR-016)
- Library defaults for components

## Design Philosophy

The existing approach aligns with project principles:
- **KISS over DRY**: Simple environment variables over complex abstractions
- **Library-first**: Use native Streamlit config, not custom wrappers
- **Avoid over-engineering**: Don't solve problems that don't exist

## Lesson Learned

Centralized configuration is appropriate for enterprise applications with:
- Multiple deployment environments
- Complex validation requirements
- Frequent configuration changes
- Multiple teams

For a local desktop RAG application, simple distributed configuration is more appropriate and maintainable.