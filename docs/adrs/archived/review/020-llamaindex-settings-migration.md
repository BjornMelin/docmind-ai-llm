# ADR-020: LlamaIndex Native Settings Migration

## Title

Migration from Pydantic Settings to LlamaIndex Native Settings

## Version/Date

1.0 / August 12, 2025

## Status

Integrated into ADR-021 (LlamaIndex Native Architecture Consolidation)

## Context

DocMind AI uses pydantic-settings (223 lines in `/src/models/core.py`) violating KISS principles through abstraction layers between configuration and LlamaIndex usage. Research shows 87% code reduction possible (150 lines → 20 lines) with native LlamaIndex Settings providing direct ecosystem integration.

## Related Requirements

- KISS principle compliance (simplicity first)

- Library-first architecture (leverage LlamaIndex ecosystem)

- 1-week deployment target

- Integration with existing 77-line ReActAgent architecture

## Alternatives

- **Keep Pydantic-Settings**: Score 0.605, KISS 0.30 (abstraction complexity)

- **Hybrid Approach**: Increases complexity, violates KISS further

- **LlamaIndex Native Settings**: Score 0.7875, KISS 0.90 (87% code reduction) ✅ **SELECTED**

## Decision

**Migrate to LlamaIndex native Settings** for configuration management, eliminating pydantic-settings dependency and abstraction layers. Decision score: 0.7875 based on KISS compliance (0.90), integration quality (0.95), validation trade-off (0.50), migration risk (0.60).

## Related Decisions

- ADR-001 (Architecture Overview): Updates Settings reference

- ADR-015 (LlamaIndex Migration): Pure ecosystem adoption

- ADR-018 (Library-First Refactoring): Continues simplification pattern

## Design

**Implementation**: Direct LlamaIndex Settings assignment from environment variables.

```python

# Before: Pydantic-settings complexity (150 lines)
class Settings(BaseSettings):
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o-mini")
    # ... 30+ configuration fields with validators

# After: LlamaIndex native simplicity (20 lines)
from llama_index.core import Settings

Settings.llm = OpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY")
)
Settings.embed_model = OpenAIEmbedding(
    model=os.getenv("EMBED_MODEL", "text-embedding-3-small")
)
Settings.chunk_size = int(os.getenv("CHUNK_SIZE", "512"))
Settings.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "20"))
```

**Integration**: Simple SettingsLoader validates environment variables and configures Settings globally for all LlamaIndex components.

**Testing**: Validate environment loading, configuration validation, ReActAgent integration, and performance (no regression from current startup time).

## Consequences

### Positive Outcomes

- **87% code reduction** (150 → 20 lines)

- **KISS compliance** improved from 0.30 to 0.90

- **Single configuration system** vs dual pydantic + LlamaIndex

- **Eliminated abstraction layers** and dependency burden

- **Lazy loading** and global Settings propagation

- **Native ecosystem integration** with familiar patterns

### Trade-offs

- Manual validation required (acceptable for simplicity gain)

- Basic environment variable handling (sufficient for current needs)

## Migration Timeline

**Phase 1 (Days 1-3)**: Replace pydantic-settings with direct Settings usage, implement SettingsLoader

**Phase 2 (Days 4-5)**: Testing, validation, performance benchmarking with ReActAgent

**Implementation**: 1 week (August 12-19, 2025)

**Risk Level**: Low (phased approach with rollback capability)

**Success Metrics**: 87% code reduction, KISS 0.90 compliance, no performance regression

## Changelog

**1.0 (August 12, 2025)**: Initial migration decision based on research showing 87% code reduction opportunity. Aligns with ADR-018 library-first refactoring success and completes pure LlamaIndex ecosystem adoption from ADR-015.
