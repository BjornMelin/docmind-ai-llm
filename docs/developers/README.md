# DocMind AI Developer Documentation

> **Streamlined developer documentation following industry best practices with the Divio documentation framework**

This directory contains documentation for DocMind AI developers. The documentation has been organized into **5 core guides** plus specialized references, eliminating redundancy while preserving all essential technical content.

## Quick Navigation

### Core Documentation (5 Essential Guides)

| Guide                                                     | Purpose                                                          | Audience                                 | Time to Read |
| --------------------------------------------------------- | ---------------------------------------------------------------- | ---------------------------------------- | ------------ |
| **[Getting Started](getting-started.md)**                 | Complete 30-minute onboarding from zero to productive            | New developers                           | 30 min       |
| **[System Architecture](system-architecture.md)**         | Deep understanding of multi-agent coordination and system design | Developers needing architectural context | 45 min       |
| **[Developer Handbook](developer-handbook.md)**           | Practical implementation guidance, testing, and maintenance      | Active developers building features      | 60 min       |
| **[Configuration Guide](configuration.md)**               | Complete configuration guide with GPU optimization               | DevOps, Developers, Admins               | 30 min       |
| **[Operations Guide](operations-guide.md)**               | Production deployment and operational procedures                 | DevOps, SRE, production teams            | 45 min       |

### Specialized References

| Guide                                                            | Purpose                                                            | Audience                                    |
| ---------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------- |
| **[Architecture Overview](architecture-overview.md)**            | Executive technical summary with performance specs                 | Technical leads, architects                 |
| **[Cache Implementation Guide](cache.md)**                       | Wiring, configuration, operations, troubleshooting                 | Developers integrating cache                |
| **[GraphRAG Adapters Guide](guides/graphrag-adapters.md)**       | Adapter protocol, registry wiring, and optional dependency lanes   | Developers extending GraphRAG               |
| **[Multimodal Pipeline Guide](guides/multimodal-pipeline.md)**   | End-to-end PDF images → retrieval → UI → persistence               | Developers/operators shipping multimodal    |
| **[Testing Guide](../testing/testing-guide.md)**                 | Canonical testing strategy and commands                            | QA engineers, developers writing tests      |
| **[CI/CD Pipeline](ci-cd-pipeline.md)**                          | Continuous integration and deployment workflows                    | DevOps engineers, release managers          |

### Documentation Framework

Following the **Divio Documentation System** for optimal developer experience:

- **Tutorial** ([Getting Started](getting-started.md)) - Learning-oriented guidance
- **How-to Guides** ([Developer Handbook](developer-handbook.md)) - Problem-solving oriented
- **Reference** ([Configuration Guide](configuration.md)) - Information-oriented reference and setup guide
- **Explanation** ([System Architecture](system-architecture.md)) - Understanding-oriented
- **Operations** ([Operations Guide](operations-guide.md)) - Production-oriented

## User Journey Paths

### New Developer Onboarding

1. **Start Here**: [Getting Started](getting-started.md) - 30-minute setup
2. **Understand the System**: [System Architecture](system-architecture.md) - Core concepts
3. **Learn Development Practices**: [Developer Handbook](developer-handbook.md) - Implementation patterns
4. **Configure for Your Environment**: [Configuration Guide](configuration.md) - Optimization

### Experienced Developer Quick Access

1. **Implementation Guidance**: [Developer Handbook](developer-handbook.md) - Patterns and practices
2. **Architecture Reference**: [System Architecture](system-architecture.md) - System design
3. **Configuration Tuning**: [Configuration Guide](configuration.md) - Performance optimization
4. **Cache Wiring**: [Cache Implementation](cache.md) - Wiring and operations

### DevOps/Production Teams

1. **Deployment Procedures**: [Operations Guide](operations-guide.md) - Production deployment
2. **Configuration Management**: [Configuration Guide](configuration.md) - Environment setup
3. **Performance Optimization**: [Operations Guide](operations-guide.md) - Production tuning

## Key Architectural Principles

### Unified Configuration Architecture (ADR-024)

- **Single Source of Truth**: `from src.config import settings`
- **76% Complexity Reduction**: Simplified from legacy over-engineered approach
- **Pydantic Settings V2**: Nested models with full validation
- **DOCMIND\_ Prefix**: Consistent environment variable naming

### Development Standards

- **KISS Principle**: Simplicity over complexity
- **Library-First**: Use established solutions over custom code
- **Type Safety**: Full type hints and Pydantic validation
- **Testing**: Comprehensive test coverage with three-tier strategy

## Quick Reference

### Essential Commands

```bash
# Setup (detailed in Getting Started guide)
uv sync && uv run streamlit run app.py

# Testing (see Testing Guide)
uv run python scripts/run_tests.py                # Full test suite
uv run python scripts/run_tests.py --unit         # Unit tests only
uv run python scripts/run_tests.py --integration  # Integration tests

# Code Quality (detailed in Developer Handbook)
ruff format . && ruff check . --fix      # Format and lint
uv run python scripts/performance_monitor.py --run-tests --check-regressions # Performance check
```

### Configuration Pattern

```python
# Always use this pattern (detailed in Configuration Guide)
from src.config import settings

# Access any configuration
model_name = settings.vllm.model
embedding_model = settings.embedding.model_name
chunk_size = settings.processing.chunk_size
```

## Getting Help

| Issue Type                 | Primary Guide                                         | Secondary Resources                                              |
| -------------------------- | ----------------------------------------------------- | ---------------------------------------------------------------- |
| **Setup Problems**         | [Getting Started](getting-started.md)                 | [Configuration Guide](configuration.md)                          |
| **Architecture Questions** | [System Architecture](system-architecture.md)         | [Architecture Overview](architecture-overview.md), [ADRs](adrs/) |
| **Implementation Help**    | [Developer Handbook](developer-handbook.md)           | [Cache Implementation](cache.md)                                 |
| **Performance Problems**   | [Operations Guide](operations-guide.md)               | [Configuration Guide](configuration.md)                          |
| **Production Deployment**  | [Operations Guide](operations-guide.md)               | [Configuration Guide](configuration.md)                          |
| **Testing Issues**         | [Developer Handbook](developer-handbook.md)           | [Testing Guide](../testing/testing-guide.md)                     |

## Contributing

1. **Read**: [Getting Started](getting-started.md) → [Developer Handbook](developer-handbook.md)
2. **Understand**: [System Architecture](system-architecture.md) + relevant [ADRs](adrs/)
3. **Configure**: [Configuration Guide](configuration.md) for optimal development
4. **Deploy**: [Operations Guide](operations-guide.md) for production considerations

## Architecture Decision Records (ADRs)

All architectural decisions are documented in the [adrs/](adrs/) directory. Key ADRs include:

- **[ADR-001](adrs/ADR-001-modern-agentic-rag-architecture.md)**: Modern Agentic RAG Architecture
- **[ADR-024](adrs/ADR-024-configuration-architecture.md)**: Configuration Architecture (Unified Settings)
- **[ADR-011](adrs/ADR-011-agent-orchestration-framework.md)**: Agent Orchestration Framework
- **[All ADRs](adrs/)**: Complete list of 26 architectural decisions

## Documentation Organization

**Structure**: This documentation follows a **5 core guides + specialized references** approach:

| Category                       | Files                                                   | Purpose             |
| ------------------------------ | ------------------------------------------------------- | ------------------- |
| **Core Guides** (5)            | Essential documentation covering 90% of developer needs | Primary navigation  |
| **Specialized References** (4) | Deep-dive topics for specific use cases                 | Secondary resources |
| **ADRs** (26)                  | Architectural decisions and technical rationale         | Reference material  |

### Migration Notes

If you're looking for content from the previous documentation structure:

- **Setup/Installation** → [Getting Started](getting-started.md)
- **Architecture/Multi-Agent** → [System Architecture](system-architecture.md) + [Architecture Overview](architecture-overview.md)
- **Development/Testing** → [Developer Handbook](developer-handbook.md) + [Testing Guide](../testing/testing-guide.md)
- **Environment/Model Config** → [Configuration Guide](configuration.md)
- **Deployment/Performance** → [Operations Guide](operations-guide.md)
- **Cache/Components** → [Cache Implementation](cache.md)

## Documentation Standards

- **Format**: Markdown with proper heading hierarchy
- **Framework**: Divio documentation system (Tutorial, How-to, Reference, Explanation)
- **Structure**: Clear table of contents and user journey optimization
- **Code**: Include imports and ensure examples are runnable
- **Diagrams**: Use Mermaid for system flows and architecture
- **Links**: Maintain cross-references between consolidated guides

---

**Welcome to DocMind AI development!**

This documentation represents a production-ready system with 95% ADR compliance and excellent code quality scores. The unified architecture approach ensures you only need to learn one pattern: `from src.config import settings`.

**Start your journey**: [Getting Started Guide](getting-started.md) (30 minutes to productive)
