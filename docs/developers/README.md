# Developer Documentation

This directory contains comprehensive documentation for DocMind AI developers. All guides follow consistent formatting with kebab-case naming conventions and structured content organization.

## Quick Navigation

### 🚀 Getting Started
- **[Developer Setup](developer-setup.md)** - Complete setup guide (30 min quickstart)
- **[Architecture Guide](architecture.md)** - System design and components
- **[Development Guide](development-guide.md)** - Coding standards and practices

### 📊 System Documentation
- **[Multi-Agent System](multi-agent-system.md)** - Agent coordination and orchestration
- **[GPU and Performance](gpu-and-performance.md)** - Hardware optimization guides
- **[Model Configuration](model-configuration.md)** - AI model setup and configuration

### 🔧 Implementation Guides  
- **[Retrieval & Search Implementation](retrieval-search-implementation.md)** - Search system details
- **[Testing Guide](testing.md)** - Testing strategies and frameworks
- **[Deployment Guide](deployment.md)** - Production deployment procedures

### 📋 Reports and Analysis
- **[Validation Report](validation-report.md)** - Comprehensive system validation
- **[Task 2.1.2 Deliverable](task-2-1-2-deliverable.md)** - Environment variable mapping deliverable
- **[Future Development](future-development.md)** - Roadmap and planned enhancements
- **[Lessons Learned](lessons-learned.md)** - Project insights and best practices
- **[Maintenance Procedures](maintenance-procedures.md)** - Operational maintenance

## Documentation Structure

Each guide follows this standard structure:

- **Purpose**: What this document covers
- **Audience**: Who should read this
- **Prerequisites**: Required knowledge or setup
- **Step-by-step instructions**: Clear, actionable content
- **Examples**: Runnable code snippets
- **Troubleshooting**: Common issues and solutions

## Key Architectural Principles

### Unified Configuration Architecture (ADR-024)
- **Single Source of Truth**: `from src.config import settings`
- **76% Complexity Reduction**: Simplified from legacy over-engineered approach
- **Pydantic Settings V2**: Nested models with full validation
- **DOCMIND_ Prefix**: Consistent environment variable naming

### Development Standards
- **KISS Principle**: Simplicity over complexity
- **Library-First**: Use established solutions over custom code
- **Type Safety**: Full type hints and Pydantic validation
- **Testing**: Comprehensive test coverage with three-tier strategy

## Quick Reference

### Essential Commands
```bash
# Setup
uv sync && streamlit run src/app.py

# Testing  
pytest tests/unit/ -v                    # Fast unit tests
pytest tests/integration/ -v             # Cross-component tests
python scripts/run_tests.py              # Full test suite

# Code Quality
ruff format . && ruff check . --fix      # Format and lint
python scripts/performance_validation.py # Performance check
```

### Configuration Pattern
```python
# Always use this pattern
from src.config import settings

# Access any configuration
model_name = settings.vllm.model
embedding_model = settings.embedding.model_name
chunk_size = settings.processing.chunk_size
```

### Key File Locations
```
src/config/settings.py     # All configuration
src/models/                # Pydantic models  
src/agents/coordinator.py  # Multi-agent system
src/retrieval/             # Search and retrieval
docs/adrs/                 # Architectural decisions
```

## Getting Help

1. **Setup Issues**: See [Developer Setup](developer-setup.md) troubleshooting
2. **Architecture Questions**: Check [Architecture Guide](architecture.md)
3. **Development Problems**: Review [Development Guide](development-guide.md)
4. **Performance Issues**: Consult [GPU and Performance](gpu-and-performance.md)
5. **ADR References**: Check [../adrs/](../adrs/) for architectural decisions

## Contributing

1. **Read**: [Developer Setup](developer-setup.md) → [Development Guide](development-guide.md)
2. **Understand**: Review relevant [ADRs](../adrs/) for architectural context
3. **Follow**: Unified configuration patterns and coding standards
4. **Test**: Comprehensive testing with quality checks
5. **Document**: Update relevant guides for significant changes

## Documentation Standards

- **Format**: Markdown with proper heading hierarchy
- **Naming**: kebab-case for all files
- **Structure**: Table of contents for documents >50 lines
- **Code**: Include imports and ensure examples are runnable
- **Diagrams**: Use Mermaid for system flows and architecture
- **Links**: Use relative paths and maintain cross-references

---

**Welcome to DocMind AI development!**

This documentation represents a production-ready system with 95% ADR compliance and excellent code quality scores. The unified architecture approach ensures you only need to learn one pattern: `from src.config import settings`.

Start with [Developer Setup](developer-setup.md) to get running in 30 minutes.