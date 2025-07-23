# DocMind AI Documentation Overview

Welcome to the documentation for **DocMind AI**, a local LLM-powered document analysis tool built with Streamlit, LangChain, and Ollama. This documentation is organized into sections for users, developers, and architectural decision records (ADRs).

## Project Summary

DocMind AI enables privacy-focused analysis of various document formats using local large language models. Key features include customizable prompts, hybrid search with Jina v4 and FastEmbed, GPU optimization, and interactive chat for document queries.

## Documentation Structure

- **[Product Requirements Document (PRD)](PRD.md)**: Defines functional and non-functional requirements, scope, and timeline.
- **[User Documentation](user/)**:
  - [Getting Started](user/getting-started.md): Installation and setup instructions.
  - [Usage Guide](user/usage-guide.md): Detailed guide for using the app’s features.
  - [Troubleshooting](user/troubleshooting.md): Solutions for common user issues.
- **[Developer Documentation](developers/)**:
  - [Setup](developers/setup.md): Development environment setup.
  - [Architecture](developers/architecture.md): High-level architecture overview.
  - [Contributing](developers/contributing.md): Guidelines for contributing code and documentation.
  - [LangChain Usage](developers/langchain-usage.md): Details on LangChain integration.
  - [Testing](developers/testing.md): Instructions for writing and running tests.
  - [Deployment](developers/deployment.md): Guide for local, Docker, and production deployments.
- **[Architectural Decision Records (ADRs)](adrs/)**:
  - [001: Architecture Overview](adrs/001-architecture-overview.md)
  - [002: Embedding Choices](adrs/002-embedding-choices.md)
  - [003: GPU Optimization](adrs/003-gpu-optimization.md)
  - [004: Document Loading](adrs/004-document-loading.md)
  - [005: Text Splitting](adrs/005-text-splitting.md)
  - [006: Analysis Pipeline](adrs/006-analysis-pipeline.md)
  - [007: Reranking Strategy](adrs/007-reranking-strategy.md)
  - [008: Session Persistence](adrs/008-session-persistence.md)
  - [009: UI Framework](adrs/009-ui-framework.md)
  - [010: LangChain Integration](adrs/010-langchain-integration.md)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines, which redirect to [developers/contributing.md](developers/contributing.md) for detailed instructions.

## License

The project is licensed under MIT—see [LICENSE](../LICENSE).

## Support

For issues or questions, open an issue on [GitHub](https://github.com/BjornMelin/docmind-ai) or check [user/troubleshooting.md](user/troubleshooting.md).
