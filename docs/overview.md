# DocMind AI Documentation Overview

Welcome to the documentation for **DocMind AI**, a local-first document analysis system. This documentation is organized into sections for users, developers, API reference, and architectural decision records (ADRs).

## Project Summary

DocMind AI is an offline-first document analysis system featuring a 5-agent LangGraph supervisor coordination system, hybrid retrieval, and local processing for privacy.

## Documentation Structure

- **[Product Requirements Document (PRD)](PRD.md)**: Complete system requirements with validated performance specifications
- **[User Documentation](user/)**:
  - [Getting Started](user/getting-started.md): Installation and setup
  - [Configuration](user/configuration.md): Basic settings and examples
  - [Troubleshooting & FAQ](user/troubleshooting-faq.md): Common issues and answers
- **[Developer Documentation](developers/)**:
  - [Getting Started](developers/getting-started.md): Development setup and first run
  - [Architecture Overview](developers/architecture-overview.md): High-level system view
  - [System Architecture](developers/system-architecture.md): Technical diagrams and flows
  - [Cache Implementation](developers/cache.md): Cache wiring and operations
  - [Configuration Reference](developers/configuration-reference.md): Developer configuration details
  - [CI/CD Pipeline](developers/ci-cd-pipeline.md): GitHub Actions pipeline (3.11 + 3.10)
  - [Operations Guide](developers/operations-guide.md): Deployment and operations
  - [Developer Handbook](developers/developer-handbook.md): Patterns and workflows
  - [Testing Guide](testing/testing-guide.md): How to write and run tests
- **[API Documentation](api/)**:
  - [API Reference](api/api.md): REST and Python API with examples
- **Architectural Decision Records (ADRs)**:
  - [ADR Index](developers/adrs/): Complete collection of architectural decisions

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines.

## License

The project is licensed under MIT—see [LICENSE](../LICENSE).

## Support

For issues or questions, open an issue on [GitHub](https://github.com/BjornMelin/docmind-ai) or check [user/troubleshooting-faq.md](user/troubleshooting-faq.md).
