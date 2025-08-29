# DocMind AI Documentation Overview

Welcome to the documentation for **DocMind AI**, a local LLM document analysis system using vLLM, LangGraph, and LlamaIndex with 100-160 tok/s decode performance on RTX 4090. This documentation is organized into sections for users, developers, API reference, and architectural decision records (ADRs).

## Project Summary

DocMind AI is an offline-first document analysis system featuring a 5-agent LangGraph supervisor coordination system with the Qwen3-4B-Instruct-2507-FP8 model. Key features include 128K context windows, FP8 optimization achieving 100-160 tok/s decode performance, hybrid vector search, multi-agent reasoning, and complete privacy through local-only processing on RTX 4090 hardware.

## Documentation Structure

- **[Product Requirements Document (PRD)](PRD.md)**: Complete system requirements with validated performance specifications
- **[User Documentation](user/)**:
  - [Getting Started](user/getting-started.md): Installation and setup instructions
  - [GPU Requirements](user/gpu-requirements.md): GPU hardware requirements and setup guide
  - [Multi-Agent Coordination Guide](user/multi-agent-coordination-guide.md): Guide to the 5-agent system
  - [Usage Guide](user/usage-guide.md): Detailed guide for using the app's features
  - [Troubleshooting](user/troubleshooting.md): Solutions for common user issues
- **[Developer Documentation](developers/)**:
  - [Setup](developers/setup.md): Development environment setup
  - [Architecture](developers/architecture.md): High-level architecture overview
  - [LangGraph Supervisor Architecture](developers/langgraph-supervisor-architecture.md): Multi-agent system architecture
  - [vLLM Integration Guide](developers/vllm-integration-guide.md): Complete vLLM setup and optimization
  - [Multi-Agent Performance Tuning](developers/multi-agent-performance-tuning.md): Performance optimization guide
  - [Qwen3 FP8 Configuration](developers/qwen3-fp8-configuration.md): Model-specific configuration guide
  - [GPU Setup](developers/gpu-setup.md): GPU requirements and configuration
  - [Performance Validation](developers/performance-validation.md): Performance testing and validation
  - [Contributing](developers/contributing.md): Guidelines for contributing code and documentation
  - [Testing](developers/testing.md): Instructions for writing and running tests
  - [Deployment](developers/deployment.md): Guide for local, Docker, and production deployments
- **[API Documentation](api/)**:
  - [Multi-Agent API](api/multi-agent-api.md): Complete API reference for the multi-agent system
- **[Architectural Decision Records (ADRs)](adrs/)**:
  - [Architecture Overview](adrs/ARCHITECTURE-OVERVIEW.md): Complete system architecture
  - [ADR-011: Agent Orchestration Framework](adrs/ADR-011-agent-orchestration-framework.md): LangGraph supervisor system
  - [ADR-004: Local-First LLM Strategy](adrs/ADR-004-local-first-llm-strategy.md): vLLM and local model strategy
  - [ADR-010: Performance Optimization Strategy](adrs/ADR-010-performance-optimization-strategy.md): FP8 and performance optimization
  - [ADR-021: Chat Memory & Context Management](adrs/ADR-021-chat-memory-context-management.md): 128K context window management
  - [Additional ADRs](adrs/): Complete collection of architectural decisions

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for contribution guidelines, which redirect to [developers/contributing.md](developers/contributing.md) for detailed instructions.

## License

The project is licensed under MIT—see [LICENSE](../LICENSE).

## Support

For issues or questions, open an issue on [GitHub](https://github.com/BjornMelin/docmind-ai) or check [user/troubleshooting.md](user/troubleshooting.md).
