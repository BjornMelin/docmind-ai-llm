# ADR 006: Document Analysis Pipeline

## Version/Date

v1.0 / July 22, 2025

## Status

Accepted

## Context

The analysis pipeline must generate structured outputs (summaries, insights, actions, questions) with customizable prompts and handle large documents.

## Decision

- Use **LangChain LLMChain** with **PydanticOutputParser** for structured output (AnalysisOutput model).
- Support customizable prompts, tones, instructions, and length via `prompts.py`.
- Handle large documents via chunking (RecursiveCharacterTextSplitter) or map-reduce summarization (`load_summarize_chain`).
- Implement in `utils.py:analyze_documents()` with error handling and raw output fallback.

## Rationale

- LLMChain simplifies prompt management and execution.
- Pydantic ensures consistent, structured outputs.
- Chunking/map-reduce scales to large documents.
- Fallbacks improve reliability.

## Alternatives Considered

- Custom parsing: Error-prone, less maintainable.
- No chunking: Fails for large docs.

## Consequences

- Pros: Structured, scalable, user-configurable.
- Cons: Complex prompt logic; simplified via predefined options.
