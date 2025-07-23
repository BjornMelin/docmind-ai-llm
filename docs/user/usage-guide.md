# DocMind AI Usage Guide

## UI Overview

- **Sidebar:** Model selection, backend, GPU toggle, context size, theme.
- **Main Section:** Upload, previews, analysis options, results, chat.

## Uploading Documents

- Supported formats: PDF (with image previews), DOCX, TXT, etc.
- Multiple files allowed; previews expand for snippets/images.

## Configuring Analysis

- **Prompt:** Predefined or custom.
- **Tone/Instructions/Length:** Customize LLM style/output.
- **Mode:** Separate or combined analysis.
- **Advanced:** Chunked/late chunking, multi-vector embeddings.

## Viewing Results

- Structured: Summary, insights, actions, questions.
- Export: JSON/Markdown.
- Raw output if parsing fails.

## Chat Interface

- Ask questions; uses hybrid RAG for context.
- History preserved in session.

## Best Practices

- Use GPU for speed.
- Start with small docs; enable chunking for large ones.
- Save/load sessions for persistence.

Troubleshooting: Check logs in `logs/app.log`; ensure Ollama runs.
