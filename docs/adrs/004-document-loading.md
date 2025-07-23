# ADR 004: Document Loading Strategy

## Status

Accepted

## Context

DocMind AI must support a wide range of document formats for analysis, ensuring efficient loading and multimodal capabilities (e.g., text and images from PDFs).

## Decision

- Use lightweight, specialized loaders:
  - **PyMuPDF**: PDF and EPUB (text + image extraction).
  - **python-docx**: DOCX, ODT, RTF, PPTX.
  - **Polars**: XLSX, CSV, XML, JSON, MD.
  - **extract-msg**: MSG (email).
  - **TextLoader (LangChain)**: TXT, code files (PY, JS, etc.).
- Process in `utils.py:load_documents()` with temporary files for Streamlit uploads.
- Store metadata (source, images) in LangChain Document objects.

## Rationale

- Specialized loaders are lightweight and reliable compared to all-in-one solutions.
- PyMuPDF enables multimodal (image) support for PDFs.
- Polars handles structured data efficiently.
- Temporary files ensure Streamlit compatibility without permanent storage.

## Alternatives Considered

- Unstructured.io: Too heavy, cloud dependencies.
- Apache Tika: Complex setup, slower for local use.

## Consequences

- Pros: Broad format support, fast, multimodal-ready.
- Cons: Temporary files require cleanup; handled via `os.remove()`.
