# Testing DocMind AI

This guide details how to write, run, and maintain tests for DocMind AI to ensure reliability and quality.

## Testing Framework

- **Tool**: pytest v8.3.1 (installed via `uv sync --extra test`).

- **Location**: Tests in `tests/` directory.

- **Coverage**: Aim for >80% coverage on `utils.py` and `app.py`.

## Setup

1. Install test dependencies:

   ```bash
   uv sync --extra test
   ```

2. Run tests:

   ```bash
   pytest
   ```

## Performance Validation

For comprehensive performance benchmarks and validation results, see [performance-validation.md](./performance-validation.md).

## Writing Tests

- **Unit Tests**:
  - Target: Functions in `utils.py` (e.g., `load_documents`, `create_vectorstore`).
  - Example:

    ```python
    def test_load_pdf(tmp_path):
        pdf_path = tmp_path / "test.pdf"
        with open(pdf_path, "w") as f:
            f.write("%PDF-1.4 dummy content")
        docs = load_documents([pdf_path])
        assert len(docs) > 0
        assert isinstance(docs[0], Document)
    ```

  - Focus: Edge cases (empty files, unsupported formats, large docs).

- **Integration Tests**:
  - Target: End-to-end flows in `app.py` (e.g., upload → analysis → chat).
  - Example:

    ```python
    def test_analysis_pipeline(monkeypatch):
        monkeypatch.setattr("utils.load_documents", lambda x: [Document(page_content="test")])
        result = analyze_documents(mock_llm, ["test"], "Comprehensive", "", "Neutral", "General Assistant", "Concise", 4096)
        assert isinstance(result, AnalysisOutput)
    ```

- **Edge Cases**:
  - Large documents (>10MB).
  - GPU on/off scenarios.
  - Invalid model configs.
  - Missing Ollama server.

## Running Tests

- All tests: `pytest`.

- Specific file: `pytest tests/test_utils.py`.

- Coverage: `pytest --cov=docmind_ai --cov-report=html`.

## Best Practices

- Mock external dependencies (e.g., Ollama, Qdrant) using `unittest.mock`.

- Test error handling (e.g., file errors, parsing failures).

- Update tests for new features; maintain DRY with fixtures.

## CI Integration

- GitHub Actions runs `pytest` and `ruff check` on PRs.

- Ensure tests pass before submitting PRs.

Report issues or suggest test cases via GitHub.
