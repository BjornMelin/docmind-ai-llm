# Contributing to DocMind AI

DocMind AI welcomes contributions from the community! This guide outlines how to contribute, including setting up the development environment, coding standards, and the contribution process.

## Getting Started

1. **Fork and Clone:**

   ```bash
   git clone https://github.com/<your-username>/docmind-ai.git
   cd docmind-ai
   ```

2. **Set Up Environment:**

   ```bash
   uv venv
   source .venv/bin/activate
   uv sync
   ```

   For tests or GPU support:

   ```bash
   uv sync --extra test
   uv sync --extra gpu
   ```

3. **Run Locally:**

   ```bash
   streamlit run app.py
   ```

## Coding Standards

- **Style:** Use Google-style docstrings, type hints, and max line length of 88.
- **Linting:** Run `ruff check .` and `ruff format .` (configured in `pyproject.toml`).
  - Selected rules: E, F, I, UP, N, S, B, A, C4, PT, SIM, TID, D.
  - Ignored: D203, D213, S301, S603, S607, S108.
- **Principles:** Follow KISS (Keep It Simple, Stupid), DRY (Don't Repeat Yourself), and YAGNI (You Aren't Gonna Need It).
- **Dependencies:** Prefer library-first solutions to minimize custom code.

## Contribution Workflow

1. **Create a Branch:**

   ```bash
   git checkout -b feat/<feature-name>  # or fix/<bug-name>
   ```

2. **Develop and Test:**
   - Write code in `app.py`, `utils.py`, or appropriate modules.
   - Add tests in `tests/` using pytest.
   - Ensure tests pass: `pytest`.
3. **Commit Changes:**
   - Use clear commit messages (e.g., "Add hybrid search with Jina v4 and SPLADE++").
   - Follow conventional commits if possible.
4. **Push and Create PR:**
   - Push: `git push origin feat/<feature-name>`.
   - Open a Pull Request on GitHub with a descriptive title and details linking to issues.
5. **Code Review:**
   - Address feedback promptly.
   - Ensure CI (ruff, pytest) passes.

## Testing

- Write unit tests for `utils.py` functions (e.g., document loading, vectorstore creation).
- Test integration flows in `app.py` (e.g., analysis, chat).
- Cover edge cases: large documents, GPU on/off, unsupported formats.
- Run: `pytest`.

## Deployment

- **Local:** Use `streamlit run app.py`.
- **Docker:** Build and run:

  ```bash
  docker-compose up --build
  ```

- Ensure environment variables are set (see `.env.example`).

## Reporting Issues

- Use GitHub Issues for bugs, feature requests, or questions.
- Provide detailed descriptions, including steps to reproduce and logs (`logs/app.log`).

## Community

- Join discussions on GitHub.
- Contact maintainers via issues or [LinkedIn](https://www.linkedin.com/in/bjorn-melin/).

Thank you for contributing to DocMind AI!
