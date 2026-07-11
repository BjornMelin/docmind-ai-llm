# Support policy

DocMind AI is a community-maintained, local-first open-source project. Support
is best effort and has no response-time or resolution-time guarantee.

## Supported versions

The newest minor release in the current major line receives fixes. Pre-1.0
releases remain supported only until 1.0.0 is published; after that, upgrade to
the latest 1.x release before reporting a bug.

DocMind requires Python `>=3.12,<3.14`. CPython 3.12.13 is the CI and container
baseline. CPython 3.13.12 has a lean compatibility lane in CI. Use the
checked-in `uv.lock` or the repository container image for reproducible
installations.

## Get help

1. Check the [troubleshooting guide](docs/user/troubleshooting-faq.md).
2. Search [existing issues](https://github.com/BjornMelin/docmind-ai-llm/issues).
3. Open an issue with reproduction steps, the DocMind version, operating system,
   Python version, and sanitized logs.

Do not include documents, prompts, API keys, credentials, private endpoints, or
other sensitive data. Follow [SECURITY.md](SECURITY.md) for vulnerabilities.
