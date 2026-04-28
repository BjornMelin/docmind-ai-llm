---
ADR: 065
Title: Python 3.12-Compatible Security Baseline
Status: Accepted
Version: 1.0
Date: 2026-04-28
Supersedes: ADR-064
Superseded-by:
Related: ADR-042, ADR-063, ADR-064
Tags: python, packaging, security, compatibility
References:
  - https://www.python.org/downloads/release/python-31213/
  - https://devguide.python.org/versions/
---

## Description

Standardize the repository on a Python 3.12-compatible runtime baseline with
CPython 3.12.13 as the primary development and CI version.

## Context

The Dependabot security remediation exposed a real package compatibility
conflict under Python 3.13: `llama-index-embeddings-fastembed` declares
support for Python `<3.13`, while the project still needs the LlamaIndex
FastEmbed integration for retrieval work. Keeping a Python 3.13-only baseline
would require dependency graph contortions or feature removal unrelated to the
security alerts.

## Decision

1) Set `requires-python = ">=3.12,<3.14"`.

2) Set the primary local, CI, and container runtime to CPython 3.12.13.

3) Set Ruff and Pyright to target Python 3.12.

4) Keep Python 3.13 in the supported range only when dependencies declare
compatible metadata.

## Consequences

### Positive outcomes

- Security updates can resolve without hacky Python 3.13-only workarounds.
- The FastEmbed/LlamaIndex integration remains available.
- The runtime baseline matches the dependency ecosystem that the project
actually uses today.

### Trade-offs

- Python 3.13-only modernization is deferred until the dependency graph fully
supports it.
- Tooling cannot rely on Python 3.13-only syntax or stdlib additions.
