---
ADR: 065
Title: Python 3.12-compatible security baseline
Status: Accepted
Version: 1.1
Date: 2026-07-10
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

The April 2026 Dependabot remediation exposed a Python 3.13 conflict in the
then-installed LlamaIndex FastEmbed adapter. That adapter has since been
removed. The project now uses direct FastEmbed support, required LlamaIndex
core, and selected adapters while retaining the Python 3.12-compatible
baseline for the current dependency graph.

## Decision

1. Set `requires-python = ">=3.12,<3.14"`.

2. Set the primary local, CI, and container runtime to CPython 3.12.13.

3. Set Ruff and Pyright to target Python 3.12.

4. Keep Python 3.13 in the supported range when dependencies declare compatible
   metadata.

## Consequences

### Positive outcomes

- Security updates resolve without Python 3.13-only workarounds.
- Direct FastEmbed and selected LlamaIndex integrations remain available.
- The runtime baseline matches the current dependency graph.

### Trade-offs

- Python 3.13-only modernization is deferred until the dependency graph fully
supports it.
- Tooling cannot rely on Python 3.13-only syntax or stdlib additions.

## Amendment

The July 2026 package hard cut removed the LlamaIndex FastEmbed adapter and the
`llama-index` meta-package. The original conflict remains historical decision
context, not a current dependency requirement.
