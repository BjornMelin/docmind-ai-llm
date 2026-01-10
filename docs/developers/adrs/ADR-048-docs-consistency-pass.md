---
ADR: 048
Title: Docs Consistency Pass — Align Specs/Handbook With Code and Add Drift Checks
Status: Proposed
Version: 1.0
Date: 2026-01-09
Supersedes:
Superseded-by:
Related: 027, 024
Tags: docs, maintainability, release
---

## Description

Perform a targeted documentation consistency pass to ensure the repository can ship v1 with trustworthy specs and developer docs.

## Context

The repo contains multiple drift points (examples):

- specs referencing removed files (e.g., `src/processing/document_processor.py`)
- handbook examples referencing placeholder APIs (`src.utils.document.process_document`)
- changelog referencing ADR numbers that did not exist (partially addressed by adding ADR-039/040)

This erodes trust and makes maintenance harder.

## Alternatives

- A: Do nothing (ship with drift; unacceptable)
- B: Manual doc fixes only (Selected baseline)
- C: Manual fixes + add a lightweight automated drift check (Selected)
- D: Move drifting docs into archived folders (only when content is truly historical)

### Decision Framework (≥9.0)

> **Complexity** refers to development + CI/tooling overhead (higher score = simpler solution).

| Option                            | Complexity (40%) | Perf (30%) | Alignment (30%) |   Total |
| --------------------------------- | ---------------: | ---------: | --------------: | ------: |
| **C: Manual fixes + drift check** |                9 |         10 |               9 | **9.3** |
| B: Manual only                    |               10 |         10 |               7 |     8.9 |
| D: Archive                        |                6 |         10 |               7 |     7.6 |
| A: Do nothing                     |               10 |         10 |               0 |     7.0 |

## Decision

1. Fix concrete drift in specs and developer docs to match the shipped v1 code and APIs.

2. Add a lightweight automated check in CI/quality gates to prevent regressions:

- Scan non-archived docs for referenced `src/…` paths and assert files exist
- Keep allowlist/suppressions minimal:
  - **Allowed**: intentional examples showing patterns (not real paths), archived ADRs/specs (`/superseded/`), and historical changelogs
  - **Not allowed**: paths to deleted files in active docs; use `# REMOVED` comments or update text
- Suppression file: `scripts/doc_drift_allowlist.txt` (one path per line)

## Consequences

### Positive Outcomes

- Docs become a reliable source of truth again.
- Future drift is caught early in CI.

### Trade-offs

- The drift check may require a small allowlist for intentional examples/archived references.
