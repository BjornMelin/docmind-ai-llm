---
ADR: 048
Title: Docs Consistency Pass — Align Specs and Handbook With Code
Status: Implemented
Version: 1.1
Date: 2026-07-11
Supersedes:
Superseded-by:
Related: 027, 024, 029
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
- B: Manual fixes + existing documentation gates (Selected)
- C: Add a general inline source-path scanner and allowlist
- D: Move drifting docs into archived folders (only when content is truly historical)

### Decision Framework (≥9.0)

Complexity is weighted highest (40%) to reflect the maintenance burden on a small team; lower effort and minimal CI/tooling overhead are prioritized over novel automation.

> **Complexity** refers to development effort required (lower effort = higher score).

| Option | Complexity (40%) | Perf (30%) | Alignment (30%) | Total | Decision |
| --- | --- | --- | --- | --- | --- |
| B: Manual fixes + existing gates | 10 | 10 | 9 | **9.7** | Selected |
| C: General source-path scanner | 7 | 10 | 8 | 8.1 | Rejected |
| D: Archive | 6 | 10 | 7 | 7.6 | Rejected |
| A: Do nothing | 10 | 10 | 0 | 7.0 | Rejected |

Scoring notes (brief):

- **B**: reuses link, schema, structural-parity, and Markdown gates without a
  second suppression system.
- **C**: inline source paths appear in historical context, deletion records, and
  conceptual examples, so a general scanner would require ambiguous allowlists.
- **D**: higher effort to triage/archive content (6), but perf unaffected (10).
- **A**: trivial effort (10) but fails alignment (0) because drift remains.

## Decision

1. Fix concrete drift in specs and developer docs to match the shipped v1 code and APIs.

2. Keep the implemented documentation gates as the automated contract:

- `scripts/check_links.py` validates internal Markdown links.
- `scripts/verify_structural_parity.py` validates the documented top-level
  `src/` package layout.
- `scripts/validate_schemas.py` validates repository schemas.
- Markdownlint validates active Markdown syntax and structure.

Exact inline code paths remain a review responsibility because active design
records may intentionally describe deleted paths. There is no general inline
source-path scanner or suppression file.

## Consequences

### Positive Outcomes

- Docs become a reliable source of truth again.
- Link, schema, top-level structure, and Markdown drift are caught in CI.

### Trade-offs

- Inline code-path accuracy still requires focused review when source owners move.
- Historical and conceptual path references do not require a suppression file.
