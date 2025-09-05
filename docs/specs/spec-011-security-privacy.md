---
spec: SPEC-011
title: Security & Privacy: Offline-First, Egress Controls, License Notes
version: 1.0.0
date: 2025-09-05
owners: ["ai-arch"]
status: Final
related_requirements:
  - NFR-SEC-001: Network egress SHALL be off by default.
  - NFR-SEC-002: Providers and endpoints SHALL be allowlisted.
  - NFR-COMP-001: Licenses SHALL be documented for embedded models.
related_adrs: ["ADR-031","ADR-024"]
---


## Objective

Ensure offline by default, with explicit allowlists for endpoints. Document licenses for packaged weights and OCR/PDF libs.

## File Operations

### UPDATE

- `src/config/settings.py`: add allowlist list and enforcement checks.
- `README.md`: license notes section.

## Acceptance Criteria

```gherkin
Feature: Egress control
  Scenario: Unknown endpoint
    Given a non-allowlisted base_url
    Then the app SHALL refuse to connect and show a clear error
```
