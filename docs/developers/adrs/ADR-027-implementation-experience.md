---
ADR: 027
Title: Implementation Experience (Configuration & Testing Lessons)
Status: Accepted
Version: 1.2
Date: 2025-09-02
Supersedes:
Superseded-by:
Related: 024, 026, 014
Tags: lessons, process
References:
- [Project ADR Template](./adr-template.md)
---

## Description

Capture key lessons from migrating to clean settings and testing: avoid server‑style patterns in a local app; prefer fixtures over prod hooks; document user scenarios.

## Context

We temporarily removed essential user flexibility (hardware/provider choices) and later restored it. This ADR records the takeaway to guide future changes.

## Decision Drivers

- Local app context first
- Minimal surface area; reversible changes

## Decision

Institutionalize “fixtures not prod hooks”, “env not code branches”, and user‑scenario validation prior to refactors.

## High-Level Architecture

- Lessons feed into config/tests; no code in prod settings

## Related Requirements

### Functional Requirements

- FR‑1: Avoid prod test hooks; use fixtures

### Non-Functional Requirements

- NFR‑1: Keep patterns simple and documented

### Performance Requirements

- PR‑1: Fixture overhead remains minimal

### Integration Requirements

- IR‑1: Apply across modules consistently

## Consequences

### Positive Outcomes

- Fewer regressions; clearer PR reviews; faster rollbacks

### Negative Consequences / Trade-offs

- Requires discipline to avoid convenience edits in prod settings

### Ongoing Maintenance & Considerations

- Periodic reviews to catch regressions into prod config

## Changelog

- 1.2 (2025‑09‑04): Standardized to template; expanded consequences

- 1.1 (2025‑09‑02): Accepted; lessons summarized
