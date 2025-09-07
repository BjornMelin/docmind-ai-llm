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

```gherkin
Scenario: AES‑GCM round-trip
  Given encryption enabled and a test image
  When I encrypt to .enc and then decrypt
  Then the decrypted bytes SHALL match the original and metadata SHALL include encrypted=true and kid
```

### Encryption of Page Images (AES‑GCM)

The system MAY encrypt page images at rest using AES‑GCM when enabled.

- Keys: Provided via environment/keystore; never logged in plaintext.
- Metadata: `encrypted=true`, `alg="AES-256-GCM"`, `kid` recorded; pHash computed prior to encryption.
- Files: Use `.enc` extension; decrypt via provided utility before reading bytes.
- AAD: SHOULD include non‑sensitive context (e.g., page_id) to bind ciphertext to metadata.
- Rotation: Keys SHOULD support rotation; new ingests use the new kid; old objects remain decryptable while key material exists.
