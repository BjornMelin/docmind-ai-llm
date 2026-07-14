---
spec: SPEC-011
title: Secure local data and control network access
version: 1.3.0
date: 2026-07-10
owners: ["ai-arch"]
status: Final
related_requirements:
  - NFR-SEC-001: Network egress SHALL be off by default.
  - NFR-SEC-002: Providers and endpoints SHALL be allowlisted.
  - NFR-COMP-001: Licenses SHALL be documented for embedded models.
related_adrs: ["ADR-031","ADR-024","ADR-059"]
---


## Objective

Ensure offline by default, with explicit allowlists for endpoints. Document licenses for packaged weights and OCR/PDF libs.

## File operations

### Files to update

- `src/config/settings.py`: add allowlist list and enforcement checks.
- `README.md`: license notes section.

## Acceptance criteria

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

### Encrypt page images with AES-GCM

The system MAY encrypt page images at rest using AES‑GCM when enabled.

- Keys: Provided via environment/keystore; never logged in plaintext.
- Metadata: `encrypted=true`, `alg="AES-256-GCM"`, `kid` recorded; pHash computed prior to encryption.
- Files: Use `.enc` extension; decrypt via provided utility before reading bytes.
- AAD: SHOULD include non‑sensitive context (e.g., page_id) to bind ciphertext to metadata.
- Rotation: Keys SHOULD support rotation; new ingests use the new kid; old objects remain decryptable while key material exists.
- Consumption: SigLIP visual scoring SHALL use the encrypted-aware image loader and MUST release decoded images on success, error, or timeout.

## Control network access

- Default posture MUST be offline‑first; remote endpoints disabled unless explicitly allowlisted.
- LM Studio endpoints MUST end with `/v1`.
- Add tests to reject non‑allowlisted URLs when policy is strict.
- When remote endpoints are disabled, allowlisted hostnames MUST resolve to public IPs.
  Hosts that resolve to private/link-local/reserved ranges are rejected (defense-in-depth
  against SSRF and DNS rebinding). If you intentionally need private/internal endpoints
  (e.g., Docker service hostnames), set `DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true`
  or use an architecture that keeps the endpoint on loopback (e.g., shared network
  namespace so the app connects to `http://localhost`).
- Document parsing and OCR MUST always use the local Docling, pypdfium2, and
  RapidOCR path. No environment variable selects a remote parser or another PDF
  backend.
- The parser has no remote, VLM, or arbitrary-code model fallback.
- PDF parsing MUST fail before conversion when app-owned Docling files are
  missing or differ from the source-controlled manifest. RapidOCR MUST use the
  models and checksums supplied by its locked wheel. Health output reports only
  relative Docling paths and integrity reasons.
- PDF and other binary parser failures MUST fail closed with a typed parsing
  error. Binary source bytes MUST NOT be decoded as UTF-8 fallback text or
  published to retrieval stores.
- Fail-open behavior is limited to optional work after a successful parse, such
  as searchable-PDF export, visual enrichment, and best-effort indexing.
- Searchable-PDF export uses OCRmyPDF and Tesseract only as an explicit,
  fail-open utility. It requires POSIX process groups; Windows users must run it
  under WSL2. Cancellation, timeout, and failure kill and reap the subprocess
  group. The utility does not send document content to hosted APIs.

## Enable optional Ollama Cloud tools

- Ollama Cloud access (including `web_search` / `web_fetch`) is a **remote endpoint** and MUST remain disabled unless explicitly enabled.
- Enabling web tools requires:
  - `DOCMIND_OLLAMA_ENABLE_WEB_SEARCH=true`
  - An API key via `DOCMIND_OLLAMA_API_KEY`
  - Remote endpoints enabled (`DOCMIND_SECURITY__ALLOW_REMOTE_ENDPOINTS=true`)
  - For defense-in-depth, also include `https://ollama.com` in `DOCMIND_SECURITY__ENDPOINT_ALLOWLIST` (note: allowlist enforcement applies when `ALLOW_REMOTE_ENDPOINTS=false`)
- Secrets (API keys) MUST NOT be logged.

## Redact secrets and telemetry

- Secrets MUST NEVER be logged. Use Pydantic v2 secrets and ensure redaction in any structured logs.
- Telemetry MUST be minimal and PII‑safe; avoid logging raw prompts unless explicitly enabled for development.

## Keep staleness checks and exports local

- Staleness computation MUST be local; the UI MUST NOT trigger network calls for staleness checks.
- Export paths MUST be non‑egress; sanitize file names and block symlink traversal; enforce base path resolution.
