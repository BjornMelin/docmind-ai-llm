#!/usr/bin/env bash
set -euo pipefail

# Fail if forbidden aggregator imports exist in src/ or tests/
echo "Running aggregator guardrail checks..."

for pattern in \
  "from\\s+src\\.agents\\s+import" \
  "from\\s+src\\.agents\\.tools\\s+import" \
  "src\\.agents\\.tools\\.ToolFactory" \
  "src\\.agents\\.tools\\.logger" \
  "src\\.agents\\.tools\\.time"; do
  if rg -n --no-heading -S "$pattern" src tests >/dev/null; then
    echo "Forbidden pattern found: $pattern" >&2
    rg -n --no-heading -S "$pattern" src tests || true
    exit 1
  fi
done

echo "Guardrail checks passed."
