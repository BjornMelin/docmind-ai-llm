#!/bin/sh
set -eu

python scripts/parser_health.py --check
exec "$@"
