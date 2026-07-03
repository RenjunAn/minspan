#!/usr/bin/env bash
# Train MinSpan on the paper training set (see data/README.md).
# Extra args are forwarded to minspan.train (e.g. ablation flags, output dir).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python -m minspan.train \
  --train-data data/train.jsonl \
  --validation-data data/validation.jsonl \
  "$@"
