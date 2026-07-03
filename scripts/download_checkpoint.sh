#!/usr/bin/env bash
# Download the paper checkpoint from HuggingFace into
# checkpoints/pitagger, the canonical location all eval configs point at.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/checkpoints/pitagger}"

# model files only — the HF repo also hosts dataset copies under datasets/
if command -v hf >/dev/null 2>&1; then
  hf download Shi-lab/PITagger --exclude 'datasets/*' --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download Shi-lab/PITagger --exclude 'datasets/*' --local-dir "$DEST"
else
  echo "ERROR: install huggingface_hub (pip install -U huggingface_hub)" >&2
  exit 2
fi
echo "Checkpoint ready at $DEST"
