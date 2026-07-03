#!/usr/bin/env bash
# Run the PIArena evaluation with the MinSpan defense (Qwen3-4B backend,
# paper setting). Requires checkpoints/minspan (scripts/download_checkpoint.sh)
# and PIArena deps installed (pip install -e benchmarks/piarena).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/benchmarks/piarena"
exec python main.py --config configs/experiments/modernbert_tagger.yaml "$@"
