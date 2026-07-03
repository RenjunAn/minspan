#!/usr/bin/env bash
# Run the AgentDyn evaluation with the MinSpan defense (DeepSeek-V4 Flash
# backend, paper setting). Requires checkpoints/pitagger and the AgentDyn
# environment (cd benchmarks/agentdyn && uv sync).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MODERNBERT_TAGGER_CHECKPOINT="${MODERNBERT_TAGGER_CHECKPOINT:-$ROOT/checkpoints/pitagger}"
export DEFENSES="${DEFENSES:-modernbert_tagger}"
cd "$ROOT/benchmarks/agentdyn"
exec bash scripts/run_defense_token_taggers.sh "$@"
