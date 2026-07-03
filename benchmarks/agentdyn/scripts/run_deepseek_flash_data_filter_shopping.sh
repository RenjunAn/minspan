#!/usr/bin/env bash
# Run defended DeepSeek Flash DataFilter experiments for the shopping suite.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SUITES="${SUITES:-shopping}"
export ATTACKS="${ATTACKS:-none important_instructions}"
export LOGDIR="${LOGDIR:-runs}"

exec ./scripts/run_defense_deepseek_flash_data_filter.sh "$@"
