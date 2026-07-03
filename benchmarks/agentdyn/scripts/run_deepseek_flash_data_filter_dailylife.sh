#!/usr/bin/env bash
# Run defended DeepSeek Flash DataFilter experiments for the dailylife suite.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SUITES="${SUITES:-dailylife}"
export ATTACKS="${ATTACKS:-none important_instructions}"
export LOGDIR="${LOGDIR:-runs}"

exec ./scripts/run_defense_deepseek_flash_data_filter.sh "$@"
