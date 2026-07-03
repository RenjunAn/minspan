#!/usr/bin/env bash
# Run defended DeepSeek Flash PI sanitizer experiments for the github suite.

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SUITES="${SUITES:-github}"
export ATTACKS="${ATTACKS:-none important_instructions}"
export LOGDIR="${LOGDIR:-runs}"

exec ./scripts/run_defense_deepseek_flash_pi_sanitizer.sh "$@"
