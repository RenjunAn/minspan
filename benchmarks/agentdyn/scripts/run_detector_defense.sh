#!/usr/bin/env bash
# Run one HF-backed prompt-injection detector defense on AgentDyn's three suites.
#
# Defaults:
#   suites: shopping github dailylife
#   agents: deepseek-v4-flash deepseek-v4-pro
#   attacks: none important_instructions
#
# Useful overrides:
#   LOGDIR=runs_detector FORCE_RERUN=1 ./scripts/run_defense_piguard_detector.sh
#   AGENTS="deepseek-v4-flash" ATTACKS="important_instructions" ./scripts/run_defense_prompt_guard_2_detector.sh
#   UV_RUN_FLAGS="" ./scripts/run_defense_transformers_pi_detector.sh --max-workers 3
#   DRY_RUN=1 ./scripts/run_defense_transformers_pi_detector.sh

set -u -o pipefail

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run_detector_defense.sh DEFENSE [extra benchmark args]

DEFENSE must be one of:
  transformers_pi_detector
  piguard_detector
  prompt_guard_2_detector
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

DEFENSE="$1"
shift

case "$DEFENSE" in
  transformers_pi_detector)
    DETECTOR_MODEL="protectai/deberta-v3-base-prompt-injection-v2"
    ;;
  piguard_detector)
    DETECTOR_MODEL="leolee99/PIGuard"
    ;;
  prompt_guard_2_detector)
    DETECTOR_MODEL="meta-llama/Llama-Prompt-Guard-2-86M"
    ;;
  *)
    usage
    exit 2
    ;;
esac

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not on PATH" >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"

if [ -z "${DEEPSEEK_API_KEY:-}" ] && [ "$DRY_RUN" != "1" ]; then
  echo "ERROR: DEEPSEEK_API_KEY must be set" >&2
  exit 2
fi

LOGDIR="${LOGDIR:-runs}"
RUN_LOG="${RUN_LOG:-$LOGDIR/logs/${DEFENSE}.log}"
FORCE_RERUN="${FORCE_RERUN:-0}"
mkdir -p "$(dirname "$RUN_LOG")"

AGENTS=(${AGENTS:-deepseek-v4-flash deepseek-v4-pro})
SUITES=(${SUITES:-shopping github dailylife})
ATTACKS=(${ATTACKS:-none important_instructions})
if [ "${UV_RUN_FLAGS+x}" = "x" ]; then
  UV_RUN_FLAGS=(${UV_RUN_FLAGS})
else
  UV_RUN_FLAGS=(--no-sync)
fi
EXTRA_BENCHMARK_ARGS=("$@")

SUITE_ARGS=()
for suite in "${SUITES[@]}"; do
  SUITE_ARGS+=(-s "$suite")
done

echo "=========================================================" | tee -a "$RUN_LOG"
echo "[$(date '+%F %T')] START detector defense: $DEFENSE" | tee -a "$RUN_LOG"
echo "  detector_model: $DETECTOR_MODEL" | tee -a "$RUN_LOG"
echo "  agents: ${AGENTS[*]}" | tee -a "$RUN_LOG"
echo "  suites: ${SUITES[*]}" | tee -a "$RUN_LOG"
echo "  attacks: ${ATTACKS[*]}" | tee -a "$RUN_LOG"
echo "  logdir: $LOGDIR" | tee -a "$RUN_LOG"
echo "  uv flags: ${UV_RUN_FLAGS[*]}" | tee -a "$RUN_LOG"
echo "  dry_run: $DRY_RUN" | tee -a "$RUN_LOG"

if [ "$DRY_RUN" != "1" ] && ! uv run "${UV_RUN_FLAGS[@]}" python - <<'PY' 2>&1 | tee -a "$RUN_LOG"; then
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
PY
  echo "ERROR: torch/transformers preflight failed. Run: uv sync --dev --all-extras --no-lock" | tee -a "$RUN_LOG" >&2
  exit 2
fi

run_one() {
  local agent="$1"
  local attack="$2"
  local args=("${SUITE_ARGS[@]}" --model "$agent" --defense "$DEFENSE" --logdir "$LOGDIR")

  if [ "$attack" = "important_instructions" ]; then
    args+=(--attack important_instructions)
  elif [ "$attack" != "none" ]; then
    echo "ERROR: unsupported attack mode: $attack" | tee -a "$RUN_LOG" >&2
    return 2
  fi

  if [ "$FORCE_RERUN" = "1" ]; then
    args+=(-f)
  fi

  args+=("${EXTRA_BENCHMARK_ARGS[@]}")

  echo "=========================================================" | tee -a "$RUN_LOG"
  echo "[$(date '+%F %T')] START $DEFENSE $agent attack=$attack" | tee -a "$RUN_LOG"
  echo "  cmd: uv run ${UV_RUN_FLAGS[*]} python -m agentdojo.scripts.benchmark ${args[*]}" | tee -a "$RUN_LOG"

  if [ "$DRY_RUN" = "1" ]; then
    echo "[$(date '+%F %T')] DRY-RUN skip execution" | tee -a "$RUN_LOG"
    return 0
  fi

  local t0
  t0=$(date +%s)
  uv run "${UV_RUN_FLAGS[@]}" python -m agentdojo.scripts.benchmark "${args[@]}" 2>&1 | tee -a "$RUN_LOG"
  local rc=${PIPESTATUS[0]}
  local dur=$(( $(date +%s) - t0 ))

  echo "[$(date '+%F %T')] END   $DEFENSE $agent attack=$attack rc=$rc dur=${dur}s" | tee -a "$RUN_LOG"
  return "$rc"
}

failures=0
for agent in "${AGENTS[@]}"; do
  for attack in "${ATTACKS[@]}"; do
    if ! run_one "$agent" "$attack"; then
      failures=$((failures + 1))
    fi
  done
done

echo "=========================================================" | tee -a "$RUN_LOG"
if [ "$failures" -eq 0 ]; then
  echo "[$(date '+%F %T')] DONE $DEFENSE all commands passed" | tee -a "$RUN_LOG"
  exit 0
fi

echo "[$(date '+%F %T')] DONE $DEFENSE failures=$failures" | tee -a "$RUN_LOG"
exit 1
