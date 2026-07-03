#!/usr/bin/env bash
# Run the DataFilter bidirectional and ModernBERT token taggers on AgentDyn.

set -u -o pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not installed or not on PATH" >&2
  exit 2
fi

DRY_RUN="${DRY_RUN:-0}"
LOGDIR="${LOGDIR:-runs}"
RUN_LOG="${RUN_LOG:-$LOGDIR/logs/token_taggers.log}"
FORCE_RERUN="${FORCE_RERUN:-1}"
ALLOW_TAGGER_FAILURES="${ALLOW_TAGGER_FAILURES:-0}"
read -r -a DEFENSE_LIST <<< "${DEFENSES:-modernbert_tagger datafilter_bidir_tagger}"
read -r -a AGENT_LIST <<< "${AGENTS:-DEEPSEEK_V4_FLASH}"
read -r -a SUITE_LIST <<< "${SUITES:-shopping github dailylife}"
read -r -a ATTACK_LIST <<< "${ATTACKS:-none important_instructions}"
if [ "${UV_RUN_FLAGS+x}" = "x" ]; then
  read -r -a UV_FLAGS <<< "$UV_RUN_FLAGS"
else
  UV_FLAGS=(--no-sync)
fi
EXTRA_BENCHMARK_ARGS=("$@")

mkdir -p "$(dirname "$RUN_LOG")"

validate_defense() {
  local defense="$1"
  case "$defense" in
    modernbert_tagger)
      if [ "$DRY_RUN" != "1" ]; then
        : "${MODERNBERT_TAGGER_CHECKPOINT:?MODERNBERT_TAGGER_CHECKPOINT must be set}"
        if [ ! -d "$MODERNBERT_TAGGER_CHECKPOINT" ]; then
          echo "ERROR: ModernBERT checkpoint not found: $MODERNBERT_TAGGER_CHECKPOINT" >&2
          return 2
        fi
      fi
      ;;
    datafilter_bidir_tagger)
      if [ "$DRY_RUN" != "1" ]; then
        : "${DATAFILTER_TAGGER_CHECKPOINT:?DATAFILTER_TAGGER_CHECKPOINT must be set}"
        : "${DATAFILTER_BACKBONE_MODEL:?DATAFILTER_BACKBONE_MODEL must be set}"
        if [ ! -d "$DATAFILTER_TAGGER_CHECKPOINT" ]; then
          echo "ERROR: bidirectional tagger checkpoint not found: $DATAFILTER_TAGGER_CHECKPOINT" >&2
          return 2
        fi
      fi
      ;;
    *)
      echo "ERROR: unsupported token tagger defense: $defense" >&2
      return 2
      ;;
  esac
}

run_one() {
  local defense="$1"
  local agent="$2"
  local suite="$3"
  local attack="$4"
  local started_ns
  started_ns="$(date +%s%N)"
  local args=(
    -s "$suite"
    --model "$agent"
    --defense "$defense"
    --tool-output-format json
    --max-workers 1
    --logdir "$LOGDIR"
  )

  if [ "$attack" = "important_instructions" ]; then
    args+=(--attack important_instructions)
  elif [ "$attack" != "none" ]; then
    echo "ERROR: unsupported attack mode: $attack" >&2
    return 2
  fi
  if [ "$FORCE_RERUN" = "1" ]; then
    args+=(-f)
  fi
  args+=("${EXTRA_BENCHMARK_ARGS[@]}")

  echo "=========================================================" | tee -a "$RUN_LOG"
  echo "[$(date '+%F %T')] START $defense $agent suite=$suite attack=$attack" | tee -a "$RUN_LOG"
  echo "  cmd: uv run ${UV_FLAGS[*]} python -m agentdojo.scripts.benchmark ${args[*]}" | tee -a "$RUN_LOG"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[$(date '+%F %T')] DRY-RUN skip execution" | tee -a "$RUN_LOG"
    return 0
  fi

  uv run "${UV_FLAGS[@]}" python -m agentdojo.scripts.benchmark "${args[@]}" 2>&1 | tee -a "$RUN_LOG"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ] && [ "$ALLOW_TAGGER_FAILURES" != "1" ]; then
    uv run "${UV_FLAGS[@]}" python -m agentdojo.scripts.token_tagger_traces \
      --logdir "$LOGDIR" \
      --defense "$defense" \
      --since-ns "$started_ns" 2>&1 | tee -a "$RUN_LOG"
    local trace_rc=${PIPESTATUS[0]}
    if [ "$trace_rc" -ne 0 ]; then
      rc="$trace_rc"
    fi
  fi
  echo "[$(date '+%F %T')] END $defense $agent suite=$suite attack=$attack rc=$rc" | tee -a "$RUN_LOG"
  return "$rc"
}

failures=0
for defense in "${DEFENSE_LIST[@]}"; do
  if ! validate_defense "$defense"; then
    exit 2
  fi
  for agent in "${AGENT_LIST[@]}"; do
    for suite in "${SUITE_LIST[@]}"; do
      for attack in "${ATTACK_LIST[@]}"; do
        if ! run_one "$defense" "$agent" "$suite" "$attack"; then
          failures=$((failures + 1))
        fi
      done
    done
  done
done

if [ "$failures" -eq 0 ]; then
  echo "[$(date '+%F %T')] DONE token tagger defenses" | tee -a "$RUN_LOG"
  exit 0
fi

echo "[$(date '+%F %T')] DONE token tagger defenses failures=$failures" | tee -a "$RUN_LOG"
exit 1
