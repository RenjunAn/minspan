#!/usr/bin/env bash
# Adaptive-attack robustness experiment on PIArena (local Qwen3-4B).
#
# The adaptive attack (adaptive_task_camouflage) embeds the user task verbatim
# and frames the injection as a required sub-step, targeting the task-
# conditioning MinSpan relies on. Three defenses are compared so the source of
# any robustness is isolated:
#   none               is the attack actually potent? (must raise ASR)
#   modernbert_tagger  MinSpan (task-conditioned) under the adaptive attack
#   ablation-no-task-cond  the same filter with task conditioning removed
#
# Runs the short-text / RAG datasets (fast; representative of naturalistic
# injections where task conditioning matters).
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/benchmarks/piarena"
export TORCHDYNAMO_DISABLE=1
PY="$ROOT/.venv/bin/python"

ATTACK="${ATTACK:-adaptive_task_camouflage}"
DATASETS="${DATASETS:-squad_v2 dolly_closed_qa dolly_information_extraction nq_rag msmarco_rag hotpotqa_rag}"

run () {  # config_name  defense  extra_yaml
  local name="$1" defense="$2" ckpt="$3" blank="$4"
  local cfg="/tmp/adaptive_${name}.yaml"
  {
    echo "backend_llm: Qwen/Qwen3-4B-Instruct-2507"
    echo "attack: $ATTACK"
    echo "defense: $defense"
    echo "name: adaptive_${name}"
    echo "seed: 42"
    if [ -n "$ckpt" ]; then
      echo "defense_config:"
      echo "  checkpoint_path: $ckpt"
      echo "  device: cuda"
      echo "  batch_size: 8"
      [ "$blank" = "1" ] && echo "  blank_instruction: true"
    fi
  } > "$cfg"
  for ds in $DATASETS; do
    echo "[adaptive] $name / $ds"
    "$PY" main.py --config "$cfg" --dataset "$ds" 2>&1 | tail -1
  done
}

run nodef            none              ""                                 0
run minspan          modernbert_tagger ../../checkpoints/minspan          0
run no_task_cond     modernbert_tagger ../../checkpoints/ablation-no-task-cond/best 1
echo "ADAPTIVE-DONE"
