#!/usr/bin/env bash
# Train the two paper ablations plus a matched control, sequentially.
#
# Protocol: the paper checkpoint was trained in two stages on other hardware
# (ModernBERT-base -> 3 epochs P2 data -> 1 epoch P3 data, lr 1e-5). The
# ablations are trained locally in a single stage from ModernBERT-base on the
# full P3 training set (which contains the P2 records), 2 epochs, lr 3e-5
# (the encoder default), all other hyperparameters as in the paper stage
# (batch 1 x grad-accum 16, instruction dropout 0.15, boundary weight 1.0
# radius 2, seed 42). Ablation deltas are read against the matched control,
# not against the paper checkpoint.
#
#   control            full data, task conditioning on
#   no_task_cond       instruction blanked in every training record
#   no_hard_negatives  clean-hard-negative subset removed from training
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

python scripts/make_ablation_data.py

COMMON=(
  --backbone-type encoder
  --model-name answerdotai/ModernBERT-base
  --epochs 2
  --instruction-dropout 0.15
  --seed 42
)

python -m minspan.train "${COMMON[@]}" \
  --output-dir checkpoints/ablation-control "$@"

python -m minspan.train "${COMMON[@]}" \
  --instruction-dropout 1.0 \
  --output-dir checkpoints/ablation-no-task-cond "$@"

python -m minspan.train "${COMMON[@]}" \
  --train-data data/ablations/train_no_hard_negatives.jsonl \
  --output-dir checkpoints/ablation-no-hard-negatives "$@"
