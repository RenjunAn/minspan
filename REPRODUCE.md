# Reproducibility map

Every table/figure in the paper, the data behind it, and the current status of
the chain. Raw run outputs are local-only (see `docs/HISTORY.md`); normalized
result data lives in `results/`.

## Paper results → sources

| Paper item | Content | Source | Status |
|---|---|---|---|
| Table 1 (`tab:paired-costs`) | Cross-model paired costs on AgentDyn | AgentDyn Appendix G (published) + our DeepSeek runs (`benchmarks/agentdyn/runs/deepseek-*`) | ⚠ collector to be written → `results/paired_costs.csv` |
| Table 2 (`tab:deepseek`) | DeepSeek V4 Flash/Pro per-defense BU/UA/ASR | `benchmarks/agentdyn/runs/deepseek-v4-{flash,pro}-*` | ⚠ collector to be written |
| Fig. `security_utility_scatter` | UA vs ASR scatter, all backend–defense pairs | `figures/src/agentdyn_scatter.csv` (legacy extraction) | ✗ rebuild from raw runs |
| Fig. `piarena_per_dataset` | MinSpan per-dataset Direct results | `figures/src/piarena_per_dataset.csv` ← `results/piarena_main.csv` | ⚠ re-derive from raw |
| Fig. `cumulative_latency` | Cumulative latency along trajectories | AgentDyn runs wall-clock records | ⚠ collector to be written |
| §5 PIArena main table | 13-dataset Direct ASR/Utility, Qwen3-4B backend | `results/piarena_main.csv` ← `benchmarks/piarena/results/` | ⚠ verify against raw |
| §5 AgentDyn main table | BU/UA/ASR, DeepSeek-V4 Flash backend | `results/agentdyn_main.csv` ← `benchmarks/agentdyn/runs/deepseek-v4-flash-modernbert_tagger-*` | ⚠ verify against raw |
| §5 local token metrics | Injection/clean recall, exact match, latency per split | `results/local_token_eval.csv` ← `results/evaluation-matrix/` (local-only) | ⚠ verify against raw |
| §4 training set description | Composition of the training data | `results/training_data_summary.csv`, `data/stats.json` | ✓ |
| §5 latency (2.26 ms) | Per-call median latency | local Direct-test measurement | ⚠ re-measure on unified hardware |

Legend: ✗ broken chain, ⚠ chain exists but must be re-derived from raw data,
✓ reproducible from this repo.

## Model

- Paper checkpoint: HF `Shi-lab/PITagger` (`scripts/download_checkpoint.sh`).
  Verification: re-run PIArena Direct eval, expect mean ASR 6.38 / utility 73.31.
- Training: `scripts/train.sh` (config in `minspan/train.py`); dataset
  construction: `minspan/build_training_data.py` (see `data/README.md`).

## Pending experiments

P0: ablations (no task-conditioning; no clean hard negatives) · CommandSans
reproduction on both benchmarks · train/test contamination audit · Figure 1
(method overview). P1: multi-seed / bootstrap CIs · per-environment AgentDyn
tables · fair same-hardware latency comparison.
