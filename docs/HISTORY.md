# Code and data lineage

This repository was assembled on 2026-07-03 as a clean artifact for the paper.
Development happened in separate private repositories; everything below is the
record needed to trace results back to them. The development repos are frozen
at git tag `paper-v1` (2026-07-03) and are the canonical archive for anything
not present here.

## Where the code came from

| This repo | Origin (all @ `paper-v1`) | Upstream lineage |
|---|---|---|
| `minspan/` | `DataFilter` repo, `tagger/` package (original code; renamed) | — |
| `benchmarks/agentdyn/` | `AnshiHuawei/agent-over-defense` fork | agentdojo (`ethz-spylab/agentdojo`, MIT) + AgentDyn dynamic suites; our additions: MinSpan defense (`token_tagger.py`), fixes |
| `benchmarks/piarena/` | `RenjunAn/PIArena` fork | PIArena (`sleeepeer/PIArena`, ACL 2026); our additions: MinSpan defense, DeepSeek sanitizer adapter, OOM/stability fixes |
| `results/*.csv` | Consolidated from run outputs; aggregation history in the `data_analysis` repo | — |

## Historical names

During development the method and its versions had different names, which
still appear in checkpoints, archived runs, and dataset split identifiers:

| Historical name | Meaning |
|---|---|
| `tagger`, `modernbert_tagger`, `token_tagger` | MinSpan (the defense registration keys in both harnesses still use `modernbert_tagger`) |
| P3, `Tagger1` | MinSpan, paper version — checkpoint `Shi-lab/PITagger`, training set `data/train.jsonl` |
| P2 | Earlier training phase (72,628 records); P3 = P2 + 48,000 targeted records |
| DataFilter | Generative-filtering *baseline* (Yizhu et al., CC BY-NC 4.0), not our method |

Dataset split identifiers `p3_direct_test`, `p3_strategy_test`,
`p3_clean_hard_negative_test` are baked into the data records and the
checkpoint's evaluation contract, so they keep their historical names.
`minspan/build_training_data.py` (historically `p3_data_generation.py`)
constructs the paper training set from the archived P2 base; regenerating from
scratch therefore needs the `paper-v1` DataFilter archive (`data/tagger-p2`).

## Large local-only assets (gitignored)

| Path | Size | Contents | Canonical archive |
|---|---|---|---|
| `data/*.jsonl` | ~380M | training set + test splits | DataFilter repo @ `paper-v1` (`data/tagger-p3`) |
| `benchmarks/agentdyn/runs/` | ~1.3G | raw AgentDyn trajectories behind the paper's AgentDyn numbers | AgentDyn fork @ `paper-v1` |
| `benchmarks/piarena/results/` | ~327M | raw PIArena outputs | PIArena fork @ `paper-v1` |
| `results/evaluation-matrix/` | ~648M | raw local token-level evaluation outputs | DataFilter repo @ `paper-v1` |

## Model

Paper checkpoint: HF `Shi-lab/PITagger` (trained off-machine; no local copy of
the original training run). Verification: download and re-run the PIArena
Direct evaluation; expected mean over 13 datasets: ASR 6.38, utility 73.31.

## License notes

The DataFilter development repo carries the upstream DataFilter code's
CC BY-NC 4.0. The `minspan/` package is original work and can be licensed
independently; the vendored harnesses keep their upstream licenses.
