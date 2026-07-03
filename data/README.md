# Datasets

The paper training set and evaluation splits. Files over ~1M are **not tracked
in git** (see `docs/HISTORY.md` for the canonical archive); tracked files are
the split manifest and dataset statistics.

| File | Tracked | Contents |
|---|---|---|
| `train.jsonl` (143M) | no | Training set, 120,628 records |
| `validation.jsonl` (6.3M) | no | Validation split |
| `p3_direct_test.jsonl`, `p3_strategy_test.jsonl`, `p3_clean_hard_negative_test.jsonl` | no | Held-out test splits (split names are baked into the records) |
| `nemo_test*.jsonl`, `sep_test.jsonl`, `format_test.jsonl` | no | Cross-distribution test splits |
| `split_manifest.json`, `stats.json` | yes | Split definitions and dataset statistics |

Construction: `minspan/build_training_data.py` extends an earlier 72,628-record
base with 48,000 targeted records (direct attacks, compliance-style attacks,
clean hard negatives, agent/tool-output formats). Upstream sources and helpers:
`minspan/data_generation.py`, `minspan/nemotron_to_minspan.py`,
`minspan/augment.py`. Regeneration from scratch requires the archived base
dataset (`docs/HISTORY.md`).

TODO before release: publish the dataset on HuggingFace so untracked splits
are downloadable.
