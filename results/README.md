# Results

Normalized result data behind the paper's tables and figures (see
`REPRODUCE.md` for the full mapping). Regenerate with
`scripts/collect_agentdyn.py` and `scripts/collect_piarena.py`.

- `piarena_main.csv` — PIArena per-dataset per-attack utility/ASR, full precision (Qwen3-4B backend)
- `piarena_table.csv` — macro-averaged PIArena table incl. published baselines (leaderboard rounding convention)
- `agentdyn_main.csv` — MinSpan on AgentDyn: per-suite BU/UA/ASR, removal/edit rates, latency (DeepSeek-V4 Flash)
- `agentdyn_pairs.csv` — every backend–defense pair, overall BU/UA/ASR (published Appendix G + our runs)
- `paired_costs.csv` — cross-model paired costs (paper Table `tab:paired-costs`)
- `deepseek_table.csv` — both DeepSeek backends × all defenses (paper Table `tab:deepseek`)
- `defense_ops.csv` — filtering-defense audit: removal quality, benign edits, fabrications, parse failures, latency
- `local_token_eval.csv` — token-level metrics per held-out split
- `local-eval/` — checkpoint-verification eval outputs (local-only, gitignored)
- `training_data_summary.csv` — training set composition
- `reference/` — published AgentDyn Appendix G numbers (arXiv 2602.03117v3) + extraction script
- `evaluation-matrix/` — raw local evaluation outputs (local-only, gitignored)
