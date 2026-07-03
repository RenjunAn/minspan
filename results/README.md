# Results

Normalized result data behind the paper's tables and figures (see
`REPRODUCE.md` for the full mapping).

- `piarena_main.csv` — PIArena per-dataset utility/ASR (Qwen3-4B backend)
- `agentdyn_main.csv` — AgentDyn per-suite BU/UA/ASR and latency (DeepSeek-V4 Flash)
- `local_token_eval.csv` — token-level metrics per held-out split
- `training_data_summary.csv` — training set composition
- `evaluation-matrix/` — raw local evaluation outputs (local-only, gitignored)

Planned: collector scripts that regenerate every CSV here directly from
`benchmarks/agentdyn/runs/` and `benchmarks/piarena/results/`.
