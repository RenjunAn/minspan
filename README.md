# MinSpan

Task-conditioned minimal-span filtering for dynamic LLM agents. MinSpan jointly
encodes the trusted user task and the untrusted tool output with a 149.6M
bidirectional encoder (ModernBERT), predicts a Keep/Drop label for every output
token in a single non-autoregressive forward pass, deletes only the character
spans identified as injected, and copies every remaining character verbatim.

Artifact repository for *Guardrails Should Not Become Barricades:
Task-Conditioned Minimal-Span Filtering for Dynamic LLM Agents* (ICLR 2027
submission). The method, both benchmark harnesses, baselines, and the result
data all live here; training and evaluation run from this repository alone.

## Layout

| Path | Contents |
|---|---|
| `minspan/` | Method: model, training, evaluation, dataset construction |
| `benchmarks/agentdyn/` | AgentDyn harness (agentdojo-based; MinSpan defense integrated) |
| `benchmarks/piarena/` | PIArena harness (MinSpan defense integrated) |
| `baselines/commandsans/` | CommandSans reimplementation |
| `scripts/` | Entry points: checkpoint download, training, benchmark runs |
| `data/` | Training and evaluation datasets (large files local-only; see `data/README.md`) |
| `results/` | Result data behind the paper's tables and figures |
| `figures/` | Figure generation scripts |
| `docs/HISTORY.md` | Code and data lineage |

## Quickstart

```bash
# 1. method package (training / local evaluation)
pip install -e .

# 2. paper checkpoint from HF Shi-lab/PITagger
bash scripts/download_checkpoint.sh

# 3. train (or retrain ablations) on the paper training set
bash scripts/train.sh --output-dir checkpoints/minspan-retrain

# 4. benchmarks (each harness manages its own deps)
pip install -e benchmarks/piarena && bash scripts/eval_piarena.sh
(cd benchmarks/agentdyn && uv sync) && bash scripts/eval_agentdyn.sh
```

Backend requirements: PIArena's paper setting serves Qwen3-4B locally;
AgentDyn's paper setting calls the DeepSeek-V4 Flash API (set the API key per
`benchmarks/agentdyn/README.md`).

## Reproducibility

`REPRODUCE.md` maps every table and figure in the paper to the script and data
that produce it. Provenance of the vendored harnesses and datasets is recorded
in `docs/HISTORY.md`.

## License

TBD before public release. Vendored harnesses keep their upstream licenses
(agentdojo: MIT; PIArena: see `benchmarks/piarena/LICENSE`).
