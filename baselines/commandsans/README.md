# CommandSans baseline

Reimplementation of CommandSans: task-agnostic sanitization of tool outputs.
A token classifier (XLM-RoBERTa-base) tags every word of a tool output as
INSTRUCTION or DATA and the instruction words are deleted before the output
reaches the agent. Unlike MinSpan, the classifier never sees the user task.

This is an independent reimplementation from the paper's method description
(binary word labels from `<instruction>`-tagged annotations, 512-token windows
with 256 overlap, class-weighted cross-entropy, example-level splits). No
third-party code or checkpoint is included.

## Workflow

```bash
pip install -e baselines/commandsans[data]

# 1. sample raw annotation inputs (BFCL v3 + OpenOrca, 2,000 each, seed 42)
python baselines/commandsans/scripts/prepare_data.py --output-dir baselines/commandsans/data

# 2. LLM annotation (resumable; costs API credits — run with --limit first)
export DEEPSEEK_API_KEY=...
python -m commandsans.annotate \
  --input baselines/commandsans/data/raw.jsonl \
  --output baselines/commandsans/data/labeled.jsonl \
  --base-url https://api.deepseek.com --model deepseek-chat \
  --api-key-env DEEPSEEK_API_KEY

# 3. train the classifier
python -m commandsans.train \
  --train-data baselines/commandsans/data/labeled.jsonl \
  --output-dir checkpoints/commandsans

# 4. run the benchmarks with defense=commandsans
#    PIArena:  configs/experiments/commandsans.yaml
#    AgentDyn: DEFENSES=commandsans COMMANDSANS_CHECKPOINT=$PWD/checkpoints/commandsans/best \
#              bash scripts/eval_agentdyn.sh
```

## Trained checkpoint (2026-07-03)

Annotated with DeepSeek (deepseek-chat, temperature 0): 3,911 raw records →
3,442 validated (469 rejected by the exact-preservation/tag checks). Training
(xlm-roberta-base, 3 epochs, seed 42) reached word-level validation
F1 0.728 (precision 0.826, recall 0.651); checkpoint at
`checkpoints/commandsans/best` (local-only), metrics in
`checkpoints/commandsans/history.json`. Behavioral spot-check: benign prose
untouched, direct injections removed, imperative-looking code over-deleted —
the task-agnostic over-defense pattern the paper discusses.

## Harness integration

- AgentDyn: `benchmarks/agentdyn/src/agentdojo/agent_pipeline/commandsans_defense.py`
  (defense name `commandsans`, checkpoint via `COMMANDSANS_CHECKPOINT`)
- PIArena: `benchmarks/piarena/piarena/defenses/commandsans/`
  (config `configs/experiments/commandsans.yaml`)

Both adapters reuse the harnesses' existing tool-output filtering pipelines,
so CommandSans is logged and scored exactly like the other filtering defenses.
