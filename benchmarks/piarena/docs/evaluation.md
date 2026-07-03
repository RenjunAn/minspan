---
title: Evaluation
slug: evaluation
category: guide
---

# Evaluation

PIArena has two main evaluation paths:

- `main.py` for standard attacks and defenses
- `main_search.py` for search-based attacks such as `pair`, `tap`, `strategy_search`, and `nanogcg`

## Standard Evaluation

Use `main.py` when the attack is directly constructed from the input sample.

```bash
python main.py \
  --dataset squad_v2 \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attack combined \
  --defense promptguard \
  --name demo \
  --seed 42
```

You can also use a config file:

```yaml
dataset: squad_v2
backend_llm: Qwen/Qwen3-4B-Instruct-2507
attack: combined
defense: promptguard
name: demo
seed: 42

attack_config:
  inject_position: end

defense_config:
  smooth_win: 5
  max_gap: 10
  threshold: 0.01
```

```bash
python main.py --config configs/experiments/my_experiment.yaml
```

`defense_config` is also supported by `main_search.py`, `main_injecagent.py`, and `main_agentdojo.py`. This is required for defenses such as `deepseek_pisanitizer` and `modernbert_tagger`, where API credentials or checkpoint paths should come from config.

## Search-Based Evaluation

Use `main_search.py` for attacks that iteratively search for stronger injections.

```bash
python main_search.py \
  --dataset squad_v2 \
  --attack strategy_search \
  --defense datafilter \
  --backend_llm Qwen/Qwen3-4B-Instruct-2507 \
  --attacker_llm Qwen/Qwen3-4B-Instruct-2507
```

The search-based attacks supported today are:

- [`pair`](/docs/attacks/pair)
- [`tap`](/docs/attacks/tap)
- [`strategy_search`](/docs/attacks/strategy-search)
- [`nanogcg`](/docs/attacks/nanogcg)

## Batch Runs

For large experiments, edit and run the batch scripts:

- `scripts/run.py` for standard attacks
- `scripts/run_search.py` for search-based attacks
- `scripts/run_injecagent.py` and `scripts/run_agentdojo.py` for agent benchmarks

Example:

```bash
python scripts/run.py
python scripts/run_search.py
```

These scripts use `piarena.gpu_utils.GPUScheduler` to distribute work across GPUs.

## DeepSeek and ModernBERT Leaderboard Scripts

For the `deepseek_pisanitizer` and `modernbert_tagger` integrations, defense-specific scripts cover the non-WASP leaderboard scope:

```bash
# Build and optionally run one-sample smoke commands.
python scripts/run_deepseek_pisanitizer_smoke.py --dry-run
python scripts/run_modernbert_tagger_smoke.py --skip-agents

# Build and optionally run the full leaderboard matrix.
python scripts/run_deepseek_pisanitizer_full.py --dry-run
python scripts/run_modernbert_tagger_full.py
```

Each script accepts `--config` to override its default YAML. Smoke scripts create one-sample temporary datasets under `.tmp/{defense}_smoke/` for the standard/external benchmark scopes, then run the selected defense through the same entry points as the full benchmark. Agent commands are included by default and can be disabled with `--skip-agents`.

The full scripts map website leaderboard names to runner names, for example `dolly_qa` to `dolly_closed_qa`, `multinews` to `multi_news_long`, `strategy` to `strategy_search`, and `gcg` to `nanogcg`. They exclude WASP. After a non-dry run, leaderboard-compatible entries are exported to:

```text
results/leaderboard_entries/{name}.json
```

Use `--pending-only` with `--dry-run` to print only commands whose result files are missing or incomplete. Use `--local-datasets-only` when you want to skip full-runner entries whose splits are not present in the bundled PIArena dataset set, such as external `opi` or `sep` rows in environments where those splits are unavailable:

```bash
python scripts/run_modernbert_tagger_full.py --dry-run --pending-only --local-datasets-only --skip-agents
```

Use `--force-agentdojo` when you need to regenerate AgentDojo or AgentDyn logs under the same `--name`; it passes `--force-rerun` through to the vendored AgentDojo benchmark and keeps those agent commands active even with `--pending-only`.

Those entries use the same fields as `website/data/results.json`. InjecAgent utility is derived from `valid_rate`; AgentDojo and AgentDyn ASR is derived from the average injection-success flag stored as `security` in AgentDojo result logs. The full leaderboard exporter reads AgentDojo logs from `results/agent_evaluations/agentdojo/{name}/` and also accepts legacy/copied logs in a root-level `{name}/` directory. It requires every requested AgentDojo or AgentDyn suite to have result rows before it exports that aggregate entry, so partial suite logs are not treated as complete leaderboard results.

## Results

Evaluation results are saved under:

```text
results/evaluation_results/{name}/
```

Standard runs also cache attack outputs under:

```text
results/evaluation_results/{name}/tmp_attack_results/
```

Each record stores the sample, defense result, utility, and ASR so interrupted runs can resume.

When the backend model is the default `Qwen/Qwen3-4B-Instruct-2507`, LLM-judge evaluation reuses the already loaded backend model instead of loading a separate default judge. This avoids duplicating the Qwen model in memory during search-based runs that also keep an attacker vLLM engine alive.

## Agent Benchmarks

PIArena also includes agent benchmark entry points:

- `main_injecagent.py`
- `main_agentdojo.py`

Minimal setup:

```bash
git submodule update --init --recursive
cd agents/agentdojo && pip install -e . && cd ../..
```

`main_agentdojo.py` now covers both the original AgentDojo suites and the merged AgentDyn suites inside the vendored `agents/agentdojo` tree.

Examples:

```bash
python main_injecagent.py --model meta-llama/Llama-3.1-8B-Instruct --defense none
python main_agentdojo.py --model gpt-5-mini --attack none --suite workspace
python main_agentdojo.py --model gpt-4o-2024-08-06 --attack important_instructions --defense datafilter --suite shopping
python main_agentdojo.py --config configs/experiments/modernbert_tagger.yaml --attack important_instructions --suite shopping
python main_agentdojo.py --config configs/experiments/modernbert_tagger.yaml --attack important_instructions --suite shopping --force-rerun
```

Suite mapping:

- AgentDojo: `workspace`, `slack`, `travel`, `banking`
- AgentDyn: `shopping`, `github`, `dailylife`

Defense routing:

- PIArena defenses such as `datafilter`, `pisanitizer`, and `promptguard` still run through the vendored PIArena adapter inside `agentdojo`
- Configured PIArena defenses such as `deepseek_pisanitizer` and `modernbert_tagger` pass `defense_config` through the same adapter
- The PIArena adapter sanitizes each complete serialized tool output before the agent sees it, including consecutive tool messages and tool error fields; result logs include `piarena_defense_events` with original and filtered tool outputs for debugging
- benchmark-native defenses such as `tool_filter`, `repeat_user_prompt`, `piguard_detector`, and `prompt_guard_2_detector` can be selected directly with `--defense`

## How Evaluation Is Chosen

In `main.py`, evaluator selection depends on dataset name:

- `open_prompt_injection` uses `llm_judge` and `open_prompt_injection_utility`
- `sep` uses `llm_judge`
- `knowledge_corruption` uses substring matching
- long-context datasets use `llm_judge` for ASR and task-specific LongBench metrics for utility

This is why the same attack or defense can be evaluated differently depending on the dataset.
