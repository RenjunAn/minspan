---
title: DeepSeek PISanitizer
slug: defenses/deepseek-pisanitizer
category: defenses
---

# DeepSeek PISanitizer

`deepseek_pisanitizer` is an LLM-based sanitizer for untrusted tool or context outputs. It uses the AgentDyn DeepSeek sanitizer prompt and returns a cleaned context before the backend model is queried.

Source:
[piarena/defenses/deepseek_pisanitizer/defense_deepseek_pisanitizer.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/deepseek_pisanitizer/defense_deepseek_pisanitizer.py)

## How To Use

```bash
export DEEPSEEK_API_KEY="..."
python main.py --config configs/experiments/deepseek_pisanitizer.yaml
```

For agent benchmarks:

```bash
python main_injecagent.py --config configs/experiments/deepseek_pisanitizer.yaml
python main_agentdojo.py --config configs/experiments/deepseek_pisanitizer.yaml --attack important_instructions --suite shopping
```

## What It Does

The defense sends the trusted user instruction and untrusted context to an OpenAI-compatible DeepSeek endpoint. The sanitizer must return JSON with `filtered_tool_output`; PIArena then uses that value as `cleaned_context`.

If the API call fails or the response cannot be parsed, the original context is preserved and the result marks `api_ok` or `parse_ok` as false.

## Parameters

- `api_key`: optional direct API key value. Prefer private configs for this.
- `api_key_env`: environment variable to read when `api_key` is not set. Default: `DEEPSEEK_API_KEY`.
- `base_url`: OpenAI-compatible endpoint. Default: `https://api.deepseek.com`.
- `model`: sanitizer model. Default: `deepseek-v4-flash`.
- `temperature`: generation temperature. Default: `0.0`.
- `max_tokens`: optional response token limit.
- `response_format`: default JSON object response format.
- `tool_name`: weak context label for the sanitized input. Default: `piarena_context`.
