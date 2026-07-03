---
title: ModernBERT Tagger
slug: defenses/modernbert-tagger
category: defenses
---

# ModernBERT Tagger

`modernbert_tagger` is a token-level sanitization defense from AgentDyn. It loads a ModernBERT token-classification checkpoint and removes spans predicted as prompt-injection content.

Source:
[piarena/defenses/modernbert_tagger/defense_modernbert_tagger.py](https://github.com/sleeepeer/PIArena/blob/main/piarena/defenses/modernbert_tagger/defense_modernbert_tagger.py)

## How To Use

Update `checkpoint_path` in `configs/experiments/modernbert_tagger.yaml`, then run:

```bash
python main.py --config configs/experiments/modernbert_tagger.yaml
```

For agent benchmarks:

```bash
python main_injecagent.py --config configs/experiments/modernbert_tagger.yaml
python main_agentdojo.py --config configs/experiments/modernbert_tagger.yaml --attack important_instructions --suite shopping
```

## What It Does

The defense serializes the trusted instruction and untrusted context, runs the ModernBERT tagger, removes predicted `DROP` spans from the context, and returns the result as `cleaned_context`.

`execute_batch()` uses the tagger backend batching path directly, so search-based attacks and batch response paths can reuse one loaded checkpoint.

## Parameters

- `checkpoint_path`: required path to a ModernBERT tagger checkpoint directory.
- `device`: PyTorch device. Default: `cuda`.
- `batch_size`: tagger inference batch size. Default: `8`.
