# AgentDyn: Are Your Agent Security Defenses Deployable in Real-World Dynamic Environments?

[Hao Li](https://leolee99.github.io/), [Ruoyao Wen](https://github.com/ruoyaow/), [Shanghao Shi](https://shishishi123.github.io/), [Ning Zhang](https://cybersecurity.seas.wustl.edu/index.html), [Yevgeniy Vorobeychik](https://vorobeychik.com/), [Chaowei Xiao](https://xiaocw11.github.io/).

The official implementation of the paper "[AgentDyn: Are Your Agent Security Defenses Deployable in Real-World Dynamic Environments?](https://arxiv.org/pdf/2602.03117)".

AgentDyn is a dynamic, open-ended agent security benchmark featuring 60 challenging open-ended user tasks and 560 injection test cases across the Shopping, GitHub, and Daily Life scenarios. It is built on top of the [AgentDojo](https://github.com/ethz-spylab/agentdojo) framework. A huge thanks to the AgentDojo team for their admirable contribution to the community!

## Quickstart

```bash
pip install -e .
```


## Running the benchmark

For adaptability, we support an evaluation script same as AgentDojo's. Documentation on how to use the script can be obtained with the `--help` flag.

For example, to run the `shopping` suite , with `gpt-4o-2024-08-06` as the LLM, the tool filter as a defense, and the attack with important_instructions, run the following command:

```bash
python -m agentdojo.scripts.benchmark -s shopping \
    --model GPT_4O_2024_08_06 \
    --defense tool_filter --attack important_instructions
```

To run with external defenses integrated in this repo layout, you can directly use:

### Token tagger defenses

AgentDyn includes two token-level tool-output sanitizers. Both receive the
original user task and the complete JSON-serialized tool output, then remove
tokens predicted as prompt injection before the agent model sees the result.

Install the local-model dependencies:

```bash
uv sync --extra transformers
```

The DataFilter bidirectional checkpoint contains only the trained head and
tokenizer. It therefore needs the separate DataFilter backbone:

```bash
export DATAFILTER_TAGGER_CHECKPOINT=/models/datafilter-bidir/best
export DATAFILTER_BACKBONE_MODEL=/models/DataFilter
```

The ModernBERT checkpoint contains the complete fine-tuned model:

```bash
export MODERNBERT_TAGGER_CHECKPOINT=/models/modernbert-tagger/best
```

Shared runtime options default to:

```bash
export TAGGER_DEVICE=cuda
export DATAFILTER_TAGGER_BATCH_SIZE=1
export MODERNBERT_TAGGER_BATCH_SIZE=8
```

`TAGGER_BATCH_SIZE` remains available as a shared fallback when a
model-specific batch size is not set. The DataFilter default is intentionally
one on a 3090. If a multi-item batch fails, the backend recursively splits it
so only an individually failing tool output is passed through unchanged.

The DataFilter loader verifies that `DATAFILTER_BACKBONE_MODEL` and the optional
`DATAFILTER_BACKBONE_REVISION` match the values recorded during training. For
an intentional relocation or replacement, opt out explicitly:

```bash
export DATAFILTER_ALLOW_BACKBONE_MISMATCH=1
```

Run one smoke task:

```bash
uv run python -m agentdojo.scripts.benchmark \
    -s shopping -ut user_task_0 -it injection_task_0 \
    --model deepseek-v4-flash \
    --defense modernbert_tagger \
    --attack important_instructions \
    --tool-output-format json -f
```

Replace `modernbert_tagger` with `datafilter_bidir_tagger` for the frozen
DataFilter backbone model. Run both defenses across AgentDyn's three suites
and both clean/attacked settings with:

```bash
bash scripts/run_defense_token_taggers.sh
```

Keep `--max-workers 1` on a single 3090. Multiple benchmark workers would load
duplicate model copies onto the same GPU.

Token-tagger result directories include a short checkpoint fingerprint, so
different checkpoint contents do not share cached benchmark results. The
runner forces reruns by default and returns a failure status if any trace
contains a failed tagger inference. These checks can be relaxed explicitly:

```bash
export FORCE_RERUN=0
export ALLOW_TAGGER_FAILURES=1
```


Before running, please export your API key, through:

1. OpenAI Model: export OPENAI_API_KEY=XXX
2. Google Model: export GOOGLE_API_KEY=XXX
3. Open-sourced Models (Qwen, LlaMA, other supported models): export OPENROUTER_API_KEY=XXX

## Supported settings

#### Available Suites:
AgentDyn supports `shopping`,`github`, and `dailylife` suites, as well as the original four suites from AgentDojo (`banking`,`slack`, `travel` and `workspace`).

#### Available Models: 
We evaluate the following models in our paper: ``GPT_4O_MINI_2024_07_18``, ``GPT_4O_2024_08_06``, ``GEMINI_2_5_FLASH``, ``GEMINI_2_5_PRO``, ``LLAMA_3_3_70B``, ``QWEN3_235B``, ``GPT_5_1_2025_11_13``, ``GPT_5_MINI_2025_08_07``. 
Other models supported by AgentDojo are also compatible.

#### Available Defenses: 
In addition to the original defenses in AgentDojo, we provide support for [PIGuard](https://aclanthology.org/2025.acl-long.1468.pdf) and [PromptGuard2](https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M). We also support directly invoking external defenses in this workspace: [CaMeL](https://github.com/google-research/camel-prompt-injection), [Progent](https://github.com/sunblaze-ucb/progent), [DRIFT](https://github.com/SaFo-Lab/DRIFT),.

The complete list of defenses supported in our paper includes: ``repeat_user_prompt``, ``spotlighting_with_delimiting``, ``tool_filter``, ``transformers_pi_detector``, ``piguard_detector``, ``prompt_guard_2_detector``, ``camel``,  ``progent``, ``drift``.


## Inspect Results

To review the results reported in our paper, please refer to the log files in the ``(runs/)``.

## References

If you find this work useful in your research or applications, we appreciate that if you can kindly cite:

```
@articles{AgentDyn,
  title={AgentDyn: Are Your Agent Security Defenses Deployable in Real-World Dynamic Environments?},
  author={Hao Li and Ruoyao Wen and Shanghao Shi and Ning Zhang and Yevgeniy Vorobeychik and Chaowei Xiao},
  journal = {arXiv},
  eprint = {2602.03117},
  year={2026}
}
```
