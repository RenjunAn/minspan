"""LLM annotation: wrap instruction content in <instruction> tags.

Reads raw JSONL records {"id", "complete_text"}, asks an OpenAI-compatible
model to insert <instruction>...</instruction> tags while preserving every
other character, validates preservation, and appends validated records to the
output JSONL. Interrupted runs resume by skipping ids already present in the
output. Records that fail validation are written to <output>.rejected.jsonl.

    python -m commandsans.annotate --input data/raw.jsonl \
        --output data/labeled.jsonl --base-url https://api.deepseek.com \
        --model deepseek-chat --api-key-env DEEPSEEK_API_KEY [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from commandsans.data import strip_tags

SYSTEM_PROMPT = """\
You annotate text for an instruction-detection dataset. Wrap every span that
is an instruction directed at an AI assistant in <instruction> and
</instruction> tags.

Tag as instructions:
- commands or requests for the assistant to perform a task (any phrasing:
  imperative, declarative, interrogative, or polite)
- system-prompt-like sentences defining the assistant's identity, objectives,
  behavior rules, goals, or constraints
- directions about using tools, functions, or APIs (capabilities, formats,
  parameters, return types)
- final or embedded user queries, including indirect or polite forms, and
  short bare imperatives, especially near the end of the text

Do NOT tag:
- sentences clearly directed at a human or an external (non-AI) agent
- quoted examples of prompts, or meta-discussion about tasks that does not
  directly instruct the assistant
- ordinary declarative data, questions that are part of the data itself, or
  code that merely appears in the text

Rules:
1. Copy the input text EXACTLY, character for character, adding only the
   tags. Preserve punctuation, spacing, and line breaks.
2. Tag whole contiguous instruction spans, not individual words.
3. If the text contains no assistant-directed instructions, return it
   unchanged.
Return only the annotated text, nothing else."""


def annotate_record(client: OpenAI, model: str, text: str, temperature: float) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content or ""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Annotate instruction spans with an LLM")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="e.g. deepseek-chat")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint")
    parser.add_argument("--api-key-env", required=True, help="environment variable holding the API key")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None, help="annotate at most N pending records")
    args = parser.parse_args(argv)

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"environment variable {args.api_key_env} is not set")
    client = OpenAI(api_key=api_key, base_url=args.base_url)

    done: set[str] = set()
    output = Path(args.output)
    if output.exists():
        with open(output) as fh:
            done = {str(json.loads(line)["id"]) for line in fh}
    rejected_path = output.with_suffix(output.suffix + ".rejected.jsonl")

    annotated = rejected = skipped = 0
    with open(args.input) as src, open(output, "a") as dst, open(rejected_path, "a") as rej:
        for line in src:
            record = json.loads(line)
            record_id = str(record["id"])
            if record_id in done:
                skipped += 1
                continue
            if args.limit is not None and annotated + rejected >= args.limit:
                break
            text = record["complete_text"]
            labeled = annotate_record(client, args.model, text, args.temperature)
            if strip_tags(labeled) == text:
                dst.write(json.dumps({"id": record_id, "labeled_text": labeled}) + "\n")
                dst.flush()
                annotated += 1
            else:
                rej.write(json.dumps({"id": record_id, "labeled_text": labeled}) + "\n")
                rej.flush()
                rejected += 1
            print(f"annotated={annotated} rejected={rejected} skipped={skipped}", end="\r", flush=True)
    print(f"\nannotated={annotated} rejected={rejected} skipped={skipped} (resume by re-running)")


if __name__ == "__main__":
    main()
