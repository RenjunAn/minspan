"""Extract Appendix G of the AgentDyn paper into a long-format CSV/JSON.

Run from this directory after `_paper.html` has been downloaded:

    curl -sL "https://arxiv.org/html/2602.03117v3" -o _paper.html
    python3 parse_appendix_g.py

Produces:
    appendix_g.csv
    appendix_g.json

See README.md in this directory for the schema and column-order rationale.
"""

import csv
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
HTML = (HERE / "_paper.html").read_text()

# --- locate Appendix G figure ------------------------------------------------

_APPENDIX_START = HTML.find('<section id="A7"')
_APPENDIX_HTML = HTML[_APPENDIX_START:]
_FIGURE_HTML = re.search(
    r"<figure[^>]*>(.*?)</figure>", _APPENDIX_HTML, re.DOTALL
).group(1)

# --- sub-table layout (verified empirically: col1 = mean(col2..4)) -----------

BLOCKS: dict[str, list[str]] = {
    "benign_utility":       ["A7.T17.1.1",  "A7.T17a.1.1", "A7.T17b.1.1"],
    "utility_under_attack": ["A7.T17c.1.1", "A7.T17d.1.1", "A7.T17e.1.1"],
    "asr":                  ["A7.T17f.1.1", "A7.T17g.1.1", "A7.T17h.1"],
}
SUITES = ["overall", "shopping", "github", "dailylife"]

MODEL_CANONICAL = {
    "GPT-4o mini": "gpt-4o-mini",
    "GPT-4o": "gpt-4o",
    "GPT-5.1": "gpt-5.1",
    "GPT-5-mini": "gpt-5-mini",
    "Gemini-2.5 Pro": "gemini-2.5-pro",
    "Gemini-2.5 Flash": "gemini-2.5-flash",
    "Claude-Sonnet-3.5": "claude-sonnet-3.5",
    "Claude-Sonnet-4.5": "claude-sonnet-4.5",
    "Qwen3 235B-A22B": "qwen3-235b",
    "Llama 3.3 70B": "llama-3.3-70b",
    "Qwen3-Coder": "qwen3-coder",
    "Kimi-K2.5": "kimi-k2.5",
    "Meta-SecAlign 70B": "meta-secalign-70b",
    "Meta-SecAlign 8B": "meta-secalign-8b",
}

DEFENSE_CANONICAL = {
    "None": "none",
    "Prompt Sandwiching": "repeat_user_prompt",
    "Spotlighting": "spotlighting_with_delimiting",
    "ProtectAI": "transformers_pi_detector",
    "PIGuard": "piguard_detector",
    "PromptGuard2": "prompt_guard_2_detector",
    "Meta-SecAlign": "meta_secalign",
    "Tool Filter": "tool_filter",
    "CaMeL": "camel",
    "Progent": "progent",
    "DRIFT": "drift",
}


def _parse_subtable(table_id: str) -> list[tuple[str, str, list[str]]]:
    """Return [(defense, model, [overall, shopping, github, dailylife]), ...]."""
    m = re.search(
        r'<table id="' + re.escape(table_id) + r'"[^>]*>(.*?)</table>',
        _FIGURE_HTML,
        re.DOTALL,
    )
    if not m:
        return []
    body = m.group(1)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.DOTALL)
    out: list[tuple[str, str, list[str]]] = []
    current_defense: str | None = None
    for row in rows:
        cells: list[tuple[int, str]] = []
        for cm in re.finditer(r"<td([^>]*)>(.*?)</td>", row, re.DOTALL):
            rs = re.search(r'rowspan="(\d+)"', cm.group(1))
            rowspan = int(rs.group(1)) if rs else 0
            txt = re.sub(r"<[^>]+>", " ", cm.group(2))
            txt = re.sub(r"\s+", " ", txt).strip()
            cells.append((rowspan, txt))
        if not cells:
            continue
        offset = 0
        if cells[0][0] > 1:
            current_defense = cells[0][1]
            offset = 1
        if len(cells) - offset >= 5:
            model = cells[offset][1]
            vals = [cells[offset + 1 + i][1] for i in range(4)]
            out.append((current_defense or "", model, vals))
    return out


def build_records() -> list[dict]:
    records: list[dict] = []
    for metric, table_ids in BLOCKS.items():
        for tid in table_ids:
            for defense_raw, model_raw, vals in _parse_subtable(tid):
                for suite, raw in zip(SUITES, vals):
                    try:
                        value_pct: float | None = float(raw)
                    except ValueError:
                        value_pct = None
                    records.append(
                        {
                            "model": MODEL_CANONICAL.get(model_raw, model_raw),
                            "model_paper": model_raw,
                            "defense": DEFENSE_CANONICAL.get(defense_raw, defense_raw),
                            "defense_paper": defense_raw,
                            "metric": metric,
                            "suite": suite,
                            "value_pct": value_pct,
                            "value_raw": raw,
                        }
                    )
    return records


def main() -> None:
    records = build_records()
    print(f"Extracted {len(records)} records")
    csv_path = HERE / "appendix_g.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    json_path = HERE / "appendix_g.json"
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Wrote {csv_path.name} and {json_path.name}")


if __name__ == "__main__":
    main()
