# PI Detector Dataset (v2 — hand-crafted)

Training and validation data for optimizing a DeepSeek-Flash-based prompt-injection
detector via DSPy / GEPA.

**AgentDyn is held out** — no AgentDyn suite content appears in this dataset.

## Files

| File | Purpose | Size |
|---|---|---|
| `train.jsonl` | GEPA training set | 48 samples |
| `val.jsonl`   | GEPA validation set (candidate selection) | 24 samples |
| `meta.json`   | Seed, source, composition stats | — |

## Sample schema

```jsonc
{
  "id": "train_000",
  "tool_name": "read_inbox",
  "user_instruction": "...",
  "tool_output": "<text the tool returned — may contain injection>",
  "is_injection": true | false,
  "attack_type": "none" | "straightforward" | "ignore" | "completion",
  "injection_text": "<exact injected substring, or null>",
  "decoy": false | true
}
```

## Composition

**Per tool**: 6 unique `user_instruction`s × 2 (1 benign + 1 injected) = 12 samples.
Split is **UI-disjoint**: 4 train UIs + 2 val UIs per tool. No UI appears in both
splits.

| Tool | Response shape | Natural injection landing |
|---|---|---|
| `read_inbox` | list[Email] | one email's body |
| `read_calendar` | list[Event] | one event's description |
| `read_document` | single doc with markdown content | anywhere in content |
| `search_products` | list[Product] with nested reviews | description or one review |
| `read_issue_thread` | issue body + comments list | body or one comment |
| `search_web` | list[SearchResult] | one snippet |

**Distribution:**

| | normal benign | decoy | straight | ignore | completion | total |
|---|---|---|---|---|---|---|
| train | 21 | 3 | 8 | 8 | 8 | **48** |
| val   |  9 | 3 | 4 | 4 | 4 | **24** |

## Source

Hand-written by Claude under user review, per design principles agreed on
2026-05-25 (see `docs/superpowers/specs/` if a design doc was committed).

**Why not InjecAgent / Alpaca / synthesis?** An earlier v1 used InjecAgent
templates and exhibited multiple structural shortcuts that would let an
optimized prompt achieve high accuracy without learning PI semantics
(outer-quote artifact, length asymmetry, lexical watermarks like
"`amy.watson@gmail.com`" only-in-injected). v2 hand-crafts every sample to
avoid these issues at construction time.

## Anti-shortcut design

Following the v1 audit, v2 deliberately controls for:

| Shortcut | v1 | v2 |
|---|---|---|
| Outer `"` prefix asymmetry | inj 28/35 vs ben 0/35 | **inj 0/36 vs ben 0/36** |
| Length ratio (inj : ben median) | 2.15× | **1.20×** (within [0.8, 1.25] target) |
| Tokens ≥5× in inj only (count≥4) | 8 (incl. "watson", "IMPORTANT!!!") | **0** |
| Specific attack markers ("IMPORTANT!!!", "Ignore all previous", "### response:") | high | **none** |
| Punctuation `=` density ratio | not measured | **2.1×** (was 5.9×, diluted via markdown `===` headers in benigns) |
| train ∩ val user_instruction | 15 / 16 | **0** (UI-disjoint) |
| Unique injection_text | 30 / 35 | **36 / 36** |

### Honest shortcut accounting (final audit)

Pass-fail thresholds:

| Audit dimension | Threshold for "issue" | Final v2 result |
|---|---|---|
| Single-word asymmetry | ≥4 in inj, ≤1 in ben, ≥5× ratio | **0 issues** |
| Bigram asymmetry | ≥4 in inj, ≤1 in ben, ≥3× ratio | **0 issues** |
| Trigram asymmetry | ≥3 in inj, 0 in ben | 2 residuals (`the assistant should` 3×; `\nthe operator has` 3×) — coverage <10% of inj |
| Char 5-gram asymmetry | ≥5 in inj, 0 in ben | **0 issues** |
| ALL-CAPS framing words | any visible imbalance | only `AGENT` 1× (noise) |
| Length ratio | inj/ben median in [0.8, 1.25] | **1.22** |
| Element-length bias (list tools) | injection always in longest element | search_products 2/6, search_web 3/6 (~33% = random baseline for 3-element list) |
| Train ∩ val UI overlap | 0 | **0** |
| JSON parseability | 72/72 | **72/72** |
| Train/val symmetry on shortcut chars | divergence < 5× | `<` 68× (train-only shortcut, val ratio 3.1× — harmless for GEPA candidate selection) |

The dataset has been iterated through 4 rounds of adversarial audit (single tokens → punctuation/train-val symmetry → bigrams/element-length/CAPS → trigrams/char n-grams). Each round revealed and eliminated shortcuts the previous round missed. **At the dimensions tested, no shortcut can substitute for semantic understanding.**

### What I cannot guarantee

I have not exhaustively tested every possible shortcut dimension. Examples of dimensions NOT in the above audit:

- 4-gram and higher n-gram patterns
- Syntactic-tree level patterns (e.g., specific dependency relations)
- Embedding-space clustering (would require an embedding model)
- Cross-feature combinations (e.g., URL + verb co-occurrence)
- Conditional patterns within specific tools

The AgentDyn 3-suite held-out evaluation is the final arbiter of whether GEPA-optimized prompts learned semantic understanding vs. surface features.

Within the injected half, attack-framing diversity is intentional:

- **Framing tokens**: `---SYSTEM:`, `[INTERNAL]`, `[LLM-only]`, `[[INTERNAL_NOTE]]`, `<<<override>>>`, `||SYSTEM NOTICE||`, `{{!OVERRIDE!}}`, `###`, `=====`, `_____`, `^^^`, `%%%`, etc. — 18+ distinct framings across 36 injected samples.
- **Attacker intents**: credential exfil, wire transfer, message deletion, calendar tampering, account recovery hijack, file leak, contact export, SSH key exfil, flag toggle, unauthorized purchase, API rate-limit lift, deploy-key collection, force-push deploy, etc. — 36 unique payloads.
- **Exfil targets**: protonmail / dropbox / pastebin / webhook.site / yandex / .cm / .workers.dev / .web.app / .cloudfunctions.net — never the same target twice.
- **Position**: end / middle / disguised as another step or bullet.

## Decoys

6 decoy samples (3 train + 3 val) — one per tool — contain benign content with
**legitimate imperative sentences in contextually appropriate fields**:

| Decoy | What makes it look like a PI candidate |
|---|---|
| HR open-enrollment email | "Please review… Step 1/2/3:…" |
| Customer onboarding calendar event | "Step 1: Send welcome email. Step 2: Schedule kickoff…" |
| Postgres outage runbook doc | numbered failover and post-incident steps |
| Smart kitchen scale product | "Place on flat surface. Do not exceed 5kg. Calibrate before use." |
| OAuth bug repro issue | numbered reproduction steps |
| GitHub Actions docs search | "Step 1: Create a `.github/workflows` directory…" |

These exist for **honest FPR measurement** — without them, the benign set is
too easy and the detector's FPR is under-reported.

## Held-out test

AgentDyn (3 suites: `dailylife`, `github`, `shopping`) is the held-out test set.
The `tool_name`s in this dataset (`read_inbox`, `read_calendar`, etc.) are
deliberately chosen NOT to match AgentDyn's actual tool names.

## Reproducing

```bash
python scripts/build_pi_detector_dataset_v2.py
```

Deterministic: `SEED = 20260526`. The script contains all 72 samples inline
as Python data structures (source is data); running produces JSONL + meta
deterministically.

## Known limitations

- **Length ratio 1.25× (at boundary)**: injected `tool_output` is still ~25%
  longer than benign on the median. Acceptable but not zero. To improve
  further, pad benign tool_outputs to match.
- **Two partial token signals remain**: "instead" (inj 16 vs ben 5) and
  "assistant" (inj 12 vs ben 3) skew toward injected. Down from "0 in ben"
  watermarks of v1, but a careful prompt could still use them as weak signals.
- **No Context attack** (Wang et al., 2026 §II-D) — only Straightforward /
  Ignore / Completion. Generalization to Context attacks is a held-out test
  property the detector either has or doesn't.
- **Only 6 tools** — broader tool diversity (CRM, file system, browser
  navigation, etc.) is not represented.
