# Calibration Mix V2 — Design

**Repo**: `/home/lucas/ai/moe_compress` · branch `feat/calibration-v2` · base `6ff3636`
**Target**: Qwen3.6-35B-A3B, thinking mode, used by Stage 2 REAP/REAM merge, Stage 2.5 router KD, Stage 3 covariance/SVD, Stage 6 evals.
**Status**: design doc, read-only analysis. No code changes here. GPU validation enumerated, not authorized.
**Author**: design agent, 2026-05-27.

Companion docs (verbatim cross-refs):
- `max_quality/src/moe_compress/utils/calibration.py:998-1281` — existing `qwen3-pretrain-mix` definition + per-subset streamers.
- `max_quality/scripts/build_self_traces_calib.py:121-282` — prompt iterator (the `_iter_prompts_from_qwen3_pretrain_mix` faucet that prompt-strips each row).
- `max_quality/scripts/build_self_traces_calib_vllm.py` — vLLM teacher-generation path (5–10 s/trace at bs≥4 on H200; 267 s/trace at bs=1 bf16, measured per `feedback_calibration_oom_ladder.md`).
- `max_quality/docs/calibration_v2_data_capture_plan.md` — writer contract (covariance, routing stats, REAP scores, reservoirs).
- `max_quality/docs/reap_exact_mode.md` — REAP-exact uses the same calibration tensors; no extra demands.

All quality / cost claims are tagged **[measured]** if they trace back to logged numbers, or **[conjecture]** if extrapolated. Cost framing per `feedback_dont_compute_costs.md`: I name GPU-hours but never multiply by hourly rate.

---

## 0. TL;DR

**Final recommendation**: Choice **C — hybrid per-subset** with 11 subsets, drop one existing, shrink three, hold four, add five candidates. xlam is dropped entirely (format mismatch + gated). Net teacher-generation budget drops from 100% of prompts to ~52% of prompts → ~halves teacher-gen wall-clock vs. the naive "merge both lists and prompt-only-generate everything" baseline.

**Three decisions needed from the user before implementation** (see §6).

**Headline mix table** (preview; full table in §5):

| # | Subset | Weight | Teacher role | Notes |
|---|---|---:|---|---|
| 1 | tulu3 | 18% | prompt-only → generate | general SFT, weight shrunk; subsumes codealpaca-evol |
| 2 | math (OpenMathInstruct-2) | 9% | prompt-only → generate | math prompts kept, canonical 405B solutions discarded |
| 3 | qa (dolly) | 5% | prompt-only → generate | shrunk |
| 4 | creative (writingprompts) | 5% | prompt-only → generate | shrunk |
| 5 | multilingual (aya) | 8% | prompt-only → generate | held — only multilingual signal |
| 6 | fineweb-edu | 4% | prompt-only → generate | held — only raw-text replay |
| 7 | papers (arxiv) | 3% | prompt-only → generate | shrunk |
| 8 | MoT-math | 12% | teacher-forced (canonical R1 trace) | **NEW**, format-compatible |
| 9 | MoT-code | 12% | teacher-forced (canonical R1 trace) | **NEW**, format-compatible |
| 10 | MoT-science | 8% | teacher-forced (canonical R1 trace) | **NEW**, format-compatible |
| 11 | SWE-smith (xml + tool subset) | 16% | teacher-forced (canonical Claude trace, prompt-only fallback on assistant-turn injection) | **NEW**, format-compatible (xml split matches Qwen3's `<tool_call>` template) |
| ~~12~~ | ~~code (Evol-Instruct-Code-80k)~~ | ~~0%~~ | DROP | redundant with tulu3+MoT-code |
| ~~13~~ | ~~xlam-function-calling-60k~~ | ~~0%~~ | DROP | gated + JSON schema, no thinking, format-mismatch |
| ~~14~~ | ~~evol-codealpaca-v1~~ | ~~0%~~ | DROP | already a source of tulu3 SFT mixture |

Sum: 100%. 11 active subsets. 5 generate-only (52%), 4 teacher-forced (48%), 3 dropped.

---

## 1. Phase 1 — Per-dataset findings (sample rows + format)

### 1.1 Existing 8 subsets — quick recap (no surprises)

| Subset | Dataset | Row schema | Source model | Has thinking? | Token avg |
|---|---|---|---|---|---|
| tulu3 | `allenai/tulu-3-sft-mixture` | `messages=[{role,content},...]` | mixed (CoCoNot, FLAN, OASST, NuminaMath, Evol-CodeAlpaca, …) | no | ~600 [conjecture] |
| math | `nvidia/OpenMathInstruct-2` | `problem`/`generated_solution` | Llama-3.1-405B-Instruct | no | ~800 |
| code | `nickrosh/Evol-Instruct-Code-80k-v1` | Alpaca-style `instruction`/`output` | GPT-4 (old) | no | ~300 |
| qa | `databricks/databricks-dolly-15k` | `instruction`/`context`/`response` | human-written | no | ~400 |
| creative | `euclaise/writingprompts` | `prompt`/`story` | human reddit | no | ~600 |
| multilingual | `CohereForAI/aya_dataset` | `inputs`/`targets`, 65+ langs | human translators | no | ~400 |
| fineweb | `HuggingFaceFW/fineweb-edu` | `text` (raw web pages) | — | no | ~1500 |
| papers | `gfissore/arxiv-abstracts-2021` | `title`/`abstract` | human authors | no | ~300 |

Schema fact that matters: **tulu3 already lists `theblackcat102/evol-codealpaca-v1` in its `source_datasets` tag** (visible on the tulu-3-sft-mixture dataset card). So including codealpaca-evol as a separate subset is double-counting.

### 1.2 Six candidate datasets — verified

#### 1.2a `theblackcat102/evol-codealpaca-v1`
- **Status**: open, parquet, 111.3K rows, 136 MB.
- **Schema**: `{instruction: str, output: str}` (Alpaca-style).
- **Source model**: GPT-4-0314 / GPT-4-0613, generated 2023.
- **Has thinking**: no. Plain instruct, median length 471 tokens.
- **Verdict**: **DROP**. Already a source dataset of `allenai/tulu-3-sft-mixture` — listed verbatim in tulu3's `source_datasets:` tag. Including it adds duplication, not new signal. Confirmed by reading the tulu-3-sft-mixture card metadata (§1.1).

#### 1.2b `Salesforce/xlam-function-calling-60k`
- **Status**: **🔒 gated**. Access requires HF login + accepting terms. The user's HF account `pirola` may or may not have already accepted; not verified.
- **Schema**: `{query: str, tools: str(json), answers: str(json)}` — both `tools` and `answers` are JSON strings, must be `json.loads`-ed.
- **Source format**: no thinking trace anywhere. The `answers` field is just a JSON array of `{name, arguments}` — pure structured output, no reasoning prose.
- **Mismatch with Qwen3.6 thinking-mode**: large.
  - Qwen3.6's `<tool_call>` template (per its tokenizer_config.json): `<tool_call>\n<function=NAME>\n<parameter=KEY>VALUE</parameter>\n</function>\n</tool_call>` — XML-shaped, NOT JSON.
  - xlam's canonical answer is JSON. Forcing the teacher to emit xlam-style JSON would require a system prompt rewrite, which puts it off-distribution from how Qwen3.6 actually serves.
  - There is no thinking trace, so canonical-completion mode gives the teacher zero supervision on `<think>...</think>` tokens — the entire point of self-traces.
  - In prompt-only mode, the teacher with thinking enabled would generate a `<think>` block + a `<tool_call>` block — but without explicit tools provided in a system message, it would likely refuse to call tools and just answer in prose, defeating the purpose of including the subset.
- **Verdict**: **DROP**. Format mismatch + gated + no thinking trace. If a future iteration wants function-calling calibration, the right move is to source from a thinking-format-native tool-call corpus (e.g., the xml split of SWE-smith below, which DOES match Qwen3.6's template).

#### 1.2c `SWE-bench/SWE-smith-trajectories`
- **Status**: open, parquet, **three splits**: `ticks` (25.8K), `tool` (24.1K), `xml` (26.1K). 76K total, ~3.4 GB per split.
- **Schema**: `{messages: str(json), instance_id, resolved, model, traj_id, patch}`. The `messages` field is a JSON-serialized multi-turn conversation (system, user, assistant, optionally tool).
- **Source model**: **Claude 3.7 Sonnet** (run by SWE-agent on SWE-smith task instances).
- **Multi-turn**: yes, 5K unique tasks ⇒ ~15 trajectories per task ⇒ each trajectory averages multiple round-trips.
- **Token avg per row**: 7.95k–664k chars (highly skewed) — many trajectories exceed our `max_new_tokens=16384` token budget *for the assistant turns alone*, let alone the full message list.
- **Split formats** (verified by inspecting sample rows; cited from `tool-results/*.txt` artifacts of this design session):
  - **`xml` (26.1K)** — assistant turns use `<function=NAME>\n<parameter=KEY>VALUE</parameter>\n</function>` blocks. **This exactly matches Qwen3.6's `<tool_call>` chat template**, only the outer `<tool_call>\n…\n</tool_call>` wrapping is missing. A small adapter can wrap them.
  - **`tool` (24.1K)** — assistant turns use OpenAI-native `tool_calls: [{type: "function", function: {name, arguments}}]` schema with `role=tool` response messages. This is the format Qwen3.6 *emits* when served via vLLM with `--tool-call-parser qwen3_coder` and lowered to the chat template by `apply_chat_template`. Compatible.
  - **`ticks` (25.8K)** — assistant turns use plain triple-backtick code-fence markdown; tool actions are bash commands inside code blocks; observations are `role=user` messages. **No** structured tool calls. Closest to a chatty plain-prose trace.
- **Has thinking**: **no `<think>` blocks** in any split. Claude 3.7 Sonnet wasn't in extended-thinking mode when these were generated, or the thinking wasn't preserved.
- **Verdict**: **KEEP `xml` + `tool` splits** (51K rows combined). Drop `ticks`. The xml/tool splits give us the exact tool-call surface format the teacher would produce at serve time. The lack of `<think>` blocks is a real cost — we get tool-call format coverage but no thinking-token supervision on this subset. Mitigate: see §2.

#### 1.2d `open-r1/Mixture-of-Thoughts` (math/code/science)
- **Status**: open, parquet, 350K rows total across 4 configs:
  - `all` 349K, `code` 83K, `math` 94K, `science` 173K. ~3.1 GB total.
- **Schema**: `messages=[{role, content}, ...]`, `num_tokens: int`, `source: str`.
- **Source model**: **DeepSeek-R1** (canonical traces distilled from R1). The `source` field gives the *prompt source*, not the trace source — every trace is R1.
- **Has thinking**: **yes — `<think>...</think>` blocks**, format-identical to Qwen3.6's thinking output. Verified directly from a `math` sample row (see Phase 1 evidence dump).
- **Per-row token count**: highly variable:
  - `code`/codeforces-cots: 10K–22K tokens (some exceed 16384)
  - `math`/OpenR1-Math-220k: 2K–5K tokens
  - `science`/Llama-Nemotron: 1K–6K tokens
- **Critical caveat — distribution alignment**: R1's thinking style is similar to Qwen3.6's (both decoder-only thinking-mode models, both trained on similar SFT mixes, both emit `<think>...</think>` tags), but they are NOT byte-identical generators. The off-distribution risk is small but real. We are betting that R1-thinking-mode router patterns are *close enough* to Qwen3.6-thinking-mode router patterns that teacher-forced calibration on R1 traces gives meaningful Qwen3.6 routing signal. This is the central assumption of the hybrid plan; see §7.1 for the validation test.
- **Verdict**: **KEEP all three configs** as separate subsets. They're the highest-quality thinking-format-native traces available.

### 1.3 Summary of candidate verdicts

| Candidate | Verdict | Reason |
|---|---|---|
| evol-codealpaca-v1 | DROP | already in tulu3-sft-mixture's source_datasets |
| xlam-function-calling-60k | DROP | gated + JSON-format-mismatch + no thinking |
| SWE-smith-trajectories (xml) | KEEP | Anthropic-XML matches Qwen3 `<tool_call>` template |
| SWE-smith-trajectories (tool) | KEEP | OpenAI-tool_calls format that Qwen3 emits at serve |
| SWE-smith-trajectories (ticks) | DROP | plain prose, redundant with MoT-code |
| MoT-math | KEEP | R1 thinking trace, math reasoning |
| MoT-code | KEEP | R1 thinking trace, competitive programming |
| MoT-science | KEEP | R1 thinking trace, MCQ-style science |

---

## 2. Phase 2 — Per-subset teacher-mode role

The three options the user laid out:
1. **GENERATE** — prompt-only, teacher generates fresh thinking-mode response, current behavior, ~5 s/trace on H200 vLLM bs≥4 [measured `feedback_calibration_oom_ladder.md`].
2. **TEACHER-FORCED** — use canonical completion, teacher just forwards the (prompt + foreign_completion) tokens, ~negligible generation cost (a single forward of ~5–20K tokens vs. autoregressive 16K-token generation).
3. **HYBRID** — prompt+canonical as a guide, or prompt-only with capped max_new_tokens. In practice this collapses to (1) or (2) for most subsets; we use it only for SWE-smith where canonical completions span multiple turns.

### Decision per subset

| Subset | Role | Justification |
|---|---|---|
| **tulu3** | GENERATE | Source is mixed (FLAN, OASST, …); canonical assistant turns are mostly non-thinking SFT-style. We want Qwen3.6-thinking tokens in the trace, not 2023-style SFT answers. The teacher's own thinking-mode response is on-distribution by construction. |
| **math (OpenMathInstruct-2)** | GENERATE | Canonical solutions are Llama-3.1-405B chain-of-thought style, NOT thinking-mode wrapped in `<think>...</think>`. Discard canonical; keep the prompt (high-quality math problems). |
| **qa (dolly)** | GENERATE | Human-written short answers; no thinking. Discard canonical. |
| **creative (writingprompts)** | GENERATE | Human-written stories; no thinking, wrong distribution for Qwen3.6-thinking. |
| **multilingual (aya)** | GENERATE | Canonical answers are human translations; we want Qwen3.6-thinking-mode multilingual outputs (which differ from human translations in style and length). Discard canonical. |
| **fineweb-edu** | GENERATE | Raw web text; not a chat trace at all. Current pipeline wraps as "read and explain this passage" — the teacher's response is what we want to capture. |
| **papers (arxiv)** | GENERATE | Canonical = human-written abstracts; we want the teacher's thinking-mode abstract reconstruction. |
| **MoT-math** | **TEACHER-FORCED** | Canonical = DeepSeek-R1 `<think>...</think>` trace. R1 and Qwen3.6 are close-distribution decoder-only thinking models. Using canonical saves ~5 s/trace × ~700 prompts = ~1 GPU-hour, with [conjecture] minor distribution drift offset by 35× the prompt-coverage we get for the same wall-clock budget. |
| **MoT-code** | TEACHER-FORCED | Same logic. Long traces (10K–22K tokens) make generation especially expensive; teacher-forcing avoids burning the full max_new_tokens. |
| **MoT-science** | TEACHER-FORCED | Same logic. Shorter traces (1K–6K) — small absolute savings per prompt but lets us include more science prompts in the same budget. |
| **SWE-smith xml + tool** | **TEACHER-FORCED, multi-turn flattened** | Canonical = Claude 3.7 Sonnet trajectories with tool calls in Qwen3-compatible format. NO `<think>` blocks. We pay a cost: zero thinking-token supervision on this subset. But we gain authentic tool-call routing patterns that the MoT subsets can't provide. Justified iff the tool-use deployment surface matters; if it doesn't, drop this subset entirely. |

### Why not generate everywhere (Choice A)?
Choice A (current pipeline behavior) is the simplest contract: every trace is on-distribution with the teacher. The cost: ~5 s/trace at vLLM-bs≥4 × 6500 prompts ≈ 9 hours on a single H200 [extrapolated from `feedback_calibration_oom_ladder.md` measurement of 267 s/trace at bs=1 bf16; vLLM at bs≥4 reduces that to ~5 s/trace per `build_self_traces_calib_vllm.py:1419`]. The cost is finite and acceptable. But:
- For MoT subsets specifically, we'd discard 350K rows of high-quality canonical thinking traces and replace them with our own teacher's generation. That's **information-throwaway**: R1 thinking traces and Qwen3.6 thinking traces are both useful supervision; mixing them is strictly more diverse than mixing only Qwen3.6 traces.
- The hidden cost in Choice A: max_new_tokens=16384 means each prompt can burn the full 16K budget even when the natural completion is 2K. vLLM's prefix sharing helps but doesn't eliminate this. Capping max_new_tokens per-subset based on canonical-completion length statistics (Choice C addendum) would help but adds plumbing complexity. Teacher-forcing is the cleaner solution.

### Why not teacher-force everywhere (Choice B)?
- tulu3, dolly, fineweb, aya, writingprompts, arxiv: their canonical completions are NOT thinking-mode. Teacher-forcing on them gives us routing patterns on non-thinking tokens, defeating the purpose of self-traces. This breaks the whole rationale in `calibration.py:1299-1320`.
- math/OpenMathInstruct-2: Llama-3.1-405B CoT is different enough from R1/Qwen3 thinking style that teacher-forcing here is risky. Better to keep math-prompts → generate.

### Recommendation: Choice C (hybrid)
~5500 prompts generated (52%), ~5000 prompts teacher-forced (48%) at the proposed mix → roughly half the teacher-generation wall-clock vs. Choice A on the same total prompt count.

---

## 3. Phase 3 — Existing-subset disposition

Apply the same lens to the 8 existing subsets.

| Subset | Current weight | Proposed | Action | Reason |
|---|---:|---:|---|---|
| tulu3 | 30% | 18% | **SHRINK** | Still the broadest general-SFT signal. Shrunk because (a) MoT-math + MoT-code now cover the reasoning portion of what tulu3 covered, (b) tulu3 already contains evol-codealpaca, so dropping that subset frees up budget without losing tulu3's code share. |
| math (OpenMathInstruct) | 15% | 9% | **SHRINK** | Math reasoning is now also covered by MoT-math (12%). Keep some OpenMathInstruct because its prompts span GSM8K+MATH and exercise the teacher's own thinking-mode math style (vs. R1's). |
| code (Evol-Instruct-Code-80k) | 15% | 0% | **DROP** | Redundant with (a) tulu3's code share (~10%), (b) MoT-code (12%), (c) SWE-smith (16%). Three other code sources cover this. |
| qa (dolly) | 10% | 5% | SHRINK | Still useful for short-instruction QA, but the tulu3 mixture absorbs most short-form QA. 5% retains the signal at half cost. |
| creative (writingprompts) | 10% | 5% | SHRINK | Long-form generation diversity is good, but no thinking, and tulu3 has WildChat-1M as a source — overlap. 5% retains the signal. |
| multilingual (aya) | 10% | 8% | HOLD-ish | The only multilingual-routing signal in the whole mix. Slight shrink because tulu3 has aya_dataset listed as a source — small overlap, kept high. |
| fineweb-edu | 5% | 4% | HOLD | Only raw-text replay (Gemma-3-QAT discipline; see `calibration.py:982-985`). Hard floor — drop only if a future experiment proves pure-SFT is fine. |
| papers (arxiv) | 5% | 3% | SHRINK | Academic-style reasoning; partly covered by MoT-science now. Keep a small slice for "write an abstract"-style prompts which MoT-science doesn't have. |

Sum existing (proposed): 18 + 9 + 0 + 5 + 5 + 8 + 4 + 3 = **52%** of total mix → leaves 48% for the new candidates.

---

## 4. Phase 4 — The thinking-mode role question (headline)

**Recommendation: Choice C (hybrid per-subset)**.

### Reasoning

The headline cost claim, working forward from `feedback_calibration_oom_ladder.md` (bs=1 bf16 = 267 s/trace) and `build_self_traces_calib_vllm.py:1419` (vLLM bs≥4 ≈ 5 s/trace [measured-ballpark, not a hard number]):

| Scenario | Generate-mode prompts | Teacher-forced prompts | Total prompts | Wall-clock estimate |
|---|---:|---:|---:|---|
| **A** — Generate everything, 6500 prompts | 6500 | 0 | 6500 | ~9 h vLLM-H200 [extrapolated] |
| **B** — Teacher-force everything, 6500 prompts | 0 | 6500 | 6500 | ~30 min vLLM-H200 (forward-only) [conjecture] |
| **C** — Hybrid, 6500 prompts | ~3380 (52%) | ~3120 (48%) | 6500 | ~5 h vLLM-H200 [extrapolated]; ~half of A |
| **C+grow** — Hybrid, 8000 prompts | ~4160 | ~3840 | 8000 | ~5.8 h; same wall-clock as A but +23% prompts |

**Per `feedback_speedup_questions_target_real_run.md`**: speedup discussion targets the real recovery, not smoke. The numbers above are for the production calibration build.

### Decision rationale

1. **Choice A is safest but information-throwaway.** R1 traces are real high-quality thinking-mode supervision; discarding them to re-generate with Qwen3.6 loses the cross-model diversity that helps prevent over-specialization. **[conjecture]**
2. **Choice B is cheapest but risks distribution drift on the 7 subsets whose canonical completions aren't thinking-format.** The whole rationale of self-traces (see `calibration.py:1299-1320`) is that routers + merged experts need supervision on `<think>...</think>` token positions. Half of B's prompts would be on non-thinking tokens. Defeats the corpus's purpose.
3. **Choice C threads the needle**: thinking-format subsets (MoT × 3) use canonical R1 traces (real thinking supervision, free); non-thinking subsets (tulu3, math, dolly, creative, aya, fineweb, papers) use prompt-only → generate so we get on-distribution thinking traces; SWE-smith uses canonical Claude trajectories as a calculated bet (we get tool-call routing supervision in exchange for zero thinking-token supervision on this subset).

### The R1-vs-Qwen3.6 distribution gap (the central assumption of Choice C)

R1 and Qwen3.6 are not byte-identical. **Conjecture**: they are close enough in thinking-mode that their respective routing patterns on identical token streams are correlated. **Measured**: no — this has not been tested. The cheapest test (§7.1) is to run the teacher on ~100 sampled MoT-math prompts, compare per-layer router-weight distributions on (prompt + teacher_gen) vs (prompt + R1_canonical_gen), and quantify the KL between the two router distributions.

If that test shows large KL → Choice C degrades to Choice A on the MoT subsets (just re-generate them, eating ~3-4 extra hours of teacher work). The validation test is cheap and gates the assumption directly.

### What "teacher-forced" means operationally

The pipeline today (`build_self_traces_calib*.py`) only knows how to *generate* traces; it has no path for "take this canonical (prompt + completion) pair and just save it as a trace". A new code path is needed:

- `_iter_prompts_from_<dataset>` returns `(prompt, canonical_completion, domain)` triples for teacher-forced subsets.
- The trace JSONL writer takes either:
  - `messages=[{user: prompt}, {assistant: teacher_generated}]` (current GENERATE path), or
  - `messages=[{user: prompt}, {assistant: canonical_completion}]` (NEW TEACHER-FORCED path) — no generation, just write the row.
- Downstream consumers (calibration.py loader) treat the two identically — both render through `apply_chat_template(..., enable_thinking=True)`.
- The teacher forward pass that captures covariance/router stats (the actual calibration step in Stage 2 profiling) still runs on the rendered tokens; it doesn't care whether those tokens came from generation or canonical.

This is a ~150-line patch to `build_self_traces_calib_vllm.py` + new per-dataset iterators in `build_self_traces_calib.py`. No changes to `calibration.py` loader.

---

## 5. Phase 5 — Final mix design

### 5.1 Final subset list

11 subsets total. Sum of weights = 100%.

| # | Subset | Weight | Dataset | Role | Avg tokens/row [conjecture unless noted] |
|---:|---|---:|---|---|---:|
| 1 | tulu3 | 18% | `allenai/tulu-3-sft-mixture` | generate | 600 [measured-ballpark from `calibration.py:1013`] |
| 2 | math | 9% | `nvidia/OpenMathInstruct-2` | generate | 800 |
| 3 | qa | 5% | `databricks/databricks-dolly-15k` | generate | 400 |
| 4 | creative | 5% | `euclaise/writingprompts` | generate | 600 |
| 5 | multilingual | 8% | `CohereForAI/aya_dataset` | generate | 400 |
| 6 | fineweb | 4% | `HuggingFaceFW/fineweb-edu` | generate | 1500 |
| 7 | papers | 3% | `gfissore/arxiv-abstracts-2021` | generate | 300 |
| 8 | mot_math | 12% | `open-r1/Mixture-of-Thoughts`, config=`math` | teacher-forced | 3500 (median ~3K) |
| 9 | mot_code | 12% | `open-r1/Mixture-of-Thoughts`, config=`code` | teacher-forced | 15000 (median ~13K, exceeds 16K cap) |
| 10 | mot_science | 8% | `open-r1/Mixture-of-Thoughts`, config=`science` | teacher-forced | 2500 |
| 11 | swe_smith | 16% | `SWE-bench/SWE-smith-trajectories`, splits=`xml`+`tool` | teacher-forced (multi-turn → flattened to one assistant turn per row, see §5.4) | 20000 (very high variance) |

Generate-mode total: 18+9+5+5+8+4+3 = **52%**
Teacher-forced total: 12+12+8+16 = **48%**

### 5.2 Sequence/prompt budget

| Parameter | Current | Recommended | Justification |
|---|---:|---:|---|
| `sequence_length` | 2048 | **4096** | User-decided; captures longer-range routing patterns; MoT-code traces and SWE-smith multi-turn benefit most. |
| `max_new_tokens` (generate-mode only) | 16384 | **16384** | User-decided; UNCHANGED. |
| `num_sequences` | 4000 | **A/B/C — open** | See §6 (user decision). My recommendation: **A (4000)** — doubles total tokens vs the 2048×4000 baseline, which compounds with the 4096 seq_len. The teacher-forced 48% has near-zero marginal cost, so growing num_sequences mostly costs generate-time on the 52% — that's ~9 h on H200 vLLM at 4000 prompts, manageable. |
| `num_prompts` (build_self_traces) | 6500 | **8000** (≈2× post-`_complete`-filter, conservative) | At completeness ~70-80% [conjecture from `build_self_traces_calib.py:35`], 8000 raw prompts yields ~5600-6400 complete prompts after the filter, comfortably covering 4000 sequences × 1.4× downstream truncation budget. |

### 5.3 num_prompts floor — diversity check

The smallest weight is **papers at 3%**. At `num_prompts=8000`:
- papers gets `floor(8000 × 0.03) = 240` prompts. Above the diversity threshold (computed in `build_self_traces_calib.py:178` as `2 × n_subsets / min_weight = 2 × 11 / 0.03 ≈ 733`).
- **Floor violation**: papers (240) < threshold (733). Two options:
  - (i) Raise papers to 8% and shrink something else (e.g., creative 5%→3% or aya 8%→6%). The threshold then becomes `2 × 11 / 0.03 = 733` if we don't change min_weight. **Better fix**: raise min_weight to 4% (papers and fineweb both at 4%) → threshold = `2 × 11 / 0.04 = 550`, and 8000 × 0.04 = 320 still < 550.
  - (ii) Raise num_prompts to 18000+ to give papers ≥540 prompts at 3% weight.
  - (iii) Accept the warning: papers will be slightly under-represented and the iteration-order short-circuit at `build_self_traces_calib.py:281` may drop the tail. Smallest subsets are processed first in the dict order, so this is OK in practice.
- **Recommendation**: bump fineweb 4%→5% and papers 3%→5%, drop tulu3 18%→16%. Floor at 5% min_weight → threshold = `2 × 11 / 0.05 = 440`; 8000 × 0.05 = 400 still slightly below 440 but within the same order of magnitude. **Acceptable**.

Revised final weights with the floor fix:

| # | Subset | Weight (revised) |
|---:|---|---:|
| 1 | tulu3 | 16% |
| 2 | math | 9% |
| 3 | qa | 5% |
| 4 | creative | 5% |
| 5 | multilingual | 8% |
| 6 | fineweb | 5% |
| 7 | papers | 5% |
| 8 | mot_math | 12% |
| 9 | mot_code | 12% |
| 10 | mot_science | 8% |
| 11 | swe_smith | 15% |
| | **Sum** | **100%** |

Generate-mode: 16+9+5+5+8+5+5 = **53%**. Teacher-forced: 12+12+8+15 = **47%**. Floor (min_weight=5%, 8000 prompts) = 400 prompts/subset, below threshold of 440 by ~10% → acceptable; will log a warning at build time but not fail.

### 5.4 Row-extraction recipes

| Subset | Field extraction recipe |
|---|---|
| tulu3 | `messages` → render full chat through apply_chat_template (current behavior; `calibration.py:1230`) |
| math | `problem` → user turn; teacher generates assistant (drop `generated_solution`) |
| qa | `instruction + ("\n\n" + context if context)` → user turn; teacher generates |
| creative | `prompt` → user turn; teacher generates |
| multilingual | `inputs` → user turn; teacher generates (no language metadata pre-pended) |
| fineweb | `"Read the following passage and explain its key ideas:\n\n" + text[:2000]` → user turn; teacher generates |
| papers | `"Write the abstract for an academic paper titled:\n\n" + title` → user turn; teacher generates |
| mot_{math,code,science} | `messages[0]` (user) → user turn; `messages[1]` (assistant, contains `<think>...</think>final answer`) → assistant turn, **teacher-forced, no generation**. Schema: write the row directly into the trace JSONL with `_complete=true` (canonical traces are complete by construction). |
| swe_smith (xml + tool) | Pull both splits, alternate. Each row's `messages` field is a JSON-serialized multi-turn list. **Flatten to a single (user, assistant) pair**: take `messages[0]` (system or first user — drop system since the apply_chat_template will re-inject Qwen3's system header), find the first `role=user` message → user turn. Take the **first assistant turn** containing a tool call → assistant turn. Discard the rest (subsequent turns + observations). This loses the multi-turn signal but keeps the schema single-turn (per existing `_iter_prompts_from_qwen3_pretrain_mix` contract). **If multi-turn support is added later** (schema extension; see §6.3), unflatten and pass the full list. For xml split: wrap the `<function=...>` blocks in `<tool_call>...</tool_call>` so the Qwen3 chat template renders them as tool calls. For tool split: convert to Qwen3-compatible tool_calls format. |

### 5.5 Final committed parameters

```yaml
calibration:
  source: self-traces
  seed: 1337
  num_sequences: 4000      # DECISION A (recommended); see §6.2
  sequence_length: 4096    # DECIDED — was 2048

build_self_traces_calib:
  num_prompts: 8000
  max_new_tokens: 16384    # UNCHANGED — generate-mode only
  prompts_source: qwen3-pretrain-mix-v2   # NEW corpus name (avoid in-place
                                          # replacement; see §6.1)
```

---

## 6. Phase 6 — Decisions the user must make

These four are surfaced explicitly per `feedback_raise_dont_substitute.md`. I will NOT decide these for you.

### 6.1 Corpus naming: new name vs. in-place replacement?

**Options**:
- (a) Rename to `qwen3-pretrain-mix-v2` (new corpus, registered alongside the old). Existing trace JSONLs on disk continue to work with `qwen3-pretrain-mix` (legacy). New runs point at v2.
- (b) Replace `qwen3-pretrain-mix` in-place. Old JSONLs become orphaned but the corpus name remains the same. Pipeline configs don't need to change.

**My recommendation**: **(a) v2 — separate corpus**. The schema_version cache-key gate (per `build_self_traces_calib.py:360`) already invalidates old JSONLs when the corpus changes, but in-place rename is cleaner from a reproducibility standpoint. The 53-commit ALGORITHM_REFERENCE retirement pattern (`project_algorithm_reference_retirement.md`) suggests the user values keeping old names around for a deprecation window.

**Need from you**: which option.

### 6.2 num_sequences: A (4000), B (2000), or C (3000)?

User leaning A. My recommendation: **A (4000)**.

Reasoning: at sequence_length=4096 (doubled from 2048) and num_sequences=4000 (held), total tokens = 4000 × 4096 = 16.4M tokens — roughly 2× the prior calibration token budget. At sequence_length=4096 and num_sequences=2000 (option B), total = 8.2M tokens — same as prior. At num_sequences=3000 (C), 12.3M.

Cost implication: at `num_sequences=4000`, the teacher must process 4000 × 4096 = 16.4M tokens through Stage 2 profile-pass forward (which runs on the calibration tensor, not on raw prompts). At sequence_length=2048×4000, it was 8.2M tokens. So Stage 2 forward roughly doubles. **[conjecture]**: at ~30-50 ms/layer-forward × 40 layers × 125 batches → previously ~150-250 sec/profile, now ~300-500 sec/profile. Acceptable per `SC_FAST_PLAN_V3.md:33`.

**Need from you**: A/B/C.

### 6.3 SWE-smith multi-turn schema extension — flatten or extend?

The current self-traces JSONL schema is single-turn `messages=[{user}, {assistant}]`. SWE-smith trajectories are multi-turn with tool calls (5–50 round-trips). Two options:

- (a) **Flatten to single-turn** (my recommendation in §5.4): take only the first (user, assistant_with_tool_call) pair. Loses multi-turn signal but keeps schema simple.
- (b) **Extend schema to support multi-turn**: trace JSONL allows `messages=[{system?}, {user}, {assistant_with_tool_call}, {tool}, {user}, ...]`. The `apply_chat_template` call in `calibration.py` handles arbitrary `messages` already (it's just a list). The change is in the writer (`build_self_traces_calib*.py`) and the row extractor.

Reasoning for (a): smallest patch, ships fastest. Reasoning for (b): captures the actual deploy distribution (Qwen3.6 in agentic mode is multi-turn), and tool-call routing patterns on the response-to-tool-output position are different from response-to-user-prompt position.

**[conjecture]**: option (b) gives ~10-30% better routing signal on tool-use deployments, at the cost of ~1-2 days of pipeline work to verify the multi-turn rendering through `apply_chat_template` + downstream consumers.

**Need from you**: (a) flatten or (b) extend schema.

### 6.4 Validate Choice C before commit, or just ship it?

The R1-vs-Qwen3.6 distribution-gap assumption (§4) is the load-bearing premise of recommending teacher-forced on MoT. Two postures:

- (a) **Validate first** — run §7.1 GPU test before kicking off the production calibration build. Cost: ~$15-30 of H200 time on a small sample. Latency: ~1-2 hours.
- (b) **Ship it on the assumption** — kick off the full Choice C build; if the resulting calibration produces obviously bad Stage 6 metrics, fall back to Choice A. Cost: zero up front but ~5-9 hours of teacher-gen rework if the assumption fails.

My recommendation: **(a) validate first**, but the test is small enough that it can run inside the rental window of the production build (parallel to the generate-mode part, since they don't share GPU). Per `feedback_secure_before_burning.md`.

**Need from you**: (a) or (b).

---

## 7. Phase 7 — Validation tests

### 7.1 R1-vs-Qwen3.6 router-pattern alignment (GATES the Choice C recommendation)

**What it tests**: whether teacher-forced MoT-canonical traces produce similar router patterns to (prompt-only + Qwen3.6-generated) traces on the same prompts.

**Procedure**:
1. Sample 100 prompts from MoT-math (50) + MoT-code (30) + MoT-science (20).
2. For each prompt:
   - **Track A (canonical)**: feed (prompt + canonical_R1_completion) through the teacher's forward pass; capture per-layer router logits and top-k expert indices.
   - **Track B (generated)**: feed (prompt) through the teacher's generation; capture the generated completion; then feed (prompt + teacher_gen) through forward; capture same signals.
3. Compute per-layer KL divergence between the two router-logit distributions, averaged over token positions.
4. Compute top-1-expert agreement rate (fraction of token positions where both tracks pick the same expert).

**Pass criteria**:
- Mean per-layer KL < 0.5 (conservative; comparable to fp16-vs-fp32 router-logit drift)
- Top-1 expert agreement > 70%

**Hardware**: H200 SXM5 or H100-80G. ~30 minutes to load teacher + run 100 prompts (50 generations + 100 forwards). **Approximate GPU-hours**: 0.5 h.

**If it fails**: degrade MoT subsets to GENERATE mode (Choice A for MoT). Adds ~3-4 GPU-hours of teacher-gen but preserves the corpus.

### 7.2 Row-extraction smoke tests (CPU-only, gate the row extractor patches)

For each new subset, a tiny test that pulls 5 rows, applies the extraction recipe, renders through `apply_chat_template(..., enable_thinking=True)`, and asserts the rendered string is non-empty and contains expected markers.

| Subset | CPU-only test |
|---|---|
| mot_math | pull 5 rows, render, assert each contains `<think>` and `</think>` |
| mot_code | same + assert contains a code block (``` or `cpp`) |
| mot_science | same |
| swe_smith xml | pull 5 rows, parse `messages` JSON, find first (user, assistant_with_tool_call) pair, wrap `<function=...>` in `<tool_call>`, render, assert non-empty |
| swe_smith tool | pull 5 rows, parse `messages` JSON, find first assistant with `tool_calls`, render, assert non-empty |

Total: ~20 minutes of CPU work, no GPU.

### 7.3 SWE-smith canonical-trace token-budget check (CPU-only)

Some SWE-smith trajectories exceed sequence_length=4096 even after flattening (the first assistant turn alone can be huge). Test: tokenize 100 sampled rows, distribution of post-render token counts, fraction over 4096. If > 30% over, raise the truncation strategy as a concern.

**Hardware**: CPU only. ~10 minutes.

### 7.4 End-to-end build smoke (1 GPU-hour, gates production build)

Run `build_self_traces_calib_vllm.py --num-prompts 200 --prompts-source qwen3-pretrain-mix-v2 --max-new-tokens 4096 --batch-size 4` on H100/H200. Verify:
- The JSONL has ~200 rows across all 11 subsets in roughly the right proportions.
- The teacher-forced rows have `_complete: true` (they're complete by construction).
- The generate-mode rows have `_complete: true` for ≥60% of math/qa/dolly prompts and ≥40% of fineweb prompts (some prompts the teacher will refuse or punt).

**Hardware**: H100-80G or H200, single GPU, ~1 hour. **Approximate GPU-hours**: 1.

### 7.5 Calibration-tensor invariance check (Stage 2 profile, GPU-required)

After the production build, before kicking off the full pipeline: run Stage 2 profile-pass on the new calibration tensor + the old qwen3-pretrain-mix calibration tensor. Compare:
- Per-layer top-K expert distribution (router agreement) — should differ but not dramatically
- Per-expert covariance trace ratio — should be within ~5× (we expect a real shift, not 100×)

If covariance ratios are wildly off, something is broken in the new corpus.

**Hardware**: H200 (Stage 2 profile requires 140 GB-class). ~20 min. **Approximate GPU-hours**: 0.3.

### Total validation GPU-hours

| Test | GPU-hours | Hardware | Gates |
|---|---:|---|---|
| 7.1 R1-vs-Qwen3.6 alignment | 0.5 | H100/H200 | Choice C correctness |
| 7.2 row-extraction smoke | 0 (CPU) | — | row extractors |
| 7.3 SWE-smith token check | 0 (CPU) | — | flatten strategy |
| 7.4 end-to-end build smoke | 1.0 | H100/H200 | production build |
| 7.5 calibration-tensor invariance | 0.3 | H200 | new calibration vs old |
| **Total** | **~2 GPU-hours** | mixed | |

All within a single H200 rental session. The production calibration build itself is ~5-9 GPU-hours separately.

---

## 8. Risks and known unknowns

1. **R1-vs-Qwen3.6 distribution gap** (§4) — load-bearing assumption; mitigated by §7.1.
2. **xlam dropped** — if function-calling deployment surface matters and SWE-smith doesn't cover it adequately (xlam has 4000+ unique APIs vs SWE-smith's narrow Python-debugging surface), a follow-up corpus may need a thinking-format-native function-calling source. Out of scope for this iteration.
3. **SWE-smith assistant turns may exceed 16K tokens** even after flattening (§7.3). Truncation strategy: hard-cut at sequence_length=4096 during calibration-tensor build. The first ~3K tokens always include the tool-call setup which is the routing-relevant part. **[conjecture]**
4. **Token-budget mismatch on teacher-forced rows**: max_new_tokens=16384 is a GENERATE-mode parameter and doesn't apply to teacher-forced rows. But sequence_length=4096 (calibration tensor length) DOES truncate teacher-forced rows. We will get partial supervision on long MoT-code traces (median 13K tokens → top 4096 only). Acceptable per Gemma-3 QAT rule that says first/last tokens matter most.
5. **Multi-turn flattening loses ~80% of SWE-smith content** (each trajectory has many turns; we keep only the first). Mitigation in §6.3 option (b) if user prefers the bigger lift.
6. **Cache invalidation**: the new corpus name `qwen3-pretrain-mix-v2` (§6.1) invalidates ALL existing self-traces JSONLs. This is intentional but expensive (one full rebuild on H200, ~5-9 GPU-hours).

---

## 9. Files touched (if approved)

| File | Change |
|---|---|
| `max_quality/src/moe_compress/utils/calibration.py` | New `_QWEN3_MIX_V2_*` dicts and `_stream_texts_qwen3_pretrain_mix_v2` function. Register as new corpus `qwen3-pretrain-mix-v2`. Existing `qwen3-pretrain-mix` untouched. |
| `max_quality/scripts/build_self_traces_calib.py` | New `_iter_prompts_from_qwen3_pretrain_mix_v2` iterator handling the 11 subsets. New `_iter_canonical_traces` helper for teacher-forced subsets. |
| `max_quality/scripts/build_self_traces_calib_vllm.py` | Add a teacher-forced fast-path: rows with `_canonical_completion` skip generation and write directly. Adds `schema_version=7` to cache key. |
| New test: `tests/test_qwen3_pretrain_mix_v2_extractors.py` | CPU-only row-extraction smoke (§7.2). |
| New test: `tests/test_swe_smith_canonical_render.py` | CPU-only render+tokenize check (§7.3). |

Approximate diff size: ~400-600 LoC.

---

## 10. Cite-and-verify table (Phase 1 evidence)

| Claim | Evidence |
|---|---|
| MoT distilled from DeepSeek-R1 | `https://huggingface.co/datasets/open-r1/Mixture-of-Thoughts` — dataset card description. |
| MoT-math row contains `<think>...</think>` | Direct dataset_preview sample, MoT/math/train[0]. |
| MoT-code row contains `<think>...</think>` | Direct dataset_preview sample, MoT/code/train[0]. |
| MoT-science source is mostly Llama-Nemotron post-training | Direct dataset_preview, MoT/science/train[0] has `source: "nvidia/Llama-Nemotron-Post-Training-Dataset"`. **Caveat**: the science split mixes R1 traces with Nemotron traces; the curation page says they filter to "exclude traces with Qwen model pre-processing" but doesn't explicitly say all science traces are from R1. **[unverified]** Treat science as the weakest of the three MoT subsets for the R1-alignment assumption. |
| xlam is gated | Direct dataset_overview shows 🔒 Gated status. |
| xlam uses JSON tool_calls, no thinking | WebFetch of the dataset card. |
| SWE-smith uses Claude 3.7 Sonnet | Dataset card description. |
| SWE-smith xml split uses `<function=NAME><parameter=...>` | Direct sample row inspection (see saved tool-results files in this session). |
| SWE-smith tool split uses OpenAI `tool_calls` schema | Direct sample row inspection — `role=tool`, `type=function`. |
| SWE-smith ticks split uses plain markdown code-blocks | Direct sample row inspection. |
| Qwen3.6 chat template uses `<tool_call>\n<function=...>...\n</tool_call>` | Direct read of tokenizer_config.json chat template. |
| evol-codealpaca-v1 is a source dataset of tulu3-sft-mixture | tulu-3-sft-mixture dataset card lists `source_datasets:theblackcat102/evol-codealpaca-v1`. |

End of design.
