# Calibration Mix V2 — Implementation Plan

**Repo**: `/home/lucas/ai/moe_compress` · branch `feat/calibration-v2` · base `6ff3636+`
**Author**: planner agent, 2026-05-27
**Status**: file-by-file, function-by-function spec. NO code in this file — implementer agent will execute next.

**Design reference**: `tasks/CALIBRATION_MIX_V2_DESIGN.md` (verbatim companion; this plan
encodes the user-approved Option C with the §"Final approved mix" overrides from the
campaign brief: `num_prompts=8000`, `num_sequences=4000`, `sequence_length=4096`,
`max_new_tokens=16384`, 12 subsets, sum=100%).

The user's approved mix differs from the design doc's revised mix in §5.3 in three ways:
1. tulu3 is 11% (not 16%).
2. swe_smith is 12% (not 15%); `xml` split only, not xml+tool.
3. Adds a 12th subset: `function_calling` at 8% (`glaiveai/glaive-function-calling-v2`).

Final 12-subset mix (sum = 100%):

| # | Subset key | Weight | HF dataset | Policy | Multi-turn? |
|---:|---|---:|---|---|---|
| 1 | tulu3 | 11% | `allenai/tulu-3-sft-mixture` | GENERATE | no |
| 2 | math | 9% | `nvidia/OpenMathInstruct-2` | GENERATE | no |
| 3 | qa | 5% | `databricks/databricks-dolly-15k` | GENERATE | no |
| 4 | creative | 5% | `euclaise/writingprompts` | GENERATE | no |
| 5 | multilingual | 8% | `CohereForAI/aya_dataset` | GENERATE | no |
| 6 | fineweb | 5% | `HuggingFaceFW/fineweb-edu` | GENERATE | no |
| 7 | papers | 5% | `gfissore/arxiv-abstracts-2021` | GENERATE | no |
| 8 | mot_math | 12% | `open-r1/Mixture-of-Thoughts` cfg=`math` | TEACHER_FORCED | no |
| 9 | mot_code | 12% | `open-r1/Mixture-of-Thoughts` cfg=`code` | TEACHER_FORCED | no |
| 10 | mot_science | 8% | `open-r1/Mixture-of-Thoughts` cfg=`science` | TEACHER_FORCED | no |
| 11 | swe_smith | 12% | `SWE-bench/SWE-smith-trajectories` split=`xml` | TEACHER_FORCED, flatten | YES |
| 12 | function_calling | 8% | `glaiveai/glaive-function-calling-v2` | GENERATE | no |

Generate-mode (52% — 5 v1 subsets + math + function_calling): tulu3, math, qa, creative, multilingual, fineweb, papers, function_calling.
Teacher-forced (48%): mot_math, mot_code, mot_science, swe_smith.

---

## Section A — Inventory of files to touch

### A.1 `max_quality/src/moe_compress/utils/calibration.py`
**Current state** (verified at HEAD `6ff3636`):
- Lines 980-1032: `_QWEN3_MIX_WEIGHTS`, `_QWEN3_MIX_AVG_TOKENS`, `_QWEN3_MIX_DATASET` dicts for the v1 mix (8 subsets).
- Lines 1035-1049: `_parse_yaml_qwen3_pretrain_mix(cal_cfg, num_sequences, sequence_length, seed) -> CalibrationSpec` (returns spec with `source="qwen3-pretrain-mix"`).
- Lines 1052-1056: `_make_subset_seed(base_seed, subset) -> int` (md5-based per-subset offset).
- Lines 1059-1073: `_shuffled_stream(dataset_name, count, seed)` — circuit-breaker streaming helper used by all v1 subset streamers. Reusable as-is.
- Lines 1076-1197: `_stream_messages_native`, `_stream_raw_wrapped`, `_stream_problem_solution`, `_stream_instruction_output` (per-shape generic streamers; reused by v2).
- Lines 1200-1281: `_stream_texts_qwen3_pretrain_mix(spec, tokenizer) -> list[str]` (the dispatch faucet that iterates `_QWEN3_MIX_WEIGHTS` and routes each subset to the right helper).
- Lines 1284-1288: `register_corpus(CorpusAdapter(name="qwen3-pretrain-mix", parse_yaml=..., stream_texts=...))`.

**Additions** (purely additive, ~180 LoC; v1 untouched):
1. Below the v1 `register_corpus` call at line 1288, add a new "Qwen3 reasoning mix v2" section.
2. Add module-level constants (~50 LoC):
   - `_QWEN3_MIX_V2_WEIGHTS: dict[str, float]` — 12 keys summing to exactly 1.0. Use the exact percentages above as float literals (`{"tulu3": 0.11, "math": 0.09, ...}`).
   - `_QWEN3_MIX_V2_AVG_TOKENS: dict[str, int]` — per-subset avg tokens (see Section B per-subset table for values).
   - `_QWEN3_MIX_V2_DATASET: dict[str, str]` — HF dataset name per subset.
   - `_QWEN3_MIX_V2_DATASET_CONFIG: dict[str, str | None]` — HF config for MoT subsets (`"math"`, `"code"`, `"science"`); `None` for all others.
   - `_QWEN3_MIX_V2_DATASET_SPLIT: dict[str, str]` — defaults to `"train"`; `"xml"` for `swe_smith`. (Plain string per subset, no None.)
   - `_QWEN3_MIX_V2_POLICY: dict[str, str]` — `"GENERATE"` or `"TEACHER_FORCED"` per subset. Used by `build_self_traces_calib*` to pick the codepath; `_stream_texts_qwen3_pretrain_mix_v2` itself does NOT branch on this (calibration consumers always render+forward whatever message list the JSONL gave them).
3. New function `_parse_yaml_qwen3_pretrain_mix_v2(cal_cfg, num_sequences, sequence_length, seed) -> CalibrationSpec` (~15 LoC, exact-twin of `_parse_yaml_qwen3_pretrain_mix` but `source="qwen3-pretrain-mix-v2"`). Same "yaml `dataset`/`subset_weights` ignored" contract.
4. New function `_stream_texts_qwen3_pretrain_mix_v2(spec, tokenizer) -> list[str]` (~120 LoC; new helpers below dispatched per-subset). Structure mirrors `_stream_texts_qwen3_pretrain_mix`:
   - One-shot intro banner (gate on a NEW module-level `_broad_instruct_mix_v2_intro_logged` flag; do NOT reuse the v1 flag).
   - For each subset key in `_QWEN3_MIX_V2_WEIGHTS`:
     - Compute `target_subset_tokens`, `n_rows = max(1, int((target_subset_tokens / avg) * 2.0))`, `seed = _make_subset_seed(spec.seed, subset)`.
     - Resolve `ds_name`, `config` (None or "math"/"code"/"science"), `split` ("train" or "xml").
     - Dispatch to the right helper (see Section B).
     - Wrap each helper call in `try/except Exception` exactly like v1 lines 1271-1277.
5. Two NEW helpers added near the other `_stream_*` helpers (insert between `_stream_instruction_output` at line 1197 and `_stream_texts_qwen3_pretrain_mix` at line 1200):
   - `_stream_messages_with_config(dataset_name, config, split, count, tokenizer, seed)` — like `_stream_messages_native` but accepts a `config` arg passed through to `load_dataset(name, config, split=..., streaming=True)`. Reused by mot_math / mot_code / mot_science. ~30 LoC.
   - `_stream_swe_smith_xml(dataset_name, split, count, tokenizer, seed)` — JSON-decodes `row["messages"]` (it's a str, not a list of dicts), flattens to first (user, assistant) pair using the rules in Section B.4, renders through `apply_chat_template` (which already handles the `<function=...>` syntax verbatim because Qwen3's template renders any string assistant content as-is). ~50 LoC.
   - `_stream_glaive_function_calling(dataset_name, count, tokenizer, seed)` — parses Glaive's flat `system`/`chat` row, extracts the first USER turn from the `chat` string (Section B.5). ~40 LoC.
6. At the bottom: `register_corpus(CorpusAdapter(name="qwen3-pretrain-mix-v2", parse_yaml=..., stream_texts=...))`.

**Type**: ADDITIVE. v1 corpus untouched; v1 streaming helpers reused via import. Both adapters coexist in the registry.

---

### A.2 `max_quality/scripts/build_self_traces_calib.py`
**Current state**:
- Lines 121-282: `_iter_prompts_from_qwen3_pretrain_mix(num_prompts, seed, prev_num_prompts=None)` iterator. Per-subset extractors inline (`if subset == "tulu3"`, `elif subset == "fineweb"`, etc.).
- Lines 285-298: `_iter_prompts_from_jsonl` — generic JSONL prompt loader (untouched).
- Lines 308-362: `_trace_cache_key` (HF path) — folds `prompts_source` (corpus-name#num#seed format), `schema_version=6`.
- Lines 449-721: `_generate_traces(model, tokenizer, prompts, ...)` — the generation loop. Currently every row goes through `model.generate(...)`. **MUST be extended** to accept TEACHER_FORCED rows that bypass generation.
- Lines 870-881: CLI dispatch on `args.prompts == "qwen3-pretrain-mix"` → call `_iter_prompts_from_qwen3_pretrain_mix`.

**Additions** (~200 LoC):
1. NEW function `_iter_prompts_from_qwen3_pretrain_mix_v2(num_prompts, seed, prev_num_prompts=None) -> Iterator[tuple[str, str, str | None, str]]`. Returns 4-tuples `(prompt, domain, canonical_completion_or_None, policy)`:
   - `prompt` (str) — the user turn
   - `domain` (str) — subset key
   - `canonical_completion` (str | None) — None for GENERATE rows; the canonical assistant content for TEACHER_FORCED rows
   - `policy` (str) — `"GENERATE"` or `"TEACHER_FORCED"`
   - Structure mirrors v1: imports `_QWEN3_MIX_V2_DATASET`, `_QWEN3_MIX_V2_DATASET_CONFIG`, `_QWEN3_MIX_V2_DATASET_SPLIT`, `_QWEN3_MIX_V2_WEIGHTS`, `_QWEN3_MIX_V2_POLICY`, `_shuffled_stream`, `_make_subset_seed` from `moe_compress.utils.calibration`.
   - Per-subset payload extractor: the v1 7 subsets (tulu3 / fineweb / math / qa / creative / multilingual / papers) keep the same extractors with `canonical_completion=None`, `policy="GENERATE"`. Replace the `code` branch (which v2 drops) with five new branches: `mot_math`, `mot_code`, `mot_science`, `swe_smith`, `function_calling` (recipes in Section B).
   - Same diversity-floor warning at the head of the function (computed from `_QWEN3_MIX_V2_WEIGHTS`).
   - Same `prev_num_prompts` extension semantics.
   - For non-default splits / configs (`mot_*`, `swe_smith`): pass them through to `_shuffled_stream`. Since `_shuffled_stream` currently hardcodes `split="train"`, ALSO refactor `_shuffled_stream` to accept optional `config: str | None = None`, `split: str = "train"` kwargs and pass them to `load_dataset(...)`. This is a 2-line edit to `calibration.py` lines 1062-1064 and is backward-compatible (all v1 callsites pass nothing).
2. **Backward-compatibility shim**: keep `_iter_prompts_from_qwen3_pretrain_mix` returning 2-tuples `(prompt, domain)` AS-IS (do NOT widen its return type). The v2 iterator is a NEW callable; vLLM script imports v1 by name and now also imports v2 by name.
3. **TEACHER_FORCED path in `_generate_traces`** (Section C). Approach:
   - Widen the `prompts` parameter type annotation. The function currently takes `list[tuple[str, str]]`. Add a sibling typed alias to accept the 4-tuple too. Use duck-typing: detect by `len(prompt_tuple) == 4` at unpack time. Document the policy contract in the docstring.
   - In the per-batch loop, partition the batch into GENERATE rows and TEACHER_FORCED rows. Run GENERATE rows through the existing `model.generate(...)` path. For TEACHER_FORCED rows, skip generation and emit a row directly with `messages=[{user: prompt}, {assistant: canonical_completion}]`, `_complete=True`, `domain=...`, `_attempt_idx=...`, plus a NEW `completion_source` field set to `"canonical"`.
   - GENERATE rows continue to set `completion_source="teacher_generated"`.
   - Backward-compat: when `_iter_prompts_from_qwen3_pretrain_mix` (v1) feeds 2-tuples, the unpacker treats them as `(prompt, domain, None, "GENERATE")`. Existing v1 callsites keep working with no schema changes (the `completion_source` field is new — see Section C step 3 below for forward-compat under the JSONL loader).
4. CLI: extend `args.prompts == "qwen3-pretrain-mix-v2"` branch to call the new iterator (same shape as v1 branch). Reuse the same tokenizer bootstrap, `--prev-num-prompts` plumbing.
5. The CLI's `_trace_cache_key` call (line 831-838) uses `f"{args.prompts}#{args.num_prompts}#{args.seed}"`. Confirm by inspection that flipping `args.prompts` from `"qwen3-pretrain-mix"` to `"qwen3-pretrain-mix-v2"` produces a different cache key (Section E.1).
6. Update the `--prompts` help text to list both `qwen3-pretrain-mix` and `qwen3-pretrain-mix-v2` as recognized values.
7. Update the file-top docstring's "Output schema" section to mention `completion_source`.

**Type**: MIXED. New v2 iterator is additive. `_generate_traces` is modified (extends behavior; old call signature still works via tuple-length detection). `_shuffled_stream` is widened with default-kwargs (backward-compatible).

---

### A.3 `max_quality/scripts/build_self_traces_calib_vllm.py`
**Current state**:
- Lines 89-94: imports `_iter_prompts_from_qwen3_pretrain_mix`, `_iter_prompts_from_jsonl`, `_coerce_eos_ids`, `_trim_at_first_eos` from the HF script.
- Lines 105-150: `_trace_cache_key_vllm` — folds `inference_engine="vllm"`, `schema_version=8`.
- Lines 317-437: `_process_outputs(outputs, prompts_chunk, attempt_idx_chunk, ...)` — vLLM output decoder. Currently always produces GENERATE-style rows.
- Lines 929-933: dispatch on `args.prompts == "qwen3-pretrain-mix"`.

**Additions** (~150 LoC):
1. Extend the import line at 89-94 to also import `_iter_prompts_from_qwen3_pretrain_mix_v2`.
2. Bump `schema_version` in `_trace_cache_key_vllm` from `8` to `9` and document the bump alongside the v8 metadata explanation (the existing comment at lines 127-134). v9 is the version that carries `completion_source` per row.
3. Extend `_process_outputs` to accept 4-tuples in `prompts_chunk`. Same detection-by-length logic as the HF script (Section A.2 step 3). For TEACHER_FORCED rows in `prompts_chunk`: skip the vLLM-output processing for that prompt entirely (we never submitted it to vLLM; see step 4 below) and directly synthesize a JSONL row with `completion_source="canonical"`, `_complete=True`, `n_prompt_tokens=len(tokenize(canonical_prompt_rendered))`, `n_gen_tokens=0`, `has_think=...` (compute via `_has_think_block` on the canonical completion), `refusal_flag=False`.
4. Refactor `main()` chunk loop (lines 1389-1422):
   - Before `llm.generate(rendered, sp)`, partition the chunk: `gen_prompts = [p for p in chunk if p[3] == "GENERATE"]`, `tf_prompts = [p for p in chunk if p[3] == "TEACHER_FORCED"]`.
   - Render + submit only `gen_prompts` to vLLM.
   - Yield TF rows directly via a new helper `_synth_teacher_forced_rows(tf_prompts, tf_attempt_idx, tokenizer, logits_dir=...)` that mimics `_process_outputs` row shape but skips logit-sidecar emission (no per-step logprobs to capture from a canonical trace; `gen.logprobs` is irrelevant).
   - Preserve interleaving: emit GENERATE rows then TEACHER_FORCED rows in this chunk (so JSONL row order is deterministic).
5. Dispatch in `main()`: add an `elif args.prompts == "qwen3-pretrain-mix-v2":` branch that calls `_iter_prompts_from_qwen3_pretrain_mix_v2(...)`. Reuse the same `_prev_suffix`, `--prev-num-prompts` semantics.
6. Update CLI `--prompts` help to mention `qwen3-pretrain-mix-v2`.
7. Update file-top docstring "Output schema" line ~31 to mention `completion_source` and reference the new schema_version=9.

**Type**: MIXED. v2 iterator dispatch is additive. `_process_outputs` + chunk loop are modified (extended behavior; v1 chunks still work because v1 tuples remain 2-tuples).

**Note**: The vLLM imatrix/REAP/cov/per-expert-max accumulator setups must still run for TF rows IF we feed those rows through a vLLM forward pass for calibration capture. But this script only generates the JSONL — the actual calibration forward (which captures cov, REAP, etc.) runs LATER from the JSONL via the calibration pipeline. So: TF rows skip vLLM generation entirely. The accumulator setups remain gated by the existing CLI flags; they capture stats only from the GENERATE rows during the build-self-traces phase. Down-pipeline calibration captures from ALL rows (GENERATE + TF) via the standard self-traces loader. Verify this with the implementer agent before commit.

---

### A.4 `max_quality/configs/qwen36_35b_a3b_30pct.yaml`
**Current state** (lines 29-53):
- `calibration.source: qwen3-pretrain-mix`
- `calibration.seed: 1337`
- `calibration.num_sequences: 4000`
- `calibration.sequence_length: 2048`
- A multi-line comment block (lines 30-49) describing the v1 mix (Tulu3 25 / fineweb 45 / math 15 / code 15 / peS2o dropped).

**Modifications**:
1. Replace the comment block (lines 30-49) with a new v2 comment listing the 12 subsets, weights, and policies (GENERATE / TEACHER_FORCED). Keep the same "Mix is hard-coded in calibration.py; dataset/subset_weights here are ignored." line.
2. `calibration.source: qwen3-pretrain-mix-v2`.
3. `calibration.sequence_length: 4096` (was 2048).
4. `calibration.num_sequences: 4000` (UNCHANGED).
5. `calibration.seed: 1337` (UNCHANGED).

**Type**: MODIFYING (production YAML).

---

### A.5 `max_quality/configs/qwen36_35b_a3b_reap_exact.yaml`
Identical changes to A.4, against lines 43-67. Same updates: replace v1 comment block, `source: qwen3-pretrain-mix-v2`, `sequence_length: 4096`.

**Type**: MODIFYING.

---

### A.6 `max_quality/tests/test_calib_jsonl_schema_v8.py`
**Current state** (350 LoC):
- Lines 282-350: `test_cache_key_carries_schema_version_8` — asserts `schema_version=8` and `prompts_source="qwen3-pretrain-mix"`.
- Lines 96-127: `test_jsonl_row_contains_v8_metadata` — asserts the 6 v8 metadata fields are present.

**Modifications** (~50 LoC):
1. Bump schema_version assertion from 8 → 9 (matches A.3 step 2).
2. Add an assertion that the v9 row carries `completion_source` field (string, value in `{"teacher_generated", "canonical"}`).
3. Add a new test `test_teacher_forced_row_completion_source_is_canonical` that drives a TF prompt through `_synth_teacher_forced_rows` (or whatever the implementer names it) and verifies `completion_source == "canonical"`, `_complete is True`, `n_gen_tokens == 0`, the assistant message content equals the canonical completion string passed in.
4. Add a parallel test `test_generate_row_completion_source_is_teacher_generated` that drives a GENERATE prompt (4-tuple with canonical=None, policy="GENERATE") through `_process_outputs` and asserts `completion_source == "teacher_generated"`.
5. Optionally bump the docstring at lines 1-16 to mention schema v9 and the `completion_source` field.

**Type**: MODIFYING (existing tests updated; new tests added).

---

### A.7 `max_quality/tests/test_run_pipeline_reap_exact.py`
**Current state** (line 60):
- `"source": "qwen3-pretrain-mix"` in the synthesized YAML fixture.

**Modifications** (~5 LoC):
1. Change line 60 to `"source": "qwen3-pretrain-mix-v2"`.
2. The test depends on the corpus name resolving in the registry. Since v2 is added to the same registry, this should pass once A.1 is done. No other changes needed (the test mocks Stage 1+ so it never actually runs `_stream_texts_qwen3_pretrain_mix_v2`).
3. Also update the sibling `test_run_pipeline_normal_mode_regression.py:51` from `"qwen3-pretrain-mix"` → `"qwen3-pretrain-mix-v2"` for parity.

**Type**: MODIFYING.

---

### A.8 `max_quality/tests/test_reap_exact_config.py`
**Current state** (line 38):
- `assert cfg["calibration"]["source"] == "qwen3-pretrain-mix"`

**Modifications**:
1. Change line 38 to `assert cfg["calibration"]["source"] == "qwen3-pretrain-mix-v2"`.
2. Add a new line asserting `cfg["calibration"]["sequence_length"] == 4096`.

**Type**: MODIFYING.

---

### A.9 NEW — `max_quality/tests/test_qwen3_pretrain_mix_v2.py`
**Purpose**: unit-level coverage of v2 corpus internals. CPU-only, no network access (uses fixture rows hand-built in the test file). ~200 LoC.

Tests enumerated in Section F.

**Type**: NEW FILE.

---

### A.10 NEW — `max_quality/docs/calibration_mix_v2.md`
User-facing doc describing the v2 mix. ~150 LoC markdown.

Sections enumerated in Section G.

**Type**: NEW FILE.

---

### A.11 Optional — `max_quality/patches/MANIFEST.md` "Schema bumps" section
**Current state** (lines 608-625): a table of sidecar schema versions.

**Modifications** (~3 LoC):
1. Add a note above or below the table mentioning the JSONL schema_version bump 8 → 9 in `build_self_traces_calib_vllm.py` and `build_self_traces_calib.py` (the HF script's `schema_version=6` does NOT need to bump because it never carried the v8 metadata fields — those are vLLM-only — but the implementer should fold `completion_source` into the HF script's cache_key payload too, see A.2 step 5. If the HF script gets the v9 schema fold, also bump HF's `schema_version` from 6 → 7 and note that here.).

**Type**: MODIFYING (documentation only).

---

## Section B — Per-subset row-extraction recipes

For each of the 12 subsets, this section pins:
1. The HF dataset row schema (verified against Hub via `mcp__claude_ai_Hugging_Face__hub_repo_details` on 2026-05-27).
2. The fields to extract.
3. The prompt-formatting strategy.
4. For TEACHER_FORCED: the canonical completion extraction.

### B.1 Existing 8 v1 subsets — KEEP RECIPE AS-IS

The following 7 are reused verbatim from `_iter_prompts_from_qwen3_pretrain_mix` (file `build_self_traces_calib.py` lines 217-268). Carry the per-row extraction logic unchanged to v2.

**Schema reference** (verified via Hub overview):
- `tulu3`: `messages=[{role, content}, ...]` — extract first user message's content. `domain="tulu3"`.
- `math`: `{problem, generated_solution, ...}` — extract `problem` only. Discard `generated_solution`. `domain="math"`.
- `qa`: `{instruction, context, response, category}` — extract `instruction + ("\n\n" + context if context)`. Discard `response`. `domain="qa"`.
- `creative`: `{prompt, story}` — extract `prompt` only. Discard `story`. `domain="creative"`.
- `multilingual`: `{inputs, targets, ...}` — extract `inputs` only. Discard `targets`. `domain="multilingual"`.
- `fineweb`: `{text, ...}` (raw web text) — wrap as `f"Read the following passage and explain its key ideas:\n\n{text[:2000]}"`. `domain="fineweb"`.
- `papers`: `{title, abstract, ...}` — wrap as `f"Write the abstract for an academic paper titled:\n\n{title}"`. `domain="papers"`.

**Per-row config**: `canonical_completion=None`, `policy="GENERATE"`, avg_tokens per the design doc Table §5.1.

**v2 weights** (override v1):

| subset | v1 weight | v2 weight | avg_tokens |
|---|---:|---:|---:|
| tulu3 | 0.30 | 0.11 | 600 |
| math | 0.15 | 0.09 | 800 |
| qa | 0.10 | 0.05 | 400 |
| creative | 0.10 | 0.05 | 600 |
| multilingual | 0.10 | 0.08 | 400 |
| fineweb | 0.05 | 0.05 | 1500 |
| papers | 0.05 | 0.05 | 300 |

---

### B.2 `mot_math` — `open-r1/Mixture-of-Thoughts` config=`math` split=`train` — TEACHER_FORCED

**Row schema** (verified Hub):
```
{
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<think>...</think>\n\nFinal answer..."}],
  "num_tokens": int64,
  "source": str            # original prompt provenance
}
```
**Extraction**:
- `prompt`: `row["messages"][0]["content"]` (first user message). Assert `row["messages"][0]["role"] == "user"`; if not, skip the row (defensive; the dataset's preview confirms the user is index 0).
- `canonical_completion`: `row["messages"][1]["content"]` (the assistant message; contains `<think>...</think>final_answer`). Assert `row["messages"][1]["role"] == "assistant"`; if not, skip.
- `domain`: `"mot_math"`.
- `policy`: `"TEACHER_FORCED"`.
- avg_tokens: 3500 (per design doc §5.1).

**Subset config**: `config="math"`, `split="train"`.

---

### B.3 `mot_code` — `open-r1/Mixture-of-Thoughts` config=`code` split=`train` — TEACHER_FORCED

Same schema as B.2. Same extraction logic. `domain="mot_code"`. avg_tokens: 15000 (long traces; per design doc §5.1).

**Subset config**: `config="code"`, `split="train"`.

---

### B.4 `mot_science` — `open-r1/Mixture-of-Thoughts` config=`science` split=`train` — TEACHER_FORCED

Same schema as B.2. Same extraction. `domain="mot_science"`. avg_tokens: 2500.

**Subset config**: `config="science"`, `split="train"`.

**Note**: design doc §10 cite-table flags science as "mixed source — partly Llama-Nemotron, not pure R1"; behavior identical regardless of source.

---

### B.5 `swe_smith` — `SWE-bench/SWE-smith-trajectories` split=`xml` — TEACHER_FORCED, multi-turn flattened

**Row schema** (verified Hub):
```
{
  "messages":     str              # JSON-serialized list[{role, content}]; MUST parse first
  "instance_id":  str
  "resolved":     bool
  "model":        str              # "anthropic/claude-3-7-sonnet-20250219"
  "traj_id":      str
  "patch":        str
}
```
**Critical**: `messages` is a STRING (JSON-encoded), not a native list. Must `json.loads(row["messages"])` first.

**Extraction**:
1. `parsed = json.loads(row["messages"])` — list of `{role, content}` dicts.
2. Find first message with `role="user"` → `prompt = first_user["content"]`. (The xml split's `parsed[0]` is system, `parsed[1]` is the first user.)
3. Find first message with `role="assistant"` AFTER the first user → `canonical_completion = first_assistant["content"]`. This first assistant turn contains the initial tool-call (XML `<function=...>` blocks) plus any pre-tool reasoning. Subsequent turns (tool observations → next assistant turn → ...) are DISCARDED.
4. Drop the system message from `parsed` — Qwen3.6's `apply_chat_template` re-injects its own system header at calibration consumption time. Including SWE-smith's system header would double up the agent-bash-tool preamble and inflate the token count.
5. NO `<tool_call>` wrapping is added at extract time. The SWE-smith xml split's assistant content already contains literal `<function=...><parameter=...>...</parameter></function>` strings, which `apply_chat_template` renders as-is inside the assistant content block. The Qwen3 template will not re-wrap these because they appear in the assistant turn's content field (not in a separate `tool_calls` slot). This produces calibration tokens with `<function=...>` literals embedded in assistant content — acceptable for routing-pattern supervision per design doc §5.4.
6. `domain`: `"swe_smith"`.
7. `policy`: `"TEACHER_FORCED"`.
8. avg_tokens: 8000 (long; design doc §5.1 says median ~20K but we are taking first assistant only, ~half of that).

**Truncation strategy** (design doc §5.4 footnote, also caveat §8.3):
- The downstream `_tokenize_to_fixed_length` hard-truncates at `sequence_length=4096` tokens.
- A first-assistant-only turn from xml split typically falls in the 3K-12K token range. ~30-50% of swe_smith rows will be truncated at 4096.
- Acceptable: the first 4K tokens always contain the prompt + initial reasoning + first `<function=...>` block, which is the routing-relevant content.

**Subset config**: `config=None` (default config), `split="xml"`.

---

### B.6 `function_calling` — `glaiveai/glaive-function-calling-v2` split=`train` — GENERATE

**Row schema** (verified Hub):
```
{
  "system":  str    # "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required - { json_schema }"
  "chat":    str    # "USER: ... ASSISTANT: ... <|endoftext|>  FUNCTION RESPONSE: ... ASSISTANT: ... <|endoftext|>  USER: ..."
}
```
Both fields are flat strings (no parsing of nested JSON required). `chat` uses literal `USER:`, `ASSISTANT:`, `FUNCTION RESPONSE:` markers with `<|endoftext|>` separators between turns.

**Extraction**:
1. `system_text = row["system"]` — strip the `"SYSTEM: "` prefix if present. Keep the function-schema JSON literal in the system content.
2. Parse `row["chat"]` to extract the FIRST `USER: ...` block:
   - Find `"USER:"` occurrence (assert at least one exists; if not, skip row).
   - Slice from just after `"USER:"` to the next occurrence of any of: `"ASSISTANT:"`, `"FUNCTION RESPONSE:"`, `"USER:"` (after the first), or end-of-string. Strip whitespace.
   - This is the user's first turn.
3. **Prompt format**: prepend the system content as a chat-template system turn. Since `_iter_prompts_from_qwen3_pretrain_mix_v2` returns `(prompt, domain, canonical, policy)` as a single user-prompt string (and the generation script wraps it in `[{role: "user", content: prompt}]` only — see `build_self_traces_calib.py` line 526), the implementer must choose ONE of:
   - **Option (i)** — concatenate: `prompt = f"{system_text}\n\n{user_text}"`. Coarsest; the teacher's chat template will render this as one user turn that contains the system preamble. Tool schemas are visible to the teacher but not in a separate system role.
   - **Option (ii)** — extend the iterator's tuple to a 5-tuple `(prompt, domain, canonical, policy, system_prompt | None)`, and widen the generation script's `rendered = tokenizer.apply_chat_template([...])` call to prepend a `{"role": "system", "content": system_prompt}` when present.
   - **Recommendation**: Option (i) for v2 (smaller surface; keeps the iterator's tuple shape stable). Option (ii) is a follow-up if the function-calling subset shows clear under-performance in Stage 6 evals.
4. `canonical_completion`: `None` (GENERATE policy — discard the canonical `ASSISTANT:` turn since it's GPT-3.5-style JSON, not Qwen3 `<tool_call>` format).
5. `domain`: `"function_calling"`.
6. `policy`: `"GENERATE"`.
7. avg_tokens: 500 (short conversations; the function schema in `system` adds ~200 tokens, the first user turn adds ~50-150).

**Document this footnote** in `calibration_mix_v2.md` (Section G): "function_calling subset uses Glaive-v2's flat USER/ASSISTANT chat format; we wrap the schema-rich system prompt and first USER turn together as a single Qwen3 user turn (Option i). The teacher then generates its own `<tool_call>...</tool_call>` response in thinking mode."

**Halt trigger**: if `glaiveai/glaive-function-calling-v2` returns 401/403 on `load_dataset`, halt and surface (Section I).

---

## Section C — TEACHER_FORCED code path

### C.1 Policy plumbing
- Add `_QWEN3_MIX_V2_POLICY: dict[str, str]` to `calibration.py` (Section A.1 step 2).
- The iterator (`_iter_prompts_from_qwen3_pretrain_mix_v2`) emits 4-tuples carrying the policy. Build script main loop reads the policy.

### C.2 Skip-generation branch (build scripts)
- In `_generate_traces` (HF) and `_process_outputs` + main chunk loop (vLLM):
  - Partition the batch by policy.
  - GENERATE rows take the existing path (`model.generate` / `LLM.generate`).
  - TEACHER_FORCED rows skip the engine entirely. The build script writes a JSONL row using the canonical completion directly: `messages=[{user: prompt}, {assistant: canonical_completion}]`.

### C.3 Schema bump: `completion_source` field
- Each JSONL row gains `completion_source: str ∈ {"teacher_generated", "canonical"}`.
- HF script (`build_self_traces_calib.py`) bumps `_trace_cache_key` `schema_version` from 6 → 7 and adds `completion_source` to the v9-equivalent payload schema documentation.
- vLLM script (`build_self_traces_calib_vllm.py`) bumps `_trace_cache_key_vllm` `schema_version` from 8 → 9.
- Existing JSONLs on disk lack the `completion_source` field; the downstream loader must treat its absence as `"teacher_generated"` (legacy default — see C.4). NO format change to the downstream loader is required for backward compat; `completion_source` is consumed only at analysis time (NOT at calibration consumption time).

### C.4 Downstream loader compatibility
- `max_quality/src/moe_compress/utils/calibration.py` `_stream_texts_self_traces` (lines ~1685-1767) reads JSONL rows and renders via `apply_chat_template`. It currently filters on `_complete=False`. **No changes required**: TF rows are written with `_complete=True` by the build script, and `apply_chat_template` doesn't care whether the assistant content was teacher-generated or canonical. The `completion_source` field is metadata-only.
- The teacher's FORWARD pass (calibration capture step, e.g., Stage 2 profile, Stage 2.5 router KD) runs on the same rendered tokens regardless of source. This is the standard self-traces contract per `calibration.py` line 1299-1320.

### C.5 Generation-loop refactor minimization
The implementer should NOT refactor `_generate_traces` into a strategy-pattern or introduce a class hierarchy. The minimal change is:
1. Inside the batch loop, partition `batch_prompts` and `batch_domains` arrays by checking each tuple's 4th element (if present and `=="TEACHER_FORCED"`).
2. Skip the `tokenizer(rendered, ...)` + `model.generate(...)` call for TF prompts.
3. For each TF prompt, synthesize a row dict directly (mimicking the existing yield-shape at lines 670-678) and yield it.
4. GENERATE rows continue through the existing code path unchanged.

Total `_generate_traces` delta: ~40 LoC.
Total `_process_outputs` (vLLM) delta: ~30 LoC plus a new `_synth_teacher_forced_rows` helper (~50 LoC).

### C.6 What "teacher must still forward TF rows" actually means
The campaign brief item §C-4 says "The teacher must still do a FORWARD pass on (prompt + canonical_completion) at calibration consumption time — that's how cov/router stats are captured. The TEACHER-FORCED policy only skips the GENERATION step."

This is automatically satisfied: the JSONL self-traces loader (`_stream_texts_self_traces`) builds calibration tensors from the rendered (prompt + assistant_content) of every row — the assistant content is the canonical completion for TF rows. Stage 2 / 2.5 / 3 pass those tensors through the (compressed) model's forward and capture cov/router/REAP/imatrix accumulators as usual. **No code changes are needed in the calibration consumption path.** The implementer should document this in the docstring comment for the new build-script TF branch but not change consumer-side code.

---

## Section D — `sequence_length: 2048 → 4096` bump

### D.1 Primary changes
- `max_quality/configs/qwen36_35b_a3b_30pct.yaml` line 53: `sequence_length: 4096`.
- `max_quality/configs/qwen36_35b_a3b_reap_exact.yaml` line 67: `sequence_length: 4096`.

### D.2 Downstream references inventoried
Search performed at HEAD: `grep -rn "2048" max_quality/src/ max_quality/scripts/ max_quality/tests/ max_quality/configs/`. Findings:

**SAFE — do NOT modify** (these pin v1 behavior intentionally or reference other unrelated 2048 sizes):
- `max_quality/src/moe_compress/stage6/plugins/imatrix_export.py:233,359` — `ctx_size` default for imatrix export (independent of calibration sequence_length).
- `max_quality/src/moe_compress/utils/cov_sqrt.py:34,78` — hidden_size=2048 comments (Qwen3.6 has hidden=2048; unrelated to seq_len).
- `max_quality/src/moe_compress/stage2/profiling.py:44` — comment about `hidden_size=2048`.
- `max_quality/src/moe_compress/stage1/plugins/ablation_filter.py:100` — comment about logits memory at hidden=2048.
- `max_quality/src/moe_compress/stage3/plugins/block_hidden_cache.py:52` — comment ("128 × 2048 = 262K tokens") describing the OLD baseline; can update to "128 × 4096 = 524K tokens" as a doc-only edit, OR leave as-is since it's prose.
- `max_quality/src/moe_compress/stage6alt/plugins/thermo_corpus.py:126,196` — thermometer corpus has its OWN `sequence_length: 2048` default keyed off the `thermometer:` section of the YAML, NOT off `calibration.sequence_length`. Do not touch.
- `max_quality/tests/test_stage6_plugin_wikitext.py:212-289` — WikiText PPL test pins `sequence_length == 2048` per spec F-S-M-1 (different concern; WikiText eval has its own seq_len). Do not touch.
- `max_quality/tests/test_run_pipeline_*.py:54,63` — synthesizes minimal YAML with `sequence_length: 16` for unit-test speed. Independent of v2.

**MUST MODIFY** — comment/doc-only:
- `max_quality/scripts/build_self_traces_calib.py` lines 539-554: prompt-length truncation logic warns/truncates at 2048 input tokens. **Leave as-is** because:
  - The 2048 cap is on the *prompt* input length (before generation), not on the calibration sequence length.
  - Most v2 user prompts (e.g. tulu3, math, fineweb) easily fit under 2048 tokens.
  - SWE-smith prompts may exceed 2048 (they include large `<IMPORTANT>...</IMPORTANT>` system preambles), but swe_smith is TEACHER_FORCED — the prompt-length truncation block is in the GENERATE path only.
  - Action: update the comment at line 539 to clarify "2048-token prompt cap is independent of calibration `sequence_length` which the consumer truncates separately."

**MUST MODIFY** — code:
- None. The calibration consumer (`_tokenize_to_fixed_length` in `calibration.py`) reads `spec.sequence_length` from the YAML and truncates to that value. The 2048 → 4096 change is purely a YAML edit.

### D.3 Memory footprint implications
Doubling `sequence_length` doubles per-prompt forward-pass activation memory in the calibration consumer:
- Stage 1/2 phase A/B forward batches at `phase_a_batch_size=32`, `phase_b_batch_size=16` (per `qwen36_35b_a3b_30pct.yaml:66-69`).
- At 4096 × 16 batch, Phase B will use ~2× the activation memory vs. 2048 × 16. The YAML's existing 32 GB headroom comment (line 70) becomes ~14 GB. Still OK on H200 141 GB.
- If the implementer wants safety margin: drop `phase_b_batch_size: 16 → 8`. This is a SEPARATE decision and should be surfaced as a Section I halt-trigger if OOM appears during smoke validation.

### D.4 Sidecar size estimates
- The `block_hidden` cache (Stage 3) scales linearly with `num_sequences × sequence_length`. At 4000 × 4096 = 16.4M tokens × hidden_dim × layers × fp16 = ~2× the 2048 baseline. Document this in the new `calibration_mix_v2.md` (Section G).
- The `covariance` and `reap_scores` sidecars do NOT scale with sequence_length (they aggregate by layer × expert).

---

## Section E — Cache-key plumbing

### E.1 Cache-key uniqueness (v1 vs v2)
- `_trace_cache_key` (HF, `build_self_traces_calib.py`) folds `prompts_source = f"{args.prompts}#{args.num_prompts}#{args.seed}"`. With `args.prompts="qwen3-pretrain-mix-v2"` vs `"qwen3-pretrain-mix"`, the strings differ → sha256 differs → cache_key differs. **Confirmed by inspection.**
- `_trace_cache_key_vllm` (vLLM, `build_self_traces_calib_vllm.py`) uses the same field via the CLI wrapper. Same guarantee.
- `CalibrationSpec.cache_key` (`calibration.py` line 87) includes `source` in the payload. v2 spec has `source="qwen3-pretrain-mix-v2"` while v1 has `source="qwen3-pretrain-mix"` → cache keys differ. **Confirmed.**

### E.2 Cache-key stability across re-runs
- All inputs to the cache_key derivation are explicit args / config fields. No implicit timestamps or randomized inputs. Re-runs with identical args produce identical keys.

### E.3 Schema-version bump separation
- `_trace_cache_key`'s `schema_version` (HF script): bump 6 → 7 because we add `completion_source` to the row schema.
- `_trace_cache_key_vllm`'s `schema_version`: bump 8 → 9 (same reason; the vLLM-specific JSONL gains the new field).
- `CalibrationSpec.cache_key`'s `_schema_version` (in `calibration.py`): **do NOT bump**. The cache key correctly invalidates by `source` change alone; the calibration tensor itself isn't gaining new fields.

### E.4 Backward compat (cache files already on disk)
- Existing self_traces JSONL files on disk under the OLD cache_key remain present and continue to be cache-hit IF the operator runs with `args.prompts="qwen3-pretrain-mix"` and the old schema_version. **No deletion required.**
- Existing self_traces JSONLs are not invalidated by adding v2 — they're keyed off the v1 corpus name + v1 schema_version, which v2 does not touch.

### E.5 Halt trigger
If unit-test `test_cache_key_distinct_for_v1_v2` (Section F item 5) shows the v1 and v2 cache_keys hash to the same value, halt (Section I.5).

---

## Section F — Tests

### F.1 `test_calib_jsonl_schema_v8.py` (extended)
**Modifications** (per Section A.6):
1. `test_cache_key_carries_schema_version_8` → renamed to `test_cache_key_carries_schema_version_9`; assert payload contains `"schema_version": 9` and `"completion_source"` doc-string reference. Update the expected_payload literal at line 332-345 to include `schema_version=9`.
2. NEW `test_jsonl_row_v9_contains_completion_source`: drives a GENERATE row through `_process_outputs`; asserts `row["completion_source"] == "teacher_generated"`.
3. NEW `test_teacher_forced_row_completion_source_canonical`: drives a TF prompt (4-tuple, policy="TEACHER_FORCED", canonical_completion="<think>...</think>foo") through `_synth_teacher_forced_rows`; asserts:
   - `row["completion_source"] == "canonical"`
   - `row["messages"][1]["content"] == "<think>...</think>foo"`
   - `row["_complete"] is True`
   - `row["n_gen_tokens"] == 0`
   - `row["has_think"] is True`
   - `row["refusal_flag"] is False`
4. NEW `test_teacher_forced_row_n_prompt_tokens_uses_tokenizer`: same as F.1.3 but with a stub tokenizer; asserts `row["n_prompt_tokens"]` reflects the canonical-prompt-rendered length.

---

### F.2 `test_run_pipeline_reap_exact.py`
**Modifications** (per A.7):
- Line 60: `"source": "qwen3-pretrain-mix-v2"`.
- No new tests needed; existing tests should pass with the corpus name change as long as `qwen3-pretrain-mix-v2` is registered.

---

### F.3 `test_run_pipeline_normal_mode_regression.py`
**Modifications**: line 51: `"source": "qwen3-pretrain-mix-v2"`.

---

### F.4 `test_reap_exact_config.py`
**Modifications** (per A.8): assert source is v2 and sequence_length is 4096.

---

### F.5 NEW — `max_quality/tests/test_qwen3_pretrain_mix_v2.py`
~200 LoC. CPU-only. No network. Uses hand-built row fixtures.

#### F.5.1 `test_weights_sum_to_one`
- Asserts `sum(_QWEN3_MIX_V2_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-9)`.
- Asserts `set(_QWEN3_MIX_V2_WEIGHTS) == set(_QWEN3_MIX_V2_DATASET) == set(_QWEN3_MIX_V2_AVG_TOKENS) == set(_QWEN3_MIX_V2_POLICY) == set(_QWEN3_MIX_V2_DATASET_CONFIG) == set(_QWEN3_MIX_V2_DATASET_SPLIT)`.
- Asserts the 12 expected subset keys are present.

#### F.5.2 `test_policy_values_are_valid`
- For each subset, assert `_QWEN3_MIX_V2_POLICY[subset] in {"GENERATE", "TEACHER_FORCED"}`.

#### F.5.3 `test_corpus_registered`
- After importing `calibration`, assert `get_corpus_adapter("qwen3-pretrain-mix-v2")` returns a `CorpusAdapter` with the correct `name`.
- Assert `get_corpus_adapter("qwen3-pretrain-mix")` STILL returns a working v1 adapter (backward compat).

#### F.5.4 `test_cache_key_distinct_for_v1_v2`
- Build two `CalibrationSpec` instances with identical fields except `source`. Compare `cache_key("tok").{}` outputs — must differ.

#### F.5.5 `test_iter_prompts_v2_returns_4tuples` (parametric — one parameter per subset)
For each subset key, hand-build a 1-row fake dataset that matches the verified schema (Section B). Monkeypatch `_shuffled_stream` to yield that single row. Call `_iter_prompts_from_qwen3_pretrain_mix_v2(num_prompts=1, seed=0)`. Assert:
- The first yielded tuple has length 4.
- `tuple[0]` (prompt) is non-empty str.
- `tuple[1]` (domain) == subset key.
- For GENERATE subsets: `tuple[2]` is None, `tuple[3]` == `"GENERATE"`.
- For TEACHER_FORCED subsets: `tuple[2]` is non-empty str (canonical completion), `tuple[3]` == `"TEACHER_FORCED"`.

Subsets to cover: all 12. Fixture data:
- `mot_math`: `{"messages": [{"role":"user","content":"What is 2+2?"},{"role":"assistant","content":"<think>two plus two</think>4"}], "num_tokens": 100, "source": "test"}`
- `mot_code`: same shape but with code prompt.
- `mot_science`: same shape.
- `swe_smith`: `{"messages": '[{"role":"system","content":"sys"},{"role":"user","content":"<uploaded>fake</uploaded>"},{"role":"assistant","content":"<function=bash><parameter=command>ls</parameter></function>"}]', "instance_id":"x", "resolved": true, "model":"claude", "traj_id":"y", "patch":""}`
- `function_calling`: `{"system":"SYSTEM: You are helpful. Use { ... }", "chat":"USER: hi  ASSISTANT: hello <|endoftext|>"}`
- Existing 7: minimal fixtures matching their schemas (e.g., `tulu3`: `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hi back"}]}`).

#### F.5.6 `test_swe_smith_flatten_drops_system_and_subsequent_turns`
- Hand-build a SWE-smith row with system + user + assistant1 + tool + user + assistant2.
- Run the extractor.
- Assert `prompt == "<first user content>"` and `canonical_completion == "<assistant1 content>"`.
- Assert the assistant2 content is NOT in either field.

#### F.5.7 `test_glaive_extracts_first_user_turn`
- Fixture: `chat="USER: pick a function  ASSISTANT: <functioncall> {...} <|endoftext|>   FUNCTION RESPONSE: ...   ASSISTANT: ok <|endoftext|>   USER: ignored"`.
- Assert extracted prompt starts with `"pick a function"` and does NOT contain `"ignored"` (the second USER turn must be dropped).

#### F.5.8 `test_diversity_floor_no_warning_at_num_prompts_8000`
- Compute `min(_QWEN3_MIX_V2_WEIGHTS.values())` (== 0.05 — multiple subsets tied at 5%).
- Threshold = `int(2 * 12 / 0.05) == 480`.
- **The existing streamer's threshold check compares `num_prompts` (total) to threshold, NOT per-subset count** (see `build_self_traces_calib.py:175-188` for v1; v2 mirrors v1 semantics). At `num_prompts=8000`, 8000 ≫ 480 → **no warning expected**.
- Test: capture logging at WARNING level (use `caplog` fixture), call `_iter_prompts_from_qwen3_pretrain_mix_v2(num_prompts=8000, seed=0)` with monkeypatched `_shuffled_stream` returning empty iters; assert NO "diversity threshold" warning is logged.

#### F.5.9 `test_diversity_floor_warning_at_low_num_prompts`
- At `num_prompts=300` < threshold 480, the streamer must emit the diversity-threshold warning exactly once.
- Test: same `caplog` fixture and monkeypatched `_shuffled_stream`; call with `num_prompts=300`; assert exactly one WARNING-level log contains substring "diversity threshold".

**Note**: this preserves v1 semantics (warn when `num_prompts < threshold`). If we later want per-subset-aware threshold checking (warn when smallest-subset-count < threshold), that's a separate behavior change that affects v1 too and needs its own design discussion — out of scope here.

---

## Section G — Documentation: `max_quality/docs/calibration_mix_v2.md`

~150 LoC markdown. Structure:

### G.1 Header
- Title, status (NEW corpus, sibling to v1), date 2026-05-27, branch `feat/calibration-v2`.
- One-sentence summary: 12-subset reasoning-mode mix optimized for Qwen3.6-thinking calibration.

### G.2 Mix table (the full 12-row table from this plan's header).
Include columns: #, subset key, weight, HF dataset, config, split, policy, multi-turn, notes.

### G.3 Policy rationale (Generate-vs-Teacher-Forced)
- Why GENERATE for non-thinking-format sources (tulu3, math, dolly, writingprompts, aya, fineweb, papers, function_calling): canonical completions are not in `<think>...</think>` format; we want the teacher's own thinking-mode output.
- Why TEACHER_FORCED for MoT × 3: canonical R1 traces ARE in `<think>...</think>` format; reuse them directly to save ~3-4 GPU-hours of generation.
- Why TEACHER_FORCED for swe_smith: canonical Claude 3.7 trajectories provide authentic tool-call routing patterns; we accept zero `<think>` supervision on this subset (Claude trajectories don't include `<think>` blocks).
- Why no MoT-all (the 350K combined config): we pick math/code/science separately so the mix-weights are explicit; using `all` would let the source distribution dictate proportions.

### G.4 Token-budget comparison (table)
| Variant | num_prompts | seq_len | num_sequences | Total calib tokens | Notes |
|---|---:|---:|---:|---:|---|
| REAP paper (arXiv:2510.13999 §5) | n/a | 2048 | 1024 | 2.10M | c4 + evol-codealpaca; demonstrated-sufficient floor for ~30B-class MoE |
| REAM paper (arXiv:2604.04356 §5) | n/a | 512 | 3072 | 1.57M | shorter sequences; for reference only |
| GLM-4.7 recipe (Eastern Island letter) | n/a | 16384 | 24576 | 402.6M | 6 sources × 4096 samples each; calibrated for ~357B model, ~190× heavier than REAP paper baseline |
| Our v1 (`qwen3-pretrain-mix`) | 6500 | 2048 | 4000 | 8.2M | 8 subsets, all GENERATE; ~4× REAP paper baseline |
| Our v2 (`qwen3-pretrain-mix-v2`) | 8000 | 4096 | 4000 | 16.4M | 12 subsets, hybrid GENERATE/TF; ~8× REAP paper baseline, ~25× lighter than GLM-4.7; right-sized for our 35B-A3B target serving Stages 1→6 (not just Stage 2) |

### G.5 How to use
```yaml
calibration:
  source: qwen3-pretrain-mix-v2
  seed: 1337
  num_sequences: 4000
  sequence_length: 4096
```
Build self-traces:
```bash
python max_quality/scripts/build_self_traces_calib_vllm.py \
    --teacher Qwen/Qwen3.6-35B-A3B \
    --prompts qwen3-pretrain-mix-v2 \
    --num-prompts 8000 \
    --max-new-tokens 16384 \
    --reasoning-budget 4096 \
    --output artifacts/_shared/self_traces.jsonl
```

### G.6 Expected build time
- ~4160 GENERATE prompts at vLLM bs≥4 ≈ ~5-7 hours on H200 SXM5. ~3840 TF prompts add <30 minutes (no generation, just JSONL synth + tokenizer pass).
- Total: ~5-7 hours. (Per `feedback_calibration_oom_ladder.md`, --load-in-4bit + bs≥4 is the practical fit on H200 for non-FP8 teachers.)

### G.7 Migration from v1
- Switching v1 → v2 is a YAML edit only (`source: qwen3-pretrain-mix → qwen3-pretrain-mix-v2`, `sequence_length: 2048 → 4096`).
- Old `qwen3-pretrain-mix` cache JSONLs on disk are not invalidated — they remain valid for any pipeline run still pointing at v1.
- v2 produces a NEW self_traces JSONL under a different cache_key, so the two can coexist.

### G.8 Known limitations / footnotes
1. **function_calling format gap**: Glaive's `<functioncall>` JSON is not Qwen3's `<tool_call>` XML format. We use Glaive's prompts (system + first USER) and discard its canonical, letting the teacher generate Qwen3-native `<tool_call>` responses. This means we get on-distribution routing for the system+user turns but the assistant turn coverage is bounded by the teacher's actual tool-call propensity.
2. **swe_smith multi-turn flatten**: only the first (user, assistant_with_tool_call) pair is kept; ~80% of each trajectory's tokens are discarded. Future work could extend the JSONL schema to support multi-turn assistant traces. Tracked as design-doc §6.3 option (b).
3. **swe_smith lacks `<think>` blocks**: Claude 3.7 was not in extended-thinking mode when these trajectories were generated. We get tool-call routing supervision but zero `<think>`-token supervision on this 12% of the mix.
4. **mot_science source mix**: per design doc §1.2d, the science split is partly Llama-Nemotron-Post-Training, not pure R1. Treat as the weakest of the three MoT subsets for the R1-alignment assumption.
5. **No GPU validation gating**: design doc §7.1 proposed a 100-prompt KL test before commit. The campaign brief defers that; if Stage 6 metrics drop dramatically after the first v2 calibration run, fall back to TF→GENERATE for MoT subsets (Section I).

### G.9 Cross-references
- Design doc: `tasks/CALIBRATION_MIX_V2_DESIGN.md`.
- Plan doc: `tasks/CALIBRATION_MIX_V2_PLAN.md` (this file).
- Implementation lives in `max_quality/src/moe_compress/utils/calibration.py` (corpus adapter) and `max_quality/scripts/build_self_traces_calib*.py` (prompt iterators, TF synthesis).

---

## Section H — Implementation order (topologically sorted)

The implementer agent should execute in this exact order; each step ends with a committable green-tests checkpoint. Each step ~50-200 LoC.

### Step 1 — Constants + parse-yaml + registration (~80 LoC)
File: `max_quality/src/moe_compress/utils/calibration.py`
- Add `_QWEN3_MIX_V2_WEIGHTS`, `_QWEN3_MIX_V2_AVG_TOKENS`, `_QWEN3_MIX_V2_DATASET`, `_QWEN3_MIX_V2_DATASET_CONFIG`, `_QWEN3_MIX_V2_DATASET_SPLIT`, `_QWEN3_MIX_V2_POLICY`, and `_broad_instruct_mix_v2_intro_logged: bool = False`.
- Add `_parse_yaml_qwen3_pretrain_mix_v2`.
- Add a *placeholder* `_stream_texts_qwen3_pretrain_mix_v2` that raises `NotImplementedError` (filled in step 3).
- Register the v2 corpus.
- New test file `test_qwen3_pretrain_mix_v2.py` with F.5.1, F.5.2, F.5.3, F.5.4. (`test_corpus_registered`, weights-sum, policy-validity, cache-key-distinct.)
- Commit: "feat(calib-v2): register qwen3-pretrain-mix-v2 corpus, constants + parse_yaml".

### Step 2 — Widen `_shuffled_stream` for config/split (~20 LoC, source + 1 test)
File: `max_quality/src/moe_compress/utils/calibration.py`
- Add optional `config: str | None = None`, `split: str = "train"` kwargs to `_shuffled_stream`; pass them to `load_dataset(name, config, split=split, streaming=True)` if non-default.
- All v1 callsites pass nothing (kwargs default to v1 behavior). Confirm v1 tests still pass.
- Commit: "feat(calib-v2): widen _shuffled_stream with config/split kwargs".

### Step 3 — v2 streamer helpers + `_stream_texts_qwen3_pretrain_mix_v2` (~150 LoC, source + tests)
File: `max_quality/src/moe_compress/utils/calibration.py`
- Add `_stream_messages_with_config`, `_stream_swe_smith_xml`, `_stream_glaive_function_calling`.
- Fill in the `_stream_texts_qwen3_pretrain_mix_v2` body (dispatch by subset key, calling existing v1 helpers for the 7 carryover subsets and the new helpers for the 5 new ones).
- Helper-level unit tests: extend `test_qwen3_pretrain_mix_v2.py` with the 5 new-subset row-extraction smoke tests (a parametric version of F.5.5 covering all 12 — F.5.5).
- Commit: "feat(calib-v2): _stream_texts_qwen3_pretrain_mix_v2 + 5 new row extractors".

### Step 4 — HF build-script prompt iterator (~120 LoC)
File: `max_quality/scripts/build_self_traces_calib.py`
- Add `_iter_prompts_from_qwen3_pretrain_mix_v2` (returns 4-tuples).
- Helper internal to the iterator for swe_smith JSON-parse + multi-turn flatten.
- Helper internal to the iterator for glaive `USER:` extraction.
- Tests: F.5.5 (parametric, all 12), F.5.6 (swe_smith), F.5.7 (glaive), F.5.8/F.5.9 (diversity floor).
- Commit: "feat(calib-v2): HF prompt iterator for qwen3-pretrain-mix-v2".

### Step 5 — HF build-script TEACHER_FORCED branch in `_generate_traces` (~50 LoC)
File: `max_quality/scripts/build_self_traces_calib.py`
- Widen `_generate_traces` to detect 4-tuples and skip generation for TF rows.
- Add `completion_source` to every yielded row.
- Bump `_trace_cache_key` `schema_version` 6 → 7.
- Update CLI dispatch at `build_self_traces_calib.py:929-930` (currently `if args.prompts == "qwen3-pretrain-mix":`) to add an `elif` branch:
  ```python
  if args.prompts == "qwen3-pretrain-mix":
      prompts_iter = _iter_prompts_from_qwen3_pretrain_mix(
          num_prompts=args.num_prompts, seed=args.seed,
          prev_num_prompts=args.prev_num_prompts or None,
      )
  elif args.prompts == "qwen3-pretrain-mix-v2":
      prompts_iter = _iter_prompts_from_qwen3_pretrain_mix_v2(
          num_prompts=args.num_prompts, seed=args.seed,
          prev_num_prompts=args.prev_num_prompts or None,
      )
  else:
      prompts_iter = _iter_prompts_from_jsonl(Path(args.prompts))
  ```
  Also update the `--prompts` argument help text at `:772-773` to list `qwen3-pretrain-mix-v2` as an accepted value.
- Update file-top docstring "Output schema" line to mention `completion_source` field.
- No new tests for `_generate_traces` integration (it's a generator over a model.generate call — needs a model). The existing CLI exercise will catch wiring in step 8 smoke.
- Commit: "feat(calib-v2): HF build script — TEACHER_FORCED skip-generation branch".

### Step 6 — vLLM build-script TEACHER_FORCED branch (~80 LoC + test)
File: `max_quality/scripts/build_self_traces_calib_vllm.py`
- Extend imports to include `_iter_prompts_from_qwen3_pretrain_mix_v2` from `build_self_traces_calib`.
- Add `_synth_teacher_forced_rows` helper.
- Refactor `main()` chunk loop to partition by policy; submit only GENERATE rows to vLLM.
- Extend `_process_outputs` to set `completion_source="teacher_generated"` on every row it produces.
- Bump `_trace_cache_key_vllm` `schema_version` 8 → 9.
- Update CLI dispatch at `build_self_traces_calib_vllm.py:929-930` mirroring the HF script's `elif args.prompts == "qwen3-pretrain-mix-v2":` branch (calls the same imported `_iter_prompts_from_qwen3_pretrain_mix_v2`).
- Update `--prompts` argument help text at `:451-452` to list `qwen3-pretrain-mix-v2` as an accepted value.
- Update file-top docstring "Output schema" line to mention `completion_source` field.
- Update test_calib_jsonl_schema_v8.py per F.1 (rename schema_version check to assert 9, add F.1.2-4).
- Commit: "feat(calib-v2): vLLM build script — TEACHER_FORCED synth + schema v9".

### Step 7 — YAML config updates + dependent tests (~30 LoC)
Files: `max_quality/configs/qwen36_35b_a3b_30pct.yaml`, `max_quality/configs/qwen36_35b_a3b_reap_exact.yaml`, `max_quality/tests/test_reap_exact_config.py`, `max_quality/tests/test_run_pipeline_reap_exact.py`, `max_quality/tests/test_run_pipeline_normal_mode_regression.py`.
- Update YAML comments + `source` + `sequence_length`.
- Update tests' string assertions.
- Commit: "feat(calib-v2): switch production YAMLs to qwen3-pretrain-mix-v2 + seq_len=4096".

### Step 8 — Documentation (~150 LoC markdown)
Files: `max_quality/docs/calibration_mix_v2.md`, optionally `max_quality/patches/MANIFEST.md` (3-line note).
- Per Section G.
- Commit: "docs(calib-v2): user-facing doc + manifest schema-bump note".

### Step 9 — Full test sweep (no new code)
Run `pytest max_quality/tests/` end-to-end. Triage failures. If everything is green, the implementation is done.

---

## Section I — Halt triggers

The implementer agent must STOP and surface to the user (NOT silently work around) when any of these occur:

### I.1 HF dataset 404 / 401 / gated-access
**Condition**: `load_dataset("<name>")` returns HTTPError 404 / 401 / 403 on first load (excluding transient 5xx which should be retried).
**Affected**: any of the 12 datasets. Specifically `glaiveai/glaive-function-calling-v2` is a known watchpoint (verified open on 2026-05-27, but Glaive has flipped gating in the past).
**Why halt**: silently switching to an alternative dataset would diverge from the planned mix; surface and let the user accept or pick an alternative.
**Action**: log the exact dataset name and error code; exit nonzero; do not commit.

### I.2 Row-schema drift on a v2 subset
**Condition**: a v2 row-extractor's first attempt to fetch a sample row (e.g. during the F.5.5 test fixtures or during Step 4 implementation) finds the row's actual schema differs from Section B's recipe (e.g. `mot_math` row has no `messages` key, or `swe_smith.messages` is already a list rather than a JSON-encoded string).
**Why halt**: design doc and this plan encode the row schema verbatim; silent adaptation can produce wrong calibration text.
**Action**: surface the ACTUAL schema dump (`row.keys()` + a sample row's structure) and the planned schema; do not commit until the user resolves.

### I.3 Diversity-floor failure at smallest weight
**Condition**: at `num_prompts=8000`, the smallest-weight subset (5% — papers / qa / creative / fineweb tied) yields strictly fewer than 400 rows (computed integer floor) during the F.5.5 fixture run. Or the test `test_diversity_floor_warning_at_num_prompts_8000` fails to emit the expected warning.
**Why halt**: under-represented subsets may produce a calibration that misses tail domains.
**Action**: surface the per-subset row counts; user may bump `num_prompts` or shift weights.

### I.4 Golden-snapshot or downstream test breakage from seq_len=4096
**Condition**: any test in `max_quality/tests/golden/` or `test_stage*.py` fails after the YAML seq_len change.
**Why halt**: those tests pin v1 sequence shapes intentionally; breaking them silently corrupts pinned reference signals.
**Action**: surface the failing test names; user decides whether to re-baseline.

**Audit done in this plan** (Section D.2): no golden-snapshot test was found that depends on `calibration.sequence_length=2048`. The WikiText test that pins `sequence_length==2048` is an independent eval setting (Section D.2 SAFE list). Surfacing this preemptively: if the implementer finds an unexpected test failure after the YAML edit, halt and surface.

### I.5 Cache-key collision v1 vs v2
**Condition**: `test_cache_key_distinct_for_v1_v2` (F.5.4) fails — the v1 and v2 corpora produce the same `CalibrationSpec.cache_key("<tok>")` value.
**Why halt**: a collision would let v2 overwrite or be served from v1's cache file, silently corrupting both.
**Action**: surface the colliding key + the underlying `cache_key` payload diff; do not commit.

### I.6 vLLM accumulator wiring incompatible with TF rows
**Condition** (defensive — should not occur per Section A.3 final note): during Step 6 implementation, the implementer discovers that vLLM's imatrix / REAP / cov / per-expert-max accumulators emit warnings or errors when fed a chunk that contains only TF rows (i.e., empty GENERATE batch).
**Why halt**: silently emitting empty accumulators may corrupt downstream sidecars.
**Action**: surface; user may scope the TF-skip to chunk-level (only run accumulators on chunks with ≥1 GENERATE row).

---

## Section J — Constraints the implementer must obey

1. **No monkey-patching anywhere** — neither in source nor in tests beyond the existing `monkeypatch` usage in test_run_pipeline_reap_exact.py. v2 lives entirely in the existing module; no runtime patching of vLLM, transformers, or any third-party.
2. **No vLLM source-patch needed for this task.** All changes are at the calibration-pipeline level; vLLM's behavior is unchanged.
3. **Atomic writes for disk artifacts**: any new JSONL or sidecar write uses tmp + `os.replace`. The existing build scripts already do this; the new `_synth_teacher_forced_rows` helper must follow the same pattern.
4. **Update docstrings** wherever behavior changes:
   - `_iter_prompts_from_qwen3_pretrain_mix_v2`: new docstring describing 4-tuple return shape and policy plumbing.
   - `_generate_traces`: extend docstring to mention TF skip-generation behavior.
   - `_process_outputs` and `_synth_teacher_forced_rows`: new docstrings describing `completion_source` field.
   - The corpus-name dispatch comment at `_iter_prompts_from_qwen3_pretrain_mix_v2`'s call site in `main()` (HF + vLLM scripts).
   - The "Output schema" section at the top of each build script.
5. **No PR language**. Branch is `feat/calibration-v2`; commit + push + `git merge --ff-only` to `main` when ready (per `feedback_no_pr_language.md`). Sole-author, personal repo.
6. **Tests must pass before each commit**. The implementation-order step list (Section H) is sequenced so each step ends with a green test sweep. No broken builds in commit history.
7. **Small focused diffs**. Each commit ~50-200 LoC max. The review/fix loop will run reviewer→fixer ping-pong (per `feedback_review_fix_loop_protocol.md`) after the implementer; reviewable diffs are mandatory.
8. **No dollar totals** anywhere in code or docs (per `feedback_dont_compute_costs.md`). GPU-hours OK; $X.YZ never.
9. **Standing auth means act** (per `feedback_standing_auth_means_act.md`): once an implementer-step's tests are green, commit and proceed to the next step. Do not pause for confirmation between steps unless a halt trigger fires.
10. **Raise, don't substitute** (per `feedback_raise_dont_substitute.md`): if any planned step is technically infeasible as written, the implementer surfaces the concern and waits — does NOT silently choose a different approach.
11. **Reuse over re-implement**: the 5 new row extractors must reuse `_shuffled_stream`, `_make_subset_seed`, and (for the 3 MoT subsets) `_stream_messages_with_config`. Do not re-implement streaming/seed logic per-subset.
12. **Sole truth lives in `_QWEN3_MIX_V2_*` dicts**: any per-subset behavior (weight, avg_tokens, dataset, config, split, policy) must flow from those dicts, not hardcoded in the iterator/streamer body.

---

## Appendix — Quick verification (cite-and-verify)

| Plan claim | Verified by |
|---|---|
| `qwen3-pretrain-mix` is registered at `calibration.py:1284-1288` | direct file read |
| v1 mix has 8 subsets summing to 1.0 | `_QWEN3_MIX_WEIGHTS` keys/values at `calibration.py:998-1007` |
| `_generate_traces` always generates today (no TF branch) | `build_self_traces_calib.py:592-600` `model.generate(...)` unconditional |
| `_process_outputs` vLLM emits `completion_source`-free rows today | `build_self_traces_calib_vllm.py:422-437` yields dict without that key |
| MoT row schema is `{messages, num_tokens, source}` | `mcp__claude_ai_Hugging_Face__hub_repo_details` 2026-05-27 |
| MoT row's assistant content includes literal `<think>...</think>` | dataset_preview MoT/math/train[0] |
| SWE-smith `messages` column is JSON-encoded string | hub schema preview, dtype=string |
| SWE-smith xml split exists with 26.1K rows | hub dataset_structure |
| Glaive flat `system`/`chat` string schema | hub schema preview, both dtype=string |
| Glaive uses `USER:`/`ASSISTANT:`/`FUNCTION RESPONSE:` flat markers | dataset_preview row[0,1] |
| Glaive open access (not gated) on 2026-05-27 | hub overview, no `🔒 Gated` tag |
| `_complete=true` is the only filter applied by self-traces loader | `calibration.py:1429-1435` |
| `CalibrationSpec.cache_key` includes `source` | `calibration.py:97-98` |
| `_trace_cache_key_vllm` schema_version is currently 8 | `build_self_traces_calib_vllm.py:148` |
| Test `test_cache_key_carries_schema_version_8` exists and asserts 8 | `test_calib_jsonl_schema_v8.py:308-350` |
| Production YAMLs reference `qwen3-pretrain-mix` source | `qwen36_35b_a3b_30pct.yaml:50`, `_reap_exact.yaml:64` |
| Production YAMLs use `sequence_length: 2048` | same files, lines 53 / 67 |

End of plan.
