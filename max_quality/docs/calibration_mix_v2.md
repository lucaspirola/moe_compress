# Calibration mix v2 — `qwen3-pretrain-mix-v2`

**Status**: NEW corpus, sibling to v1 (`qwen3-pretrain-mix`).
**Date**: 2026-05-27
**Branch**: `feat/calibration-v2`

12-subset reasoning-mode hybrid calibration mix optimized for
Qwen3.6-thinking. Combines prompt-only GENERATE policy (the teacher
writes its own `<think>...</think>` response) with TEACHER_FORCED
policy (canonical assistant turn used directly) so the calibration
signal covers both fresh teacher outputs and the highest-quality
reasoning corpora available.

The corpus coexists with v1 in the registry — switching from v1 to v2
is a single YAML edit (`source: qwen3-pretrain-mix → qwen3-pretrain-mix-v2`,
`sequence_length: 2048 → 4096`).

---

## Mix table

| # | Subset key | Weight | HF dataset | Config | Split | Policy | Multi-turn |
|---:|---|---:|---|---|---|---|---|
| 1 | `tulu3`            | 11% | `allenai/tulu-3-sft-mixture`             | —       | train | GENERATE        | no  |
| 2 | `math`             |  9% | `nvidia/OpenMathInstruct-2`              | —       | train | GENERATE        | no  |
| 3 | `qa`               |  5% | `databricks/databricks-dolly-15k`        | —       | train | GENERATE        | no  |
| 4 | `creative`         |  5% | `euclaise/writingprompts`                | —       | train | GENERATE        | no  |
| 5 | `multilingual`     |  8% | `CohereForAI/aya_dataset`                | —       | train | GENERATE        | no  |
| 6 | `fineweb`          |  5% | `HuggingFaceFW/fineweb-edu`              | —       | train | GENERATE        | no  |
| 7 | `papers`           |  5% | `gfissore/arxiv-abstracts-2021`          | —       | train | GENERATE        | no  |
| 8 | `mot_math`         | 12% | `open-r1/Mixture-of-Thoughts`            | math    | train | TEACHER_FORCED  | no  |
| 9 | `mot_code`         | 12% | `open-r1/Mixture-of-Thoughts`            | code    | train | TEACHER_FORCED  | no  |
| 10 | `mot_science`     |  8% | `open-r1/Mixture-of-Thoughts`            | science | train | TEACHER_FORCED  | no  |
| 11 | `swe_smith`       | 12% | `SWE-bench/SWE-smith-trajectories`       | —       | xml   | TEACHER_FORCED  | YES (flattened) |
| 12 | `function_calling` |  8% | `glaiveai/glaive-function-calling-v2`    | —       | train | GENERATE        | no  |

**Sum**: 100%. GENERATE policy: 56% (8 subsets). TEACHER_FORCED policy: 44% (4 subsets).

---

## Policy rationale

### Why GENERATE for non-thinking-format sources

`tulu3`, `math`, `qa`, `creative`, `multilingual`, `fineweb`, `papers`,
and `function_calling` all share one trait: their canonical assistant
turns are NOT in `<think>...</think>` format.

  * `tulu3` mixes 2023-era SFT data (FLAN, OASST, NuminaMath, Evol-Code,
    …). Canonical answers are plain instruct prose.
  * `math` (OpenMathInstruct-2) carries Llama-3.1-405B chain-of-thought
    solutions — long, but not wrapped in `<think>`.
  * `qa` / `creative` / `multilingual` / `papers` use human-written
    canonical answers.
  * `fineweb` is raw web text; not a chat trace at all (we wrap as
    "read this passage").
  * `function_calling` (Glaive) uses GPT-3.5-style JSON in its canonical
    assistant turn — not Qwen3.6's XML `<tool_call>` template.

For all eight, we discard the canonical and let the Qwen3-thinking
teacher generate its own thinking-mode response. The resulting trace
is on-distribution by construction.

### Why TEACHER_FORCED for `mot_math` / `mot_code` / `mot_science`

The three MoT subsets are DeepSeek-R1 distillation traces with native
`<think>...</think>` blocks. R1 and Qwen3.6 are both decoder-only
thinking-mode models trained on similar SFT mixes; their token-level
reasoning style is close enough that teacher-forcing R1 traces gives
meaningful Qwen3.6 routing signal at a fraction of the generation cost.

Wall-clock savings (rough): ~3-4 GPU-hours on H200 vLLM bs≥4 across the
~3000 MoT prompts in an 8000-prompt run.

### Why TEACHER_FORCED for `swe_smith`

The `xml` split of `SWE-bench/SWE-smith-trajectories` carries
Claude-3.7-Sonnet trajectories with assistant turns in
`<function=NAME><parameter=KEY>VALUE</parameter></function>` format —
byte-compatible with Qwen3.6's `<tool_call>` template (minus the
outer `<tool_call>...</tool_call>` wrapping, which the chat template
adds on render).

The cost of this choice: zero `<think>` token supervision on this 12%
of the mix (Claude 3.7 was not in extended-thinking mode when these
trajectories were generated). The gain: authentic tool-call routing
patterns we can't get from the MoT-code subset alone.

### Why not MoT-all (the 350K combined config)

The MoT repo ships an `all` config that interleaves math/code/science
in source-determined proportions. We pick math/code/science separately
so the mix weights are explicit per-domain (12/12/8% vs. whatever
source distribution `all` happens to expose).

---

## Token-budget comparison

| Variant | num_prompts | seq_len | num_sequences | Total calib tokens | Notes |
|---|---:|---:|---:|---:|---|
| REAP paper (arXiv:2510.13999 §5) | n/a | 2048 | 1024 | 2.10M | c4 + evol-codealpaca; demonstrated-sufficient floor for ~30B-class MoE |
| REAM paper (arXiv:2604.04356 §5) | n/a | 512 | 3072 | 1.57M | shorter sequences; for reference only |
| GLM-4.7 recipe (Eastern Island letter) | n/a | 16384 | 24576 | 402.6M | 6 sources × 4096 samples each; calibrated for ~357B model |
| Our v1 (`qwen3-pretrain-mix`) | 6500 | 2048 | 4000 | 8.2M | 8 subsets, all GENERATE; ~4× REAP paper baseline |
| Our v2 (`qwen3-pretrain-mix-v2`) | 8000 | 4096 | 4000 | 16.4M | 12 subsets, hybrid; ~8× REAP paper baseline, ~25× lighter than GLM-4.7 |

The v2 budget is sized for our 35B-A3B target's pipeline depth — Stages 1
through 6 all consume from the same JSONL — without paying the GLM-4.7
357B-class overhead.

---

## How to use

YAML:

```yaml
calibration:
  source: qwen3-pretrain-mix-v2
  seed: 1337
  num_sequences: 4000
  sequence_length: 4096
```

Build self-traces (vLLM, recommended):

```bash
python max_quality/scripts/build_self_traces_calib_vllm.py \
    --teacher Qwen/Qwen3.6-35B-A3B \
    --prompts qwen3-pretrain-mix-v2 \
    --num-prompts 8000 \
    --max-new-tokens 16384 \
    --reasoning-budget 4096 \
    --output artifacts/_shared/self_traces.jsonl
```

Build self-traces (HF transformers fallback):

```bash
python max_quality/scripts/build_self_traces_calib.py \
    --teacher Qwen/Qwen3.6-35B-A3B \
    --prompts qwen3-pretrain-mix-v2 \
    --num-prompts 8000 \
    --max-new-tokens 16384 \
    --output artifacts/_shared/self_traces.jsonl
```

Both scripts honor `--prev-num-prompts N` for incremental extension of a
prior calibration set (the v2 iterator skips the first
`int(N * weight)` rows per subset, yielding only the new slice).

---

## Expected build time

  * ~4160 GENERATE prompts at vLLM bs≥4 — ~5-7 hours on H200 SXM5.
  * ~3840 TEACHER_FORCED prompts — <30 minutes (no generation, just
    JSONL synthesis + a single tokenizer pass per row for the n_prompt_tokens
    metric).
  * Total: ~5-7 hours, dominated by GENERATE generation.

Per `feedback_calibration_oom_ladder.md`, `--load-in-4bit` + bs≥4 is the
practical fit on H200 for non-FP8 Qwen3.6 teachers; FP8-quantized
teacher repos run native at bs≥4 without NF4.

---

## Migration from v1

Switching v1 → v2 is two YAML edits:

  * `source: qwen3-pretrain-mix → qwen3-pretrain-mix-v2`
  * `sequence_length: 2048 → 4096`

Both v1 and v2 corpora coexist in the registry. Existing `qwen3-pretrain-mix`
self-traces JSONLs on disk are NOT invalidated — they remain valid for
any pipeline run still pointing at v1. v2 produces a NEW JSONL under a
different cache_key, so the two coexist on disk.

`CalibrationSpec.cache_key` folds `source` into its payload, so calibration
tensor caches keyed off v1 and v2 are partitioned — no on-disk collision.

---

## Schema changes (build-script JSONLs)

Both build scripts now emit one extra field per row:

  * `completion_source`: `"teacher_generated"` (vLLM/HF `model.generate`)
    or `"canonical"` (v2 TEACHER_FORCED rows, synthesized from the source
    dataset's canonical assistant turn).

Cache-key `schema_version` bumps:

  * HF (`build_self_traces_calib.py`): 6 → 7.
  * vLLM (`build_self_traces_calib_vllm.py`): 8 → 9.

Existing v6/v8 JSONLs are NOT cache-hit by v7/v9 runs — the bumps
intentionally segregate so a v1-era cache doesn't silently mix into a
v2 calibration tensor.

---

## Known limitations / footnotes

1. **function_calling format gap**: Glaive's `<functioncall>` JSON is
   not Qwen3.6's `<tool_call>` XML format. We use Glaive's prompts
   (system schema + first USER turn, combined into a single user
   message per plan §B.6 Option (i)) and discard its canonical, letting
   the teacher generate Qwen3.6-native `<tool_call>` responses. This
   means we get on-distribution routing for the system+user turns but
   the assistant-turn coverage is bounded by the teacher's actual
   tool-call propensity.

2. **swe_smith multi-turn flatten**: only the first (user, assistant)
   pair is kept; ~80% of each trajectory's tokens are discarded
   (subsequent tool observations + assistant turns dropped). Future
   work could extend the JSONL schema to support multi-turn assistant
   traces.

3. **swe_smith lacks `<think>` blocks**: Claude 3.7 was not in
   extended-thinking mode when these trajectories were generated. This
   12% of the mix contributes tool-call routing supervision but zero
   `<think>`-token supervision.

4. **mot_science source mix**: the science split is partly
   Llama-Nemotron-Post-Training, not pure DeepSeek-R1. Treat as the
   weakest of the three MoT subsets for the R1-alignment assumption.

5. **No GPU validation gating in this commit**: the design doc §7.1
   proposed a 100-prompt KL test before commit. The campaign brief
   defers that; if Stage 6 metrics drop dramatically after the first
   v2 calibration run, the fallback is to flip TF → GENERATE for MoT
   subsets one at a time and re-measure.

---

## Cross-references

  * Plan doc: `tasks/CALIBRATION_MIX_V2_PLAN.md`
  * Design doc: `tasks/CALIBRATION_MIX_V2_DESIGN.md`
  * Corpus adapter: `max_quality/src/moe_compress/utils/calibration.py`
    (look for `_QWEN3_MIX_V2_*` dicts and
    `_stream_texts_qwen3_pretrain_mix_v2`).
  * Build scripts: `max_quality/scripts/build_self_traces_calib.py`
    (HF) and `max_quality/scripts/build_self_traces_calib_vllm.py`
    (vLLM).
  * Tests: `max_quality/tests/test_qwen3_pretrain_mix_v2.py` (corpus
    constants + registry + iterator behavior) and
    `max_quality/tests/test_calib_jsonl_schema_v8.py` (v9 schema +
    completion_source per-row contracts).
