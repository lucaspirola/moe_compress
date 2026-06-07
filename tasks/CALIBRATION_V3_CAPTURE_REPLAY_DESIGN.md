# Calibration v3 — decoupled capture via forward-only replay

**Status:** design (approved direction 2026-06-07). Supersedes the v2 capture
mechanism. Generation (the v2 mix + the 8000-row corpus) is UNCHANGED.

## Problem (what v3 fixes)

In the v2 driver (`build_self_traces_calib_vllm.py`) the MoE-internal capture
hooks fire **only during token generation**. The mix is a hybrid:

- **GENERATE rows (~52%)** — the teacher autoregressively writes the answer;
  every decode step is a model forward → capture hooks fire → captured.
- **TEACHER_FORCED rows (~48%)** — `_synth_teacher_forced_rows` only renders +
  tokenizes the `(prompt, canonical_answer)` for row metadata. **No model
  forward happens.** → hooks never fire → these rows contribute NOTHING to any
  capture sidecar.

The v2 *design doc* (`tasks/CALIBRATION_MIX_V2_DESIGN.md` §"TEACHER_FORCED",
line ~130) explicitly intended these rows to be forwarded once ("the teacher
just forwards the prompt + answer tokens, ~negligible cost"). The
implementation dropped that forward. **This is an implementation deviation
from the v2 design, not a design flaw.**

### Blast radius (verified against code, 2026-06-07)

The 48% teacher-forced split contains **all code** (mot_code 12% + swe_smith
12/16%) and most science-reasoning (mot_science) — domains that are absent or
thin on the generate side (there is **no** code generation subset). So every
*pre-computed capture sidecar* was built blind to code/science:

| Consumer | Data source | v2 status |
|---|---|---|
| **REAP faithful prune** (`reap_prune.py`) | reap_scores **sidecar**, no live fallback (fails loud) | **BLIND — the real victim** |
| imatrix / wanda / input-cov (Stages 3/4) | capture sidecars | blind (not in the probe) |
| routing_stats / stage2_profile / per_expert_max / reservoirs / block_outputs | capture sidecars | blind |
| **REAM** (`layer_merge.py` `on_profile`) | **live** Stage-2 forward over jsonl text | already domain-complete |
| **Stage 2.5 router-KD heal** | **live** teacher forward over jsonl text | already domain-complete |
| **block_refine** (Stage 3) | **live** teacher forward (falls through when uncached) | already domain-complete on its live path |

**Key takeaway:** consumers that recompute live over the raw text corpus were
never blind. Only consumers that trust a *pre-computed sidecar* were damaged —
chiefly faithful REAP. v3 makes every pre-computed sidecar honest.

## Why not the alternatives

- **100% generate (drop teacher-forcing):** removes the blind spot too, but (a)
  far more generation compute — the long code/reasoning traces (10–22K tokens)
  are exactly the expensive ones; (b) throws away curated high-quality external
  traces (R1/Claude) and the cross-model diversity they add; (c) SWE multi-turn
  agent trajectories have no clean single-prompt generated equivalent. Rejected.
- **In-driver TF-forward (make TF rows prefill inside the generation driver):**
  same mechanism as v3 but re-couples capture to the generation run; can't
  re-capture without re-running generation. Rejected in favor of decoupling.

## Numerical justification (why read-through == generation)

Decoder-only causal attention: a token's hidden state, gate logits, expert
routing and expert outputs depend **only on preceding tokens**. Therefore, for
a *fixed* realized token sequence, the per-token activations are identical
whether produced (a) incrementally during autoregressive generation (KV-cache
of tokens `0..i-1`) or (b) in a single prefill forward over the whole sequence
(token `i` attends to `0..i-1` via the causal mask). KV-cache is a compute
optimization, not a semantic change. Float nondeterminism from different
kernels/batching is ~1e-5 and negligible under aggregation over millions of
tokens.

- **Generated rows:** replaying `(prompt + the model's own generated answer)`
  reproduces the exact activations seen during generation. Identical.
- **Teacher-forced rows:** replaying `(prompt + canonical answer)` measures the
  model *processing real high-quality code/reasoning*. This is the correct
  signal for expert-importance scoring (code tokens route to code experts
  regardless of authorship). It does NOT replicate "the model writing its own
  code" — which the prune decision does not need and which only 100%-generate
  would provide, at higher cost on lower-quality self-written code.

## Design — forward-only capture replay

A new **replay mode** in the calibration driver (exact CLI/flag name decided in
the plan). Behavior:

1. **Input:** an existing self-traces JSONL (the v2 corpus,
   `self_traces_<key>.jsonl`, 8000 rows). No generation; no datasets pulled.
2. **Per row:** build the full input as the v2 driver renders it —
   `apply_chat_template([{user:prompt},{assistant:answer}],
   add_generation_prompt=False, enable_thinking=True)` → tokenize → the
   `prompt_token_ids` for one request. Identical rendering for GENERATE and
   TEACHER_FORCED rows (uniform path).
3. **Forward-only:** submit to vLLM `LLM.generate` with
   `SamplingParams(max_tokens=1, temperature=0, prompt_logprobs=0)` so vLLM runs
   **one prefill** over all `prompt_token_ids`; the single emitted token is
   discarded. The fused-MoE forward runs over every prefill token → the patched
   capture hooks fire over the entire `(prompt+answer)` sequence.
4. **Captures:** the FULL suite (reap_scores, imatrix, input_covariance,
   wanda_scalar_row, stage2_profile, per_expert_max, routing_stats,
   router_logits_stats, layer_input_reservoir, output_reservoir, block_outputs)
   — fixes every signal at once, not just reap.
5. **Sidecar output:** written to the canonical sidecar path of the **input**
   jsonl: `<jsonl>.parent/sidecars/<jsonl.stem>/<signal>.pt`. So a downstream
   config with `calibration.jsonl_path = <that jsonl>` resolves them with no
   extra wiring. (For the probe specifically, the input is the canonical
   `artifacts/_shared/self_traces_489ee0e1b17b43b0.jsonl`.)
6. **Env invariants (B0/C1):** `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process
   EngineCore so callbacks fire) + the C2 `global_num_experts` discovery patch +
   the `Qwen3NextSparseMoeBlock` block hook — all already landed in the
   d20260606 wheel. The replay must set the same capture env the generate path
   sets (`--capture-*` → `VLLM_CALIB_CAPTURE_*`).
7. **Resumable + checkpointed:** reuse the existing per-chunk `.ckpt`
   accumulator checkpointing so a preemption mid-replay resumes.

### Long-sequence handling

Some canonical R1/SWE traces exceed `--max-model-len`. Plan must choose +
justify: (a) raise `max_model_len` to cover the corpus (memory permitting), and
(b) skip-with-count any row still over the limit (log how many, which subsets),
rather than silently truncating — truncation would bias the captured
distribution. A per-row token-length histogram of the 8000 corpus should inform
the cap.

### Correctness gates (must hold before the probe trusts the output)

- reap_scores sidecar loads, `n_layers==40`, `n_experts==256`, non-empty.
- Captured token counts attribute a **non-trivial share to code/science
  subsets** (the whole point) — emit a per-subset captured-token tally and
  assert code+science > 0 with a sane fraction (~the mix weight).
- `assert_enabled_captures_nonempty` (existing fail-fast) passes for every
  enabled signal.
- A spot check: for ≥1 GENERATE row, replay-captured per-layer routing matches
  the generate-time capture within float tolerance (validates read-through ==
  generation empirically, not just by argument).

## Scope decisions (locked unless the plan/review surfaces a reason)

- **Replay the full 8000 corpus** (not a subset) — best for Hessian-like
  signals; one prefill/row is cheap (~30–60 min on H200).
- **Capture the full signal suite** — fix everything once.
- **Decoupled standalone mode** is the canonical v3 capture path going forward;
  the generate-time capture dispatch stays in place (same hook sites the replay
  exercises) but is no longer the primary capture surface. No removal in this
  change — deprecation note only, to keep the diff tight.

## Out of scope

- No change to the v2 mix, weights, or generation.
- No change to any plugin/stage consumer (they read the same sidecar paths).
- No new code path in plugins — this is calibration-driver-only.

## Process

New code → full protocol: plan → plan-review loop (to all-none) → implement →
code-review loop (to all-none), separate implementer and reviewer agents, every
category incl. nitpick. The plan must be derived by reading the **actual**
driver code + the pinned vLLM `SamplingParams`/prefill API (verify forward-only
prefill fires the MoE hooks on the pinned wheel — do not assume).
