# PLAN — Calibration v3 forward-only replay (implementation)

Implements `tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md`. Branch
`feat/calib-v3-replay`. Single modified driver file +1 new test file.

## Resolved design issues (verified against code/docs, not GPU)

- **expert_out_unweighted prefill skip → RESOLVED via chunked prefill.** The
  hook writes into a persistent buffer of `_calib_buf_rows = max_cudagraph_capture_size
  = min(max_num_seqs*2, 512) = 512` rows and SKIPS forward batches with
  `num_tokens > _calib_buf_rows` (patch ~line 1089). vLLM V1 (already active —
  the driver hard-sets `VLLM_ENABLE_V1_MULTIPROCESSING=0`) runs chunked prefill
  by DEFAULT; setting `max_num_batched_tokens=256` bounds every prefill chunk to
  ≤256 tokens, so every MoE forward has `num_tokens ≤ 256 ≤ 512` → the hook
  fires on all prefill tokens (not just the 1 decode token). KV-cache preserved
  across chunks → causal context intact → per-token activations identical.
- **`enable_chunked_prefill` kwarg** — NOT passed; V1 default. (It's a
  SchedulerConfig field, not a top-level `LLM()` kwarg on the pinned wheel.)
- **`_load_teacher_vllm`** already accepts `max_num_batched_tokens` (line ~368)
  + `max_logprobs` and passes through to `LLM()`. No signature change. Replay
  calls it with `max_num_batched_tokens=256, max_logprobs=1`.
- **SamplingParams** for replay: `SamplingParams(temperature=0.0, max_tokens=1,
  seed=args.seed)` — no `logprobs`, no `reasoning_budget`.

## Architecture decision

Add `--replay-from <jsonl>` to the SAME driver
(`max_quality/scripts/build_self_traces_calib_vllm.py`). Reuses all 11 capture
writers' env gates, setup, checkpoint/resume, dump blocks,
`assert_enabled_captures_nonempty`, `_load_teacher_vllm`. The branch enters a
standalone `_run_replay(args)` and returns before the generate path's
cache-key block. Generate path: behaviorally untouched.

## Chunked-prefill mechanism (the crux)

`max_num_batched_tokens=256` → V1 scheduler splits each request prompt into
≤256-token windows, each a separate MoE forward (`num_tokens ≤ 256 ≤
_calib_buf_rows=512`), so `expert_out_unweighted` (→ reap_scores,
per_expert_max, output_reservoir, stage2_profile gated outputs) FIRES on every
token. Size-independent hooks (`router`, `expert_in`, `expert_mid`, `block_out`,
`layer_in`) fire regardless. Full signal suite captured; no wheel rebuild.

## New helpers (pure; after line 766 in the driver)

- `_render_row_for_replay(row, tokenizer, max_model_len) -> tuple[list[int],int]|None`
  — renders `messages=[{user},{assistant}]` via `apply_chat_template(...,
  add_generation_prompt=False, enable_thinking=True)` with `TypeError` fallback
  (mirrors `_synth_teacher_forced_rows` lines 726-733), tokenizes
  (`add_special_tokens=False`), returns `(ids, n)` or `None` if `n >
  max_model_len` (skip, never truncate). Uniform for generated + canonical rows.
- `_build_replay_subset_tally(rows, token_counts) -> {subset:{n_rows,n_tokens}}`.
- `_assert_code_science_nonzero(tally, code_subsets={mot_code,swe_smith},
  science_subsets={mot_science})` — logs per-subset breakdown, asserts code>0
  AND science>0 (the v3 correctness gate).
- `_write_replay_ckpt(path, rows_done)` — atomic row-index checkpoint.

## Refactor: `_WriterState` dataclass + `_setup_all_writers`

Extract the per-writer setup blocks (driver lines ~1659-2115: the
`_check_ckpt_counter` closure + 10 writer setups + `_ckpt_existence_check`
resume guards) into `_setup_all_writers(args, out_path, llm, tokenizer,
already_done) -> _WriterState`. `_WriterState` is a `dataclasses.dataclass`
holding each `*_ckpt_path` (None when disabled) — AttributeError on typo, not
silent None. Called from BOTH `main()` (out_path = output tmp jsonl) and
`_run_replay()` (out_path = input jsonl). In `main()` replace the inline blocks
with `ws = _setup_all_writers(...)` and update downstream refs to `ws.<field>`.
Add `import dataclasses` (stdlib).

## `--replay-from` arg + redirect

Add `--replay-from` to argparse after `--allow-empty-captures` (after line
1313). In `main()`, after the env-gate blocks (after line ~1521), before the
cache-key block: `if args.replay_from is not None: return _run_replay(args)`.

## `_run_replay(args) -> int` (after `main()`)

Steps:
1. Validate `--replay-from` exists; require ≥1 `--capture-*`; `_harden_runtime_env(str(replay_jsonl), dtype)`.
2. Log chunked-prefill advisory; `_REPLAY_MAX_BATCHED_TOKENS = 256`.
3. Load + validate all JSONL rows (each needs `messages` ≥2). Count GENERATE
   (`completion_source=="teacher_generated"`) vs canonical.
4. Pre-flight length scan from stored `n_prompt_tokens + n_gen_tokens`: log
   corpus max + estimated over-`max_model_len` count + a `--max-model-len`
   recommendation. (No tokenization yet.)
5. Resume: read `<jsonl>.replay.ckpt` → `already_done`; slice
   `replay_rows = all_rows[already_done:]`; early-return if empty.
6. `_load_teacher_vllm(..., max_num_batched_tokens=256, max_logprobs=1)`; get tokenizer.
7. `ws = _setup_all_writers(args, out_path=replay_jsonl, llm, tokenizer, already_done)`.
8. `sp_replay = SamplingParams(temperature=0.0, max_tokens=1, seed=args.seed)`.
9. Chunk loop (`args.chunk_size`): render each row (skip+count over-length by
   subset); submit `[{"prompt_token_ids": ids}]` to `llm.generate(...,
   sp_replay)`; discard outputs. After first submitted chunk:
   `assert_enabled_captures_nonempty(...)`. block_outputs close_subset gate.
   All 10 per-writer periodic checkpoints keyed on `already_done + n_replayed`
   (captured-prompt counter; skips excluded). `_write_replay_ckpt(replay_ckpt,
   already_done + n_replayed + n_skipped)` (JSONL-position counter).
10. Skip histogram (+>10% warning). 
11. `_build_replay_subset_tally` + `_assert_code_science_nonzero`; post-run
    `assert_enabled_captures_nonempty`; GENERATE-row spot-check log.
12. All 10 per-writer `dump_*(out_path=replay_jsonl)` → sidecars land at
    `replay_jsonl.parent/sidecars/<stem>/<signal>.pt`; unlink each `.ckpt`.
13. `replay_ckpt.unlink()`; summary log; return 0.

Two distinct counters (both correct): captured = `already_done + n_replayed`
(skips contribute no tokens); JSONL-position = `+ n_skipped`.

## Long-sequence policy

`--max-model-len` is the cap; over-length rows skipped+counted (never
truncated). Pre-flight scan warns early. Full v2 corpus: run `--max-model-len
40960` (R1/SWE traces up to ~22-30K; fits H200 141GB @ 0.90 util with chunked
prefill bounding KV peak).

## Tests — `max_quality/tests/test_replay_helpers.py` (no GPU, no vllm, no monkeypatch)

Stub tokenizer class (plain, not a patch) with `apply_chat_template` +
`__call__`. 12 tests: render happy path / over-length→None / exactly-at-limit /
enable_thinking TypeError fallback / add_generation_prompt==False (recording
stub) / tally basic+domain-fallback+empty / assert pass / fail-no-code /
fail-no-science / custom subsets. `pytest -v` must pass with no GPU.

## GPU smoke (Phase 6) — PRIMARY GATE

Go/no-go for trusting reap: with `max_num_batched_tokens=256` over a
>512-token sequence + `--capture-reap-scores`, assert
`calibration_reap_scores.captured_entry_count() > 512` (≫1) AND no
"expert_out_unweighted capture SKIPPED" warning. If FAIL (n==1 or SKIPPED):
STOP — fallback is a wheel-rebuild hook sub-slice patch (contingent, NOT planned
now). Then: 10-row e2e (sidecars exist, reap_scores.pt shape [40,256] non-NaN);
resume test; full 8000-row run (~20-40min @ chunk_size 200).

## Build sequence

Phase 1 helpers+unit tests → Phase 2 `_setup_all_writers` refactor (re-run
existing `test_calib_ckpt_counter.py` + `test_build_self_traces_calib_vllm_c1.py`)
→ Phase 3 arg+redirect+skeleton → Phase 4 loop+checkpoints → Phase 5
gates+dumps+cleanup → Phase 6 GPU smoke → Phase 7 deprecation comment in TF
synth block.

## Remaining UNVERIFIED (GPU-only; resolved by smoke)

1. `LLM.generate()` accepts `[{"prompt_token_ids": [...]}]` on the pinned wheel
   (documented; fallback = decode to string, lossy for special tokens).
2. `max_num_batched_tokens=256` accepted with no floor forcing it higher (else
   use 512, still ≤ `_calib_buf_rows`).

> Full code bodies for every helper + `_run_replay` + the 12 unit tests are in
> the planner transcript and are reproduced verbatim by the implementer. This
> file is the authoritative spec; the implementer follows it exactly and the
> reviewer audits against it + the actual driver code.
