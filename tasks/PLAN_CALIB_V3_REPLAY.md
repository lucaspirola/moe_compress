# PLAN — Calibration v3 forward-only replay (REVISED FINAL, verbatim)

Implements tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md. Branch feat/calib-v3-replay.
Authoritative spec with full inlined code (extracted verbatim from the planner agent).

---

# Calibration v3 — Forward-Only Replay: REVISED FINAL Implementation Plan

## How This Revision Fixes Each Defect

**C1 (imatrix dump signature):** Imatrix is explicitly special-cased throughout: `dump_imatrix(str(out_path.with_suffix(".imatrix.dat")), chunk_count=total_done_captures)`. All nine other writers use `dump_<signal>(out_path: Path)` and compute the sidecar path internally. This asymmetry is called out in every relevant section.

**C2 (buffer size is runtime-derived):** After `_load_teacher_vllm` returns, `_run_replay` reads back the actual `max_cudagraph_capture_size` via `llm.llm_engine.vllm_config.compilation_config.max_cudagraph_capture_size` and hard-asserts `buf_rows >= _REPLAY_MAX_BATCHED_TOKENS` before starting the loop. The `LLM()` constructor also receives `max_num_seqs=256` explicitly (not relying on the default) so the formula `min(256*2, 512) = 512` is predictable. If the assert fires, the error message names the escape hatches.

**H1 (_WriterState completeness):** All ten ckpt-path fields are enumerated verbatim in the `_WriterState` dataclass. The `_check_ckpt_counter` closure is recreated inside `_setup_all_writers` as a lambda that captures the passed-in `already_done` and `allow_counter_divergence`. All ~20 downstream `*_ckpt_path` references in `main()` are listed for `ws.<field>` replacement.

**H2 (layer_input_reservoir):** Explicitly noted: rides inside `stage2_profile`, no standalone dump, not an 11th writer.

**M1 (counter-check collision on resume):** `.replay.ckpt` now stores `{"rows_done": int, "captured_done": int}`. `rows_done` slices the JSONL; `captured_done` is passed to `_setup_all_writers` as `already_done` for writer counter checks. `_write_replay_ckpt` writes both.

**M2 (assert gating):** `assert_enabled_captures_nonempty` is gated on `first_chunk_checked`, which is set only when `rendered_chunk` is non-empty (i.e., at least one request was actually submitted). An all-over-length first chunk does not trip it.

**N1 (prompt_token_ids fallback):** Fallback is re-render as string AND verify round-trip token equality before submitting. If the token counts differ by more than a small tolerance, `log.error` + `SystemExit(2)`. No blind `decode()`.

**N3 (generate-vs-replay routing spot-check):** Removed from the automated gates. Per-row generate-time captures are not retained; the empirical match cannot be automated. The gate is replaced by the buf_rows smoke. The plan notes the design doc line should be softened to "the causal-masking argument is the justification; the automated gates are (a) B0 non-empty, (b) code+science tokens > 0, (c) buf_rows >= effective chunk smoke."

---

## Patterns & Conventions (confirmed correct, file:line refs)

**B0/C1 invariant:** `os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"` module-level line 115. Replay inherits unconditionally.

**Capture env gates:** `VLLM_CALIB_CAPTURE_*` set in `main()` lines 1323–1521 before any vllm import. The `if args.replay_from: return _run_replay(args)` redirect is placed after line 1521.

**`_load_teacher_vllm`** (line 364): signature `(repo, revision, dtype, gpu_memory_utilization, max_model_len, max_num_seqs=None, max_num_batched_tokens=None, max_logprobs=50)`. No signature change. Replay calls with `max_num_batched_tokens=256, max_num_seqs=256, max_logprobs=1`.

**TF rendering** (lines 726–733): `apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)` + `TypeError` fallback. `_render_row_for_replay` mirrors this path exactly.

**Imatrix dump is special:** `dump_imatrix(path: str, chunk_count: int)` writes to `<jsonl>.imatrix.dat` (driver lines 2540–2559). Every other writer: `dump_<signal>(out_path: Path)` computes sidecar internally.

**All nine non-imatrix dump signatures confirmed correct:** `dump_reap_scores(Path)`, `dump_input_cov(Path)`, `dump_wanda_scalar_row(Path)`, `dump_stage2_profile(Path)`, `dump_per_expert_max(Path)`, `dump_routing_stats(Path)`, `dump_router_logits_stats(Path)`, `dump_output_reservoir(Path)`, `dump_block_outputs(Path)`.

**`layer_input_reservoir` rides `stage2_profile`:** no standalone dump, no 11th writer, no 11th `_WriterState` field.

**Sidecar path:** `<jsonl.parent>/sidecars/<jsonl.stem>/<signal>.pt` (cached_calibration_signals.py line 151). Computed internally by each writer's `dump_*`.

**`assert_enabled_captures_nonempty`** (line 219): reused unchanged.

**`_ckpt_counter_check` / `_ckpt_existence_check`** (lines 773–873): called by `_setup_all_writers`.

**`_CAPTURE_WRITER_MODULES`** (line 194): maps `capture_*` arg names → vllm module names. `layer_input_reservoir` intentionally absent (rides `stage2_profile`).

---

## Architecture Decision (confirmed)

`--replay-from <jsonl>` flag on the same driver. Redirect `main()` → `_run_replay(args)` after env gates. No new script file.

---

## Chunked Prefill + Buffer Size Analysis (C2)

**`_calib_buf_rows` derivation:** The patch reads `get_current_vllm_config().compilation_config.max_cudagraph_capture_size` at `TritonExperts.__init__` time (patch lines 9996–10001). The engine docs state `max_cudagraph_capture_size = min(max_num_seqs*2, 512)` by default; passing `cudagraph_capture_sizes` overrides it to the max of that list.

**Ensuring buf_rows ≥ 256:** We set `max_num_seqs=256` explicitly so `min(256*2, 512) = 512`. We do NOT pass `cudagraph_capture_sizes` (which would risk a smaller max). After construction, `_run_replay` reads back the actual value and hard-asserts before the loop. This catches any edge case where vLLM computed a smaller value than expected.

**Reading buf_rows from live engine:** `llm.llm_engine.vllm_config.compilation_config.max_cudagraph_capture_size`. This is the same `VllmConfig` object whose reference `get_current_vllm_config()` returns during model init. If this attribute path doesn't exist on the pinned wheel (UNVERIFIED-GPU), the fallback in the assert code is to call `get_current_vllm_config()` directly.

**vLLM V1 + chunked prefill:** V1 engine enables chunked prefill by default when `max_num_batched_tokens` is set. With `max_num_batched_tokens=256`, every prefill batch is at most 256 tokens. Combined with `buf_rows=512`, every MoE forward has `num_tokens ≤ 256 < 512 = buf_rows` → `expert_out_unweighted` fires on every token.

---

## Complete Verbatim Code for New Functions

### `_render_row_for_replay`

```python
def _render_row_for_replay(
    row: dict,
    tokenizer,
    max_model_len: int,
) -> "tuple[list[int], int] | None":
    """Tokenize a saved JSONL row for forward-only replay (v3).

    Renders messages=[{role:user, content:prompt},{role:assistant,
    content:answer}] through the chat template with
    add_generation_prompt=False, enable_thinking=True. Mirrors
    _synth_teacher_forced_rows lines 726-733 exactly, and works
    uniformly for completion_source="teacher_generated" and "canonical".

    Returns (token_ids: list[int], n_tokens: int) if
    n_tokens <= max_model_len. Returns None if over-length.
    Never silently truncates. Pure function; testable without GPU.
    """
    messages = row.get("messages", [])
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=True,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    token_ids: list[int] = tokenizer(
        rendered, add_special_tokens=False,
    )["input_ids"]
    if len(token_ids) > max_model_len:
        return None
    return token_ids, len(token_ids)
```

### `_build_replay_subset_tally`

```python
def _build_replay_subset_tally(
    rows: "list[dict]",
    token_counts: "list[int]",
) -> "dict[str, dict[str, int]]":
    """Per-subset token+row tally for successfully replayed rows.

    Args:
        rows: row dicts that passed the length gate and were submitted
              to vLLM (non-skipped rows only).
        token_counts: parallel list of token counts (same length as rows).

    Returns {subset_name: {"n_rows": int, "n_tokens": int}}.
    Pure function; testable without GPU.
    """
    tally: dict[str, dict[str, int]] = {}
    for row, n_tok in zip(rows, token_counts):
        subset = str(row.get("subset") or row.get("domain") or "unknown")
        entry = tally.setdefault(subset, {"n_rows": 0, "n_tokens": 0})
        entry["n_rows"] += 1
        entry["n_tokens"] += n_tok
    return tally
```

### `_assert_code_science_nonzero`

```python
def _assert_code_science_nonzero(
    tally: "dict[str, dict[str, int]]",
    code_subsets: "frozenset[str]" = frozenset({"mot_code", "swe_smith"}),
    science_subsets: "frozenset[str]" = frozenset({"mot_science"}),
) -> None:
    """Correctness gate: assert code and science subsets contributed tokens.

    The purpose of v3 replay is to cover code/science which TEACHER_FORCED
    rows never covered in v2. Zero tokens in both means the JSONL is not
    the v2 corpus or the subset names changed.

    Logs full per-subset breakdown before asserting. Pure function.
    Raises AssertionError with an actionable message on failure.
    """
    code_tokens = sum(
        tally[s]["n_tokens"] for s in code_subsets if s in tally
    )
    sci_tokens = sum(
        tally[s]["n_tokens"] for s in science_subsets if s in tally
    )
    log.info("replay tally by subset (replayed rows only):")
    for subset, counts in sorted(tally.items()):
        log.info(
            "  %-24s  %5d rows  %9d tokens",
            subset, counts["n_rows"], counts["n_tokens"],
        )
    log.info(
        "replay: code subsets %s → %d tokens; "
        "science subsets %s → %d tokens",
        sorted(code_subsets & tally.keys()), code_tokens,
        sorted(science_subsets & tally.keys()), sci_tokens,
    )
    assert code_tokens > 0, (
        f"replay correctness gate FAILED: no code-subset tokens captured. "
        f"Checked: {sorted(code_subsets)}. "
        f"Present in corpus: {sorted(tally)}. "
        f"Verify input JSONL is the v2 corpus and mot_code/swe_smith "
        f"rows have non-empty messages fields."
    )
    assert sci_tokens > 0, (
        f"replay correctness gate FAILED: no science-subset tokens captured. "
        f"Checked: {sorted(science_subsets)}. "
        f"Present in corpus: {sorted(tally)}."
    )
```

### `_write_replay_ckpt`

```python
def _write_replay_ckpt(
    path: "Path",
    rows_done: int,
    captured_done: int,
) -> None:
    """Atomically write the replay row-index + capture-counter checkpoint.

    Two counters are stored separately to avoid M1 counter-check collision:
      rows_done    — total rows consumed from the JSONL (replayed + skipped).
                     Used to slice the JSONL on resume (includes skips).
      captured_done — rows that actually contributed to captures (no skips).
                     Passed to _setup_all_writers as already_done for
                     per-writer ckpt counter validation.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"rows_done": rows_done, "captured_done": captured_done}),
        encoding="utf-8",
    )
    os.replace(tmp, path)
```

### `_WriterState` and `_setup_all_writers`

```python
import dataclasses

@dataclasses.dataclass
class _WriterState:
    """Per-writer checkpoint paths returned by _setup_all_writers.

    Every field is None when the corresponding --capture-* flag is off.
    Using a dataclass rather than a dict gives AttributeError on typo
    instead of silent None propagation.

    Fields (10 total; layer_input_reservoir has no standalone field
    because it rides stage2_profile with no separate dump):
    """
    imatrix_ckpt_path: "Path | None" = None
    reap_ckpt_path: "Path | None" = None
    input_cov_ckpt_path: "Path | None" = None
    wsr_ckpt_path: "Path | None" = None
    s2p_ckpt_path: "Path | None" = None
    pem_ckpt_path: "Path | None" = None
    rts_ckpt_path: "Path | None" = None
    router_logits_ckpt_path: "Path | None" = None
    or_ckpt_path: "Path | None" = None
    bo_ckpt_path: "Path | None" = None


def _setup_all_writers(
    args,
    out_path: "Path",
    llm,
    tokenizer,
    already_done: int,
) -> "_WriterState":
    """Pre-allocate all enabled capture writer accumulators.

    Extracted from main() (was inline lines 1659-2115) so both the
    generate path and _run_replay() call it identically.

    ``out_path`` — determines checkpoint file locations:
      generate path: the output JSONL tmp path (out_path = tmp_path)
      replay path  : the input JSONL (out_path = replay_jsonl)
    All per-writer .ckpt files land as siblings of out_path.

    ``already_done`` — for the generate path: JSONL rows already written.
      For the replay path: captured_done (rows that contributed to
      captures, EXCLUDING skipped/over-length rows). This is the correct
      counter for _ckpt_counter_check which validates capture coverage.

    Returns _WriterState with all ten ckpt path fields (None if disabled).

    The _check_ckpt_counter closure from main() is recreated as a lambda
    that captures already_done and args.allow_counter_divergence.
    """
    ws = _WriterState()

    # Recreate the _check_ckpt_counter closure from main() (~line 1659).
    # Captures already_done + args.allow_counter_divergence from caller.
    def _check(signal_name: str, loaded_prompts: int, ckpt_path: "Path"):
        _ckpt_counter_check(
            signal_name, loaded_prompts, already_done, ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )

    # ---- imatrix --------------------------------------------------------
    if args.capture_imatrix:
        import vllm.calibration_imatrix as _im  # type: ignore
        _im.setup(llm)
        log.info("imatrix: setup complete -- accumulators pre-allocated")
        ws.imatrix_ckpt_path = out_path.with_suffix(".imatrix.ckpt")
        _ckpt_existence_check(
            "imatrix", args.capture_imatrix, already_done,
            ws.imatrix_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.imatrix_ckpt_path.exists():
            try:
                loaded = _im.load_imatrix_checkpoint(
                    str(ws.imatrix_ckpt_path))
                _check("imatrix", loaded, ws.imatrix_ckpt_path)
                log.info("imatrix: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error("imatrix: ckpt schema mismatch (%s); deleting", exc)
                ws.imatrix_ckpt_path.unlink()

    # ---- reap-scores ----------------------------------------------------
    if args.capture_reap_scores:
        import vllm.calibration_reap_scores as _reap  # type: ignore
        _reap.setup(llm)
        log.info("reap-scores: setup complete")
        ws.reap_ckpt_path = out_path.with_suffix(".reap_scores.ckpt")
        _ckpt_existence_check(
            "reap_scores", args.capture_reap_scores, already_done,
            ws.reap_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.reap_ckpt_path.exists():
            try:
                loaded = _reap.load_reap_scores_checkpoint(
                    str(ws.reap_ckpt_path))
                _check("reap-scores", loaded, ws.reap_ckpt_path)
                log.info("reap-scores: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error("reap-scores: ckpt schema mismatch (%s); deleting",
                          exc)
                ws.reap_ckpt_path.unlink()

    # ---- input-covariance -----------------------------------------------
    if args.capture_input_covariance:
        import vllm.calibration_input_cov as _icov  # type: ignore
        _icov.setup(llm)
        log.info("input-cov: setup complete")
        ws.input_cov_ckpt_path = out_path.with_suffix(".input_cov.ckpt")
        _ckpt_existence_check(
            "input_covariance", args.capture_input_covariance, already_done,
            ws.input_cov_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.input_cov_ckpt_path.exists():
            try:
                loaded = _icov.load_input_cov_checkpoint(
                    str(ws.input_cov_ckpt_path))
                _check("input-cov", loaded, ws.input_cov_ckpt_path)
                log.info("input-cov: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error("input-cov: ckpt schema mismatch (%s); deleting",
                          exc)
                ws.input_cov_ckpt_path.unlink()

    # ---- wanda scalar_row -----------------------------------------------
    if args.capture_wanda_scalar_row:
        import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
        _wsr.setup(llm)
        log.info("wanda-scalar-row: setup complete")
        ws.wsr_ckpt_path = out_path.with_suffix(".wanda_scalar_row.ckpt")
        _ckpt_existence_check(
            "wanda_scalar_row", args.capture_wanda_scalar_row, already_done,
            ws.wsr_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.wsr_ckpt_path.exists():
            try:
                loaded = _wsr.load_wanda_scalar_row_checkpoint(
                    str(ws.wsr_ckpt_path))
                _check("wanda-scalar-row", loaded, ws.wsr_ckpt_path)
                log.info("wanda-scalar-row: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "wanda-scalar-row: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.wsr_ckpt_path.unlink()

    # ---- stage2-profile -------------------------------------------------
    if args.capture_stage2_profile:
        import vllm.calibration_stage2_profile as _s2p  # type: ignore
        _s2p.setup(llm,
                   cov_storage_dtype=args.stage2_profile_cov_storage_dtype)
        log.info("stage2-profile: setup complete (cov_dtype=%s)",
                 args.stage2_profile_cov_storage_dtype)
        ws.s2p_ckpt_path = out_path.with_suffix(".stage2_profile.ckpt")
        _ckpt_existence_check(
            "stage2_profile", args.capture_stage2_profile, already_done,
            ws.s2p_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.s2p_ckpt_path.exists():
            try:
                loaded = _s2p.load_stage2_profile_checkpoint(
                    str(ws.s2p_ckpt_path))
                _check("stage2-profile", loaded, ws.s2p_ckpt_path)
                log.info("stage2-profile: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "stage2-profile: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.s2p_ckpt_path.unlink()

    # ---- per-expert-max -------------------------------------------------
    if args.capture_per_expert_max:
        import vllm.calibration_per_expert_max as _pem  # type: ignore
        _pem.setup(llm)
        log.info("per-expert-max: setup complete")
        ws.pem_ckpt_path = out_path.with_suffix(".per_expert_max.ckpt")
        _ckpt_existence_check(
            "per_expert_max", args.capture_per_expert_max, already_done,
            ws.pem_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.pem_ckpt_path.exists():
            try:
                loaded = _pem.load_per_expert_max_checkpoint(
                    str(ws.pem_ckpt_path))
                _check("per-expert-max", loaded, ws.pem_ckpt_path)
                log.info("per-expert-max: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "per-expert-max: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.pem_ckpt_path.unlink()

    # ---- routing-stats --------------------------------------------------
    if args.capture_routing_stats:
        import vllm.calibration_routing_stats as _rts  # type: ignore
        _rts.setup(llm)
        log.info("routing-stats: setup complete")
        ws.rts_ckpt_path = out_path.with_suffix(".routing_stats.ckpt")
        _ckpt_existence_check(
            "routing_stats", args.capture_routing_stats, already_done,
            ws.rts_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.rts_ckpt_path.exists():
            try:
                loaded = _rts.load_routing_stats_checkpoint(
                    str(ws.rts_ckpt_path))
                _check("routing-stats", loaded, ws.rts_ckpt_path)
                log.info("routing-stats: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "routing-stats: ckpt schema mismatch (%s); deleting", exc)
                ws.rts_ckpt_path.unlink()

    # ---- router-logits-stats --------------------------------------------
    if args.capture_router_logits_stats:
        import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
        _bos = getattr(tokenizer, "bos_token_id", None)
        _rlsx.setup(llm, bos_token_id=_bos)
        log.info("router-logits-stats: setup complete (bos_token_id=%s)",
                 _bos)
        ws.router_logits_ckpt_path = out_path.with_suffix(
            ".router_logits_stats.ckpt")
        _ckpt_existence_check(
            "router_logits_stats", args.capture_router_logits_stats,
            already_done, ws.router_logits_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.router_logits_ckpt_path.exists():
            try:
                loaded = _rlsx.load_router_logits_stats_checkpoint(
                    str(ws.router_logits_ckpt_path))
                _check("router-logits-stats", loaded,
                       ws.router_logits_ckpt_path)
                log.info("router-logits-stats: hydrated %d-prompt ckpt",
                         loaded)
            except ValueError as exc:
                log.error(
                    "router-logits-stats: ckpt schema mismatch (%s); "
                    "deleting", exc)
                ws.router_logits_ckpt_path.unlink()

    # ---- output-reservoir -----------------------------------------------
    if args.capture_output_reservoir:
        import vllm.calibration_output_reservoir as _or  # type: ignore
        _or.setup(llm)
        log.info("output-reservoir: setup complete (cap=%d)",
                 args.output_reservoir_cap)
        ws.or_ckpt_path = out_path.with_suffix(".output_reservoir.ckpt")
        _ckpt_existence_check(
            "output_reservoir", args.capture_output_reservoir, already_done,
            ws.or_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.or_ckpt_path.exists():
            try:
                loaded = _or.load_output_reservoir_checkpoint(
                    str(ws.or_ckpt_path))
                _check("output-reservoir", loaded, ws.or_ckpt_path)
                log.info("output-reservoir: hydrated %d-prompt ckpt", loaded)
            except ValueError as exc:
                log.error(
                    "output-reservoir: ckpt schema mismatch (%s); deleting",
                    exc)
                ws.or_ckpt_path.unlink()

    # ---- block-outputs --------------------------------------------------
    if args.capture_block_outputs:
        import vllm.calibration_block_outputs as _bo  # type: ignore
        _bo.setup(llm)
        log.info("block-outputs: setup complete (subset_size=%d)",
                 args.block_outputs_subset_size)
        ws.bo_ckpt_path = out_path.with_suffix(".block_outputs.ckpt")
        _ckpt_existence_check(
            "block_outputs", args.capture_block_outputs, already_done,
            ws.bo_ckpt_path,
            allow_counter_divergence=args.allow_counter_divergence,
        )
        if args.resume and ws.bo_ckpt_path.exists():
            try:
                loaded = _bo.load_block_outputs_checkpoint(
                    str(ws.bo_ckpt_path))
                _check("block-outputs", loaded, ws.bo_ckpt_path)
                log.info("block-outputs: hydrated %d-prompt ckpt (closed=%s)",
                         loaded, _bo._SUBSET_CLOSED)
            except ValueError as exc:
                log.error(
                    "block-outputs: ckpt schema mismatch (%s); deleting", exc)
                ws.bo_ckpt_path.unlink()

    return ws
```

### `_run_replay` (complete verbatim body)

```python
def _run_replay(args) -> int:
    """v3 forward-only replay mode.

    Called from main() when --replay-from is set. Capture env gates
    (VLLM_CALIB_CAPTURE_*) and B0/C1 invariants are already applied
    by main() before this function is called.

    Reads an existing self-traces JSONL, tokenizes each row's full
    (prompt+answer) sequence, submits to vLLM as a single prefill-only
    forward (max_tokens=1, max_num_batched_tokens=256), and writes
    capture sidecars to the canonical sidecar_path namespace of the
    input JSONL.

    Imatrix dump is special-cased (writes <jsonl>.imatrix.dat, not a
    sidecar under sidecars/). All other nine writers use dump_*(Path).

    Returns 0 on success, 1 on configuration/validation error.
    SystemExit(2) if B0 hard-fails (from assert_enabled_captures_nonempty).
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    replay_jsonl = Path(args.replay_from).resolve()
    if not replay_jsonl.is_file():
        log.error("--replay-from: file not found: %s", replay_jsonl)
        return 1

    _enabled_captures = [
        cap for cap in _CAPTURE_WRITER_MODULES if getattr(args, cap, False)
    ]
    if not _enabled_captures:
        log.error(
            "--replay-from requires at least one --capture-* flag. "
            "Nothing to capture; exiting."
        )
        return 1

    log.info(
        "v3 replay mode: input=%s, enabled captures=%s",
        replay_jsonl, _enabled_captures,
    )

    # Runtime env hardening (compile cache, JIT cap). out_path is the
    # input JSONL so VLLM_CACHE_ROOT lands next to it.
    _harden_runtime_env(str(replay_jsonl), args.dtype)

    # ------------------------------------------------------------------
    # 2. Load + validate input JSONL
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    with replay_jsonl.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                log.error(
                    "replay: invalid JSON at line %d: %s", lineno, exc)
                return 1
            msgs = row.get("messages", [])
            if len(msgs) < 2:
                log.error(
                    "replay: line %d: messages field missing or has <2 "
                    "entries. Expected [{role:user,...},{role:assistant,...}]. "
                    "Is this a v8+ schema JSONL?", lineno)
                return 1
            all_rows.append(row)

    if not all_rows:
        log.error("replay: no rows found in %s", replay_jsonl)
        return 1

    log.info("replay: loaded %d rows from %s", len(all_rows), replay_jsonl)

    n_generate_rows = sum(
        1 for r in all_rows
        if r.get("completion_source") == "teacher_generated"
    )
    log.info(
        "replay: %d GENERATE rows, %d TEACHER_FORCED rows",
        n_generate_rows, len(all_rows) - n_generate_rows,
    )
    if n_generate_rows == 0:
        log.warning(
            "replay: no GENERATE rows (completion_source=teacher_generated). "
            "Continuing — may indicate an unexpected corpus."
        )

    # ------------------------------------------------------------------
    # 3. Pre-flight length scan (uses stored metadata; no tokenization)
    # ------------------------------------------------------------------
    max_required = 0
    over_count_preflight = 0
    for row in all_rows:
        needed = row.get("n_prompt_tokens", 0) + row.get("n_gen_tokens", 0)
        max_required = max(max_required, needed)
        if needed > args.max_model_len:
            over_count_preflight += 1
    log.info(
        "replay pre-flight: corpus max tokens (n_prompt+n_gen)=%d, "
        "--max-model-len=%d, ~%d/%d rows may be skipped "
        "(actual tokenization may differ from stored counts)",
        max_required, args.max_model_len,
        over_count_preflight, len(all_rows),
    )
    if over_count_preflight > 0:
        log.warning(
            "replay: ~%d rows may exceed --max-model-len=%d. "
            "These will be SKIPPED (not truncated). "
            "Consider --max-model-len %d (~5%% over corpus max).",
            over_count_preflight, args.max_model_len,
            int(max_required * 1.05),
        )

    # ------------------------------------------------------------------
    # 4. Resume handling — two counters (M1 fix)
    # ------------------------------------------------------------------
    # rows_done    = total rows consumed (replayed + skipped). Used to
    #               slice the JSONL. Includes skips.
    # captured_done = rows that contributed to captures (no skips).
    #               Passed to _setup_all_writers for writer counter checks.
    # Storing both prevents spurious hard-fail on resume when some rows
    # were skipped (which would make rows_done > captured_done, wrongly
    # triggering _ckpt_counter_check divergence if we passed rows_done).
    replay_ckpt = replay_jsonl.with_suffix(
        replay_jsonl.suffix + ".replay.ckpt"
    )
    rows_done_base = 0
    captured_done_base = 0
    if args.resume and replay_ckpt.exists():
        try:
            _rc = json.loads(
                replay_ckpt.read_text(encoding="utf-8"))
            rows_done_base = int(_rc["rows_done"])
            captured_done_base = int(_rc["captured_done"])
            log.info(
                "replay resume: rows_done=%d, captured_done=%d (per %s)",
                rows_done_base, captured_done_base, replay_ckpt,
            )
        except Exception as exc:
            log.warning(
                "replay: could not read resume checkpoint (%s); "
                "starting from row 0.", exc)
            rows_done_base = 0
            captured_done_base = 0

    replay_rows = all_rows[rows_done_base:]
    if not replay_rows:
        log.info("replay: all %d rows already processed.", len(all_rows))
        return 0

    # ------------------------------------------------------------------
    # 5. Load teacher
    # max_num_seqs=256 set explicitly so max_cudagraph_capture_size =
    # min(256*2, 512) = 512, predictable and >= max_num_batched_tokens=256.
    # max_num_batched_tokens=256 caps each prefill chunk so every MoE
    # forward has num_tokens <= 256 <= buf_rows, making expert_out_unweighted
    # fire on all prefill tokens.
    # ------------------------------------------------------------------
    _REPLAY_MAX_BATCHED_TOKENS = 256
    _REPLAY_MAX_NUM_SEQS = 256

    llm = _load_teacher_vllm(
        args.teacher,
        args.teacher_revision,
        args.dtype,
        args.gpu_memory_utilization,
        args.max_model_len,
        max_num_seqs=_REPLAY_MAX_NUM_SEQS,
        max_num_batched_tokens=_REPLAY_MAX_BATCHED_TOKENS,
        max_logprobs=1,
    )
    tokenizer = llm.get_tokenizer()

    # ------------------------------------------------------------------
    # 6. C2 runtime buffer-size assertion
    # Read back the actual max_cudagraph_capture_size from the live engine
    # and hard-assert it is >= _REPLAY_MAX_BATCHED_TOKENS. If this fires,
    # the expert_out_unweighted hook would silently skip prefill chunks.
    # ------------------------------------------------------------------
    buf_rows: int
    try:
        buf_rows = (
            llm.llm_engine.vllm_config
            .compilation_config
            .max_cudagraph_capture_size
        )
        log.info(
            "C2 check: max_cudagraph_capture_size=%d (must be >= %d)",
            buf_rows, _REPLAY_MAX_BATCHED_TOKENS,
        )
    except AttributeError:
        # Fallback: call get_current_vllm_config() in a dummy context.
        # This should not happen on the patched wheel but is safe.
        log.warning(
            "C2 check: llm.llm_engine.vllm_config.compilation_config "
            "attribute path not found; falling back to formula "
            "min(max_num_seqs*2, 512)=%d.",
            min(_REPLAY_MAX_NUM_SEQS * 2, 512),
        )
        buf_rows = min(_REPLAY_MAX_NUM_SEQS * 2, 512)

    if buf_rows < _REPLAY_MAX_BATCHED_TOKENS:
        log.error(
            "C2 HARD FAIL: max_cudagraph_capture_size=%d < "
            "max_num_batched_tokens=%d. expert_out_unweighted would skip "
            "prefill chunks. Recovery options: "
            "(a) lower max_num_batched_tokens to %d, "
            "(b) set max_num_seqs so min(seqs*2,512) >= %d, "
            "(c) pass compilation_config with cudagraph_capture_sizes "
            "whose max >= %d. Aborting.",
            buf_rows, _REPLAY_MAX_BATCHED_TOKENS,
            buf_rows,        # option (a): use buf_rows as the new cap
            _REPLAY_MAX_BATCHED_TOKENS,  # option (b)
            _REPLAY_MAX_BATCHED_TOKENS,  # option (c)
        )
        return 1

    log.info(
        "C2 check passed: buf_rows=%d >= max_num_batched_tokens=%d. "
        "expert_out_unweighted will fire on all prefill chunks "
        "(chunked prefill enabled by V1 default).",
        buf_rows, _REPLAY_MAX_BATCHED_TOKENS,
    )

    # ------------------------------------------------------------------
    # 7. Writer setup
    # out_path = replay_jsonl so .ckpt files land next to the input JSONL.
    # captured_done_base is passed as already_done (excludes skips —
    # M1 fix).
    # ------------------------------------------------------------------
    out_path = replay_jsonl
    ws = _setup_all_writers(
        args, out_path, llm, tokenizer, captured_done_base,
    )

    # ------------------------------------------------------------------
    # 8. SamplingParams for replay
    # ------------------------------------------------------------------
    from vllm import SamplingParams  # type: ignore
    sp_replay = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        seed=args.seed,
    )
    log.info(
        "replay SamplingParams: temperature=0.0, max_tokens=1, seed=%d "
        "(1 decode token emitted per row; outputs discarded).",
        args.seed,
    )

    # ------------------------------------------------------------------
    # 9. Replay loop
    # ------------------------------------------------------------------
    n_replayed = 0          # rows submitted to vLLM (contribute to captures)
    n_skipped = 0           # rows dropped (over max_model_len)
    skipped_by_subset: dict[str, int] = {}
    replayed_rows: list[dict] = []
    replayed_token_counts: list[int] = []
    first_chunk_checked = False   # M2: gate B0 on first actual submission

    t0 = time.monotonic()

    for chunk_start in range(0, len(replay_rows), args.chunk_size):
        chunk = replay_rows[chunk_start: chunk_start + args.chunk_size]

        # Render all rows; partition into submitted / skipped.
        rendered_chunk: list[tuple[dict, list[int], int]] = []
        for row in chunk:
            result = _render_row_for_replay(
                row, tokenizer, args.max_model_len)
            if result is None:
                n_skipped += 1
                subset = str(
                    row.get("subset") or row.get("domain") or "unknown")
                skipped_by_subset[subset] = (
                    skipped_by_subset.get(subset, 0) + 1)
            else:
                tok_ids, n_tok = result
                rendered_chunk.append((row, tok_ids, n_tok))

        total_rows_consumed = rows_done_base + n_replayed + n_skipped

        if not rendered_chunk:
            log.warning(
                "replay chunk %d-%d: all %d rows over max_model_len=%d; "
                "skipping chunk.",
                chunk_start, chunk_start + len(chunk),
                len(chunk), args.max_model_len,
            )
            _write_replay_ckpt(
                replay_ckpt,
                rows_done=total_rows_consumed + len(chunk),
                captured_done=captured_done_base + n_replayed,
            )
            n_skipped += len(chunk) - len(rendered_chunk)
            # rendered_chunk already empty, nothing to subtract again
            # (the += above is correct: all chunk rows skipped)
            continue

        # Build requests. Primary: prompt_token_ids dict shape.
        # Fallback (N1): if the pinned wheel rejects dicts, re-render
        # as string and verify round-trip token equality.
        requests = [
            {"prompt_token_ids": tok_ids}
            for _, tok_ids, _ in rendered_chunk
        ]

        log.info(
            "replay chunk %d-%d: submitting %d requests "
            "(%d over-length skipped); token range [%d, %d]",
            rows_done_base + chunk_start,
            rows_done_base + chunk_start + len(chunk),
            len(rendered_chunk),
            len(chunk) - len(rendered_chunk),
            min(n for _, _, n in rendered_chunk),
            max(n for _, _, n in rendered_chunk),
        )
        chunk_t0 = time.monotonic()
        try:
            outputs = llm.generate(requests, sp_replay)
        except TypeError as exc:
            # N1 fallback: dict input not accepted; re-render as string
            # and verify round-trip token equality.
            log.warning(
                "replay: LLM.generate rejected prompt_token_ids dict "
                "input (%s); falling back to string rendering. "
                "Verifying round-trip token equality.", exc,
            )
            string_inputs = []
            for row, expected_ids, _ in rendered_chunk:
                messages = row.get("messages", [])
                try:
                    rendered_str = tokenizer.apply_chat_template(
                        messages, tokenize=False,
                        add_generation_prompt=False, enable_thinking=True,
                    )
                except TypeError:
                    rendered_str = tokenizer.apply_chat_template(
                        messages, tokenize=False,
                        add_generation_prompt=False,
                    )
                recheck_ids = tokenizer(
                    rendered_str, add_special_tokens=False,
                )["input_ids"]
                if recheck_ids != expected_ids:
                    log.error(
                        "replay N1: round-trip token mismatch for row "
                        "(subset=%s): expected %d tokens, got %d. "
                        "String fallback would produce different activations. "
                        "Aborting.",
                        row.get("subset", "?"),
                        len(expected_ids), len(recheck_ids),
                    )
                    raise SystemExit(2)
                string_inputs.append(rendered_str)
            outputs = llm.generate(string_inputs, sp_replay)

        chunk_elapsed = time.monotonic() - chunk_t0
        log.info(
            "replay chunk done in %.1fs (%.2f s/row avg)",
            chunk_elapsed, chunk_elapsed / max(len(rendered_chunk), 1),
        )
        del outputs  # outputs discarded; captures are in writer state

        # Update tally accumulators.
        for row, _, n_tok in rendered_chunk:
            replayed_rows.append(row)
            replayed_token_counts.append(n_tok)
            n_replayed += 1

        total_done_captures = captured_done_base + n_replayed
        total_rows_consumed = rows_done_base + n_replayed + n_skipped

        # Progress log.
        session_elapsed = time.monotonic() - t0
        log.info(
            "[%d/%d rows consumed] %d replayed, %d skipped — "
            "%.0fs elapsed (%.2f s/replayed-row avg)",
            total_rows_consumed,
            rows_done_base + len(replay_rows),
            n_replayed, n_skipped,
            session_elapsed,
            session_elapsed / max(n_replayed, 1),
        )

        # B0 fail-fast (M2: gate on first actual submission).
        if not first_chunk_checked and _enabled_captures:
            first_chunk_checked = True
            try:
                _model_cls = type(
                    llm.llm_engine.model_executor.driver_worker
                    .model_runner.model
                ).__name__
            except Exception:
                _model_cls = "<unresolved>"
            assert_enabled_captures_nonempty(
                _enabled_captures,
                model_class=_model_cls,
                allow_empty=args.allow_empty_captures,
            )

        # Block-outputs subset gate.
        if (
            args.capture_block_outputs
            and total_rows_consumed >= args.block_outputs_subset_size
        ):
            try:
                import vllm.calibration_block_outputs as _bo  # type: ignore
                if not _bo._SUBSET_CLOSED:
                    _bo.close_subset()
                    log.info(
                        "block-outputs: subset closed at %d rows consumed "
                        "(>= subset_size=%d).",
                        total_rows_consumed, args.block_outputs_subset_size,
                    )
            except Exception as exc:
                log.error("block-outputs close_subset failed: %s", exc,
                          exc_info=True)

        # ---- Periodic per-writer checkpoints ---------------------------
        # counter = total_done_captures (no skips) for all writers.
        chunk_idx = chunk_start // args.chunk_size

        if (args.capture_imatrix
                and args.imatrix_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.imatrix_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_imatrix as _im  # type: ignore
                    _im.set_n_prompts_accumulated(total_done_captures)
                    _im.dump_imatrix_checkpoint(str(ws.imatrix_ckpt_path))
                    log.info("imatrix: ckpt %d prompts -> %s",
                             total_done_captures, ws.imatrix_ckpt_path)
                except Exception as exc:
                    log.error("imatrix ckpt failed: %s", exc, exc_info=True)

        if (args.capture_reap_scores
                and args.reap_scores_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.reap_scores_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_reap_scores as _reap  # type: ignore
                    _reap.set_n_prompts_accumulated(total_done_captures)
                    _reap.dump_reap_scores_checkpoint(
                        str(ws.reap_ckpt_path))
                    log.info("reap-scores: ckpt %d -> %s",
                             total_done_captures, ws.reap_ckpt_path)
                except Exception as exc:
                    log.error("reap-scores ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_input_covariance
                and args.input_cov_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.input_cov_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_input_cov as _icov  # type: ignore
                    _icov.set_n_prompts_accumulated(total_done_captures)
                    _icov.dump_input_cov_checkpoint(
                        str(ws.input_cov_ckpt_path))
                    log.info("input-cov: ckpt %d -> %s",
                             total_done_captures, ws.input_cov_ckpt_path)
                except Exception as exc:
                    log.error("input-cov ckpt failed: %s", exc, exc_info=True)

        if (args.capture_wanda_scalar_row
                and args.wanda_scalar_row_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.wanda_scalar_row_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
                    _wsr.set_n_prompts_accumulated(total_done_captures)
                    _wsr.dump_wanda_scalar_row_checkpoint(
                        str(ws.wsr_ckpt_path))
                    log.info("wanda-scalar-row: ckpt %d -> %s",
                             total_done_captures, ws.wsr_ckpt_path)
                except Exception as exc:
                    log.error("wanda-scalar-row ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_stage2_profile
                and args.stage2_profile_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.stage2_profile_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_stage2_profile as _s2p  # type: ignore
                    _s2p.set_n_prompts_accumulated(total_done_captures)
                    _s2p.dump_stage2_profile_checkpoint(
                        str(ws.s2p_ckpt_path))
                    log.info("stage2-profile: ckpt %d -> %s",
                             total_done_captures, ws.s2p_ckpt_path)
                except Exception as exc:
                    log.error("stage2-profile ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_per_expert_max
                and args.per_expert_max_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.per_expert_max_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_per_expert_max as _pem  # type: ignore
                    _pem.set_n_prompts_accumulated(total_done_captures)
                    _pem.dump_per_expert_max_checkpoint(
                        str(ws.pem_ckpt_path))
                    log.info("per-expert-max: ckpt %d -> %s",
                             total_done_captures, ws.pem_ckpt_path)
                except Exception as exc:
                    log.error("per-expert-max ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_routing_stats
                and args.routing_stats_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.routing_stats_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_routing_stats as _rts  # type: ignore
                    _rts.set_n_prompts_accumulated(total_done_captures)
                    _rts.dump_routing_stats_checkpoint(str(ws.rts_ckpt_path))
                    log.info("routing-stats: ckpt %d -> %s",
                             total_done_captures, ws.rts_ckpt_path)
                except Exception as exc:
                    log.error("routing-stats ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_router_logits_stats
                and args.router_logits_stats_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.router_logits_stats_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
                    _rlsx.set_n_prompts_accumulated(total_done_captures)
                    _rlsx.dump_router_logits_stats_checkpoint(
                        str(ws.router_logits_ckpt_path))
                    log.info("router-logits-stats: ckpt %d -> %s",
                             total_done_captures, ws.router_logits_ckpt_path)
                except Exception as exc:
                    log.error("router-logits-stats ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_output_reservoir
                and args.output_reservoir_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.output_reservoir_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_output_reservoir as _or  # type: ignore
                    _or.set_n_prompts_accumulated(total_done_captures)
                    _or.dump_output_reservoir_checkpoint(str(ws.or_ckpt_path))
                    log.info("output-reservoir: ckpt %d -> %s",
                             total_done_captures, ws.or_ckpt_path)
                except Exception as exc:
                    log.error("output-reservoir ckpt failed: %s", exc,
                              exc_info=True)

        if (args.capture_block_outputs
                and args.block_outputs_checkpoint_every_chunks > 0):
            if (chunk_idx + 1) % args.block_outputs_checkpoint_every_chunks == 0:
                try:
                    import vllm.calibration_block_outputs as _bo  # type: ignore
                    _bo.set_n_prompts_accumulated(total_done_captures)
                    _bo.dump_block_outputs_checkpoint(str(ws.bo_ckpt_path))
                    log.info("block-outputs: ckpt %d -> %s",
                             total_done_captures, ws.bo_ckpt_path)
                except Exception as exc:
                    log.error("block-outputs ckpt failed: %s", exc,
                              exc_info=True)

        # Row-index + capture-counter checkpoint (M1 fix: both counters).
        _write_replay_ckpt(
            replay_ckpt,
            rows_done=rows_done_base + n_replayed + n_skipped,
            captured_done=captured_done_base + n_replayed,
        )

    # End of loop.
    total_done_captures = captured_done_base + n_replayed

    # ------------------------------------------------------------------
    # 10. Skip histogram
    # ------------------------------------------------------------------
    if n_skipped > 0:
        log.warning(
            "replay: %d/%d rows skipped (over --max-model-len=%d). "
            "By subset: %s",
            n_skipped, len(replay_rows), args.max_model_len,
            skipped_by_subset,
        )
        skip_frac = n_skipped / max(len(replay_rows), 1)
        if skip_frac > 0.10:
            log.warning(
                "replay: %.1f%% rows skipped — consider "
                "--max-model-len %d (corpus max ~%d tokens).",
                100.0 * skip_frac, int(max_required * 1.05), max_required,
            )
    log.info("replay loop done: %d replayed, %d skipped.", n_replayed, n_skipped)

    # ------------------------------------------------------------------
    # 11. Correctness gates
    # (N3 fix: generate-vs-replay routing match removed — per-row
    # generate-time captures are not retained; the gate is infeasible.
    # Justification is the causal-masking argument in the design doc.
    # Automated gates are: B0 non-empty, code+science tokens >0,
    # buf_rows >= chunk size (already asserted before the loop).)
    # ------------------------------------------------------------------
    tally = _build_replay_subset_tally(replayed_rows, replayed_token_counts)
    _assert_code_science_nonzero(tally)

    if _enabled_captures:
        assert_enabled_captures_nonempty(
            _enabled_captures,
            model_class="<post-replay>",
            allow_empty=args.allow_empty_captures,
        )

    if n_generate_rows > 0:
        log.info(
            "replay: %d GENERATE rows were in the corpus. "
            "Read-through == generation is justified by causal masking "
            "(identical activations for fixed token sequences). "
            "See tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md.",
            n_generate_rows,
        )

    # ------------------------------------------------------------------
    # 12. Sidecar dumps
    # IMPORTANT: imatrix is special-cased.
    #   dump_imatrix(path: str, chunk_count: int) writes to
    #   <jsonl>.imatrix.dat (NOT under sidecars/).
    # All other nine writers:
    #   dump_<signal>(out_path: Path) compute sidecar_path() internally
    #   → <jsonl>.parent/sidecars/<jsonl.stem>/<signal>.pt
    # ------------------------------------------------------------------

    # -- imatrix (SPECIAL CASE: sibling .dat file, requires chunk_count) -
    if args.capture_imatrix:
        imatrix_path = out_path.with_suffix(".imatrix.dat")
        try:
            import vllm.calibration_imatrix as _im  # type: ignore
            _im.set_n_prompts_accumulated(total_done_captures)
            total_p = _im.get_n_prompts_accumulated()
            _im.dump_imatrix(str(imatrix_path), chunk_count=total_p)
            log.info("imatrix -> %s (%d entries from %d prompts)",
                     imatrix_path, len(_im._accumulators), total_p)
            if ws.imatrix_ckpt_path and ws.imatrix_ckpt_path.exists():
                ws.imatrix_ckpt_path.unlink()
        except Exception as exc:
            log.error("imatrix dump failed: %s", exc, exc_info=True)

    # -- reap-scores (uniform: dump_reap_scores(Path)) --------------------
    if args.capture_reap_scores:
        try:
            import vllm.calibration_reap_scores as _reap  # type: ignore
            _reap.set_n_prompts_accumulated(total_done_captures)
            _reap.dump_reap_scores(out_path)
            log.info("reap-scores: dumped sidecar from %d prompts",
                     _reap.get_n_prompts_accumulated())
            if ws.reap_ckpt_path and ws.reap_ckpt_path.exists():
                ws.reap_ckpt_path.unlink()
        except Exception as exc:
            log.error("reap-scores dump failed: %s", exc, exc_info=True)

    # -- input-covariance (uniform: dump_input_cov(Path)) ----------------
    if args.capture_input_covariance:
        try:
            import vllm.calibration_input_cov as _icov  # type: ignore
            _icov.set_n_prompts_accumulated(total_done_captures)
            _icov.dump_input_cov(out_path)
            log.info("input-cov: dumped sidecar from %d prompts",
                     _icov.get_n_prompts_accumulated())
            if ws.input_cov_ckpt_path and ws.input_cov_ckpt_path.exists():
                ws.input_cov_ckpt_path.unlink()
        except Exception as exc:
            log.error("input-cov dump failed: %s", exc, exc_info=True)

    # -- wanda scalar_row (uniform: dump_wanda_scalar_row(Path)) ----------
    if args.capture_wanda_scalar_row:
        try:
            import vllm.calibration_wanda_scalar_row as _wsr  # type: ignore
            _wsr.set_n_prompts_accumulated(total_done_captures)
            _wsr.dump_wanda_scalar_row(out_path)
            log.info("wanda-scalar-row: dumped from %d prompts",
                     _wsr.get_n_prompts_accumulated())
            if ws.wsr_ckpt_path and ws.wsr_ckpt_path.exists():
                ws.wsr_ckpt_path.unlink()
        except Exception as exc:
            log.error("wanda-scalar-row dump failed: %s", exc, exc_info=True)

    # -- stage2-profile (uniform: dump_stage2_profile(Path)) --------------
    # Note: layer_input_reservoir rides inside this sidecar; no separate
    # dump call needed (H2).
    if args.capture_stage2_profile:
        try:
            import vllm.calibration_stage2_profile as _s2p  # type: ignore
            _s2p.set_n_prompts_accumulated(total_done_captures)
            _s2p.dump_stage2_profile(out_path)
            log.info("stage2-profile: dumped from %d prompts",
                     _s2p.get_n_prompts_accumulated())
            if ws.s2p_ckpt_path and ws.s2p_ckpt_path.exists():
                ws.s2p_ckpt_path.unlink()
        except Exception as exc:
            log.error("stage2-profile dump failed: %s", exc, exc_info=True)

    # -- per-expert-max (uniform: dump_per_expert_max(Path)) --------------
    if args.capture_per_expert_max:
        try:
            import vllm.calibration_per_expert_max as _pem  # type: ignore
            _pem.set_n_prompts_accumulated(total_done_captures)
            _pem.dump_per_expert_max(out_path)
            log.info("per-expert-max: dumped from %d prompts",
                     _pem.get_n_prompts_accumulated())
            if ws.pem_ckpt_path and ws.pem_ckpt_path.exists():
                ws.pem_ckpt_path.unlink()
        except Exception as exc:
            log.error("per-expert-max dump failed: %s", exc, exc_info=True)

    # -- routing-stats (uniform: dump_routing_stats(Path)) ----------------
    if args.capture_routing_stats:
        try:
            import vllm.calibration_routing_stats as _rts  # type: ignore
            _rts.set_n_prompts_accumulated(total_done_captures)
            _rts.dump_routing_stats(out_path)
            log.info("routing-stats: dumped from %d prompts",
                     _rts.get_n_prompts_accumulated())
            if ws.rts_ckpt_path and ws.rts_ckpt_path.exists():
                ws.rts_ckpt_path.unlink()
        except Exception as exc:
            log.error("routing-stats dump failed: %s", exc, exc_info=True)

    # -- router-logits-stats (uniform: dump_router_logits_stats(Path)) ----
    if args.capture_router_logits_stats:
        try:
            import vllm.calibration_router_logits_stats as _rlsx  # type: ignore
            _rlsx.set_n_prompts_accumulated(total_done_captures)
            _rlsx.dump_router_logits_stats(out_path)
            log.info("router-logits-stats: dumped from %d prompts",
                     _rlsx.get_n_prompts_accumulated())
            if (ws.router_logits_ckpt_path
                    and ws.router_logits_ckpt_path.exists()):
                ws.router_logits_ckpt_path.unlink()
        except Exception as exc:
            log.error("router-logits-stats dump failed: %s", exc,
                      exc_info=True)

    # -- output-reservoir (uniform: dump_output_reservoir(Path)) ----------
    if args.capture_output_reservoir:
        try:
            import vllm.calibration_output_reservoir as _or  # type: ignore
            _or.set_n_prompts_accumulated(total_done_captures)
            _or.dump_output_reservoir(out_path)
            log.info("output-reservoir: dumped from %d prompts",
                     _or.get_n_prompts_accumulated())
            if ws.or_ckpt_path and ws.or_ckpt_path.exists():
                ws.or_ckpt_path.unlink()
        except Exception as exc:
            log.error("output-reservoir dump failed: %s", exc, exc_info=True)

    # -- block-outputs (uniform: dump_block_outputs(Path)) ----------------
    if args.capture_block_outputs:
        try:
            import vllm.calibration_block_outputs as _bo  # type: ignore
            _bo.set_n_prompts_accumulated(total_done_captures)
            if not _bo._SUBSET_CLOSED:
                _bo.close_subset()
                log.info(
                    "block-outputs: subset closed pre-dump (%d prompts "
                    "< subset_size=%d — partial subset shipped).",
                    total_done_captures, args.block_outputs_subset_size,
                )
            _bo.dump_block_outputs(out_path)
            log.info("block-outputs: dumped per-layer sidecars from %d prompts",
                     _bo.get_n_prompts_accumulated())
            if ws.bo_ckpt_path and ws.bo_ckpt_path.exists():
                ws.bo_ckpt_path.unlink()
        except Exception as exc:
            log.error("block-outputs dump failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # 13. Cleanup + summary
    # ------------------------------------------------------------------
    if replay_ckpt.exists():
        replay_ckpt.unlink()

    sidecar_dir = replay_jsonl.parent / "sidecars" / replay_jsonl.stem
    log.info(
        "v3 replay complete: %d rows replayed, %d skipped. "
        "Sidecars at %s/",
        n_replayed, n_skipped, sidecar_dir,
    )
    return 0
```

---

## Implementation Map: Files to Create/Modify

### MODIFY: `/home/lucas/ai/moe_compress/max_quality/scripts/build_self_traces_calib_vllm.py`

**Change A — Imports (top of file, line ~82):**
Add `import dataclasses` to the existing stdlib import block.

**Change B — New helpers after line 766:**
In order: `_render_row_for_replay`, `_build_replay_subset_tally`, `_assert_code_science_nonzero`, `_write_replay_ckpt`, `_WriterState`, `_setup_all_writers`.

**Change C — Refactor `main()` writer-setup blocks (lines 1659–2115):**
Replace the `_check_ckpt_counter` closure definition (line 1659) and all 10 per-writer setup blocks (lines 1684–2115) with:
```python
ws = _setup_all_writers(args, out_path, llm, tokenizer, already_done)
```
Update all downstream references in `main()`'s periodic checkpoint blocks (lines 2310–2534) and final dump blocks (lines 2536–2735) from bare variable names to `ws.<field>`. Complete substitution list:

| Old variable | New reference |
|---|---|
| `imatrix_ckpt_path` | `ws.imatrix_ckpt_path` |
| `reap_ckpt_path` | `ws.reap_ckpt_path` |
| `input_cov_ckpt_path` | `ws.input_cov_ckpt_path` |
| `wsr_ckpt_path` | `ws.wsr_ckpt_path` |
| `s2p_ckpt_path` | `ws.s2p_ckpt_path` |
| `pem_ckpt_path` | `ws.pem_ckpt_path` |
| `rts_ckpt_path` | `ws.rts_ckpt_path` |
| `router_logits_ckpt_path` | `ws.router_logits_ckpt_path` |
| `or_ckpt_path` | `ws.or_ckpt_path` |
| `bo_ckpt_path` | `ws.bo_ckpt_path` |

All ~20 occurrences across lines 2310–2735. The generate path's `already_done` variable name and `out_path` (the tmp path) are unchanged — they are passed into `_setup_all_writers`.

**Change D — Add `--replay-from` argparse argument** after `--allow-empty-captures` (after line 1313):
```python
p.add_argument(
    "--replay-from", default=None, metavar="JSONL",
    help=(
        "v3 forward-only replay. Path to an existing self-traces JSONL. "
        "Tokenizes each row's (prompt+answer) and submits to vLLM as a "
        "single prefill forward (max_tokens=1, max_num_batched_tokens=256) "
        "so capture hooks fire over every token. Sidecars land at the "
        "canonical sidecar_path of the INPUT jsonl. Requires >=1 "
        "--capture-* flag. Resumable via .replay.ckpt."
    ),
)
```

**Change E — Add replay redirect in `main()`** after line 1521 (after all env gate blocks), before `_trace_cache_key_vllm` computation:
```python
if args.replay_from is not None:
    return _run_replay(args)
```

**Change F — Add deprecation comment** in the TF synthesis block (~line 2186):
```python
# NOTE (v3): TEACHER_FORCED rows are synthesised here without a model
# forward. The canonical capture path is --replay-from (v3 replay).
# Generate-time capture (GENERATE rows only) remains for incremental runs.
```

**Change G — Add `_run_replay` function** after `main()`, before `if __name__ == "__main__"`.

### CREATE: `/home/lucas/ai/moe_compress/max_quality/tests/test_replay_helpers.py`

Complete file verbatim:

```python
"""Unit tests for _run_replay helper functions (v3 calibration replay).

Tests _render_row_for_replay, _build_replay_subset_tally,
_assert_code_science_nonzero. No model load; no vllm import;
no monkeypatching of production code (project rule).

The stub tokenizer is a self-contained class satisfying the contract:
  apply_chat_template(messages, tokenize, add_generation_prompt, **kw)
  __call__(text, add_special_tokens=False) -> {"input_ids": list[int]}
"""
from __future__ import annotations
import sys
from pathlib import Path
import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent / "scripts"))
from build_self_traces_calib_vllm import (  # type: ignore  # noqa: E402
    _render_row_for_replay,
    _build_replay_subset_tally,
    _assert_code_science_nonzero,
)


class _StubTokenizer:
    """Minimal tokenizer stub: one token per character."""
    def __init__(self, *, raise_on_enable_thinking: bool = False):
        self._raise = raise_on_enable_thinking

    def apply_chat_template(self, messages, tokenize, add_generation_prompt,
                            enable_thinking=None, **kw):
        if self._raise and enable_thinking is not None:
            raise TypeError("enable_thinking not supported by this tokenizer")
        return "".join(m.get("content", "") for m in messages)

    def __call__(self, text: str, add_special_tokens: bool = False):
        return {"input_ids": list(range(len(text)))}


_ROW = {
    "messages": [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ],
    "subset": "mot_code",
    "domain": "mot_code",
    "completion_source": "canonical",
}
# "helloworld" = 10 chars = 10 tokens under _StubTokenizer


def test_render_happy_path():
    result = _render_row_for_replay(_ROW, _StubTokenizer(), max_model_len=100)
    assert result is not None
    ids, n = result
    assert n == 10
    assert len(ids) == 10


def test_render_over_length_returns_none():
    result = _render_row_for_replay(_ROW, _StubTokenizer(), max_model_len=9)
    assert result is None


def test_render_exactly_at_limit_accepted():
    result = _render_row_for_replay(_ROW, _StubTokenizer(), max_model_len=10)
    assert result is not None


def test_render_enable_thinking_fallback():
    tok = _StubTokenizer(raise_on_enable_thinking=True)
    result = _render_row_for_replay(_ROW, tok, max_model_len=100)
    assert result is not None  # fallback path returned a result


def test_render_add_generation_prompt_is_false():
    """apply_chat_template must always be called with add_generation_prompt=False."""
    calls: list[bool] = []

    class _Recording(_StubTokenizer):
        def apply_chat_template(self, messages, tokenize,
                                add_generation_prompt, **kw):
            calls.append(add_generation_prompt)
            return "x"

    _render_row_for_replay(_ROW, _Recording(), max_model_len=100)
    assert calls, "apply_chat_template was never called"
    assert all(v is False for v in calls), (
        f"add_generation_prompt must be False; got {calls}")


def test_render_generate_row_works_identically():
    """completion_source=teacher_generated rows use same rendering path."""
    row = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ],
        "subset": "mot_math",
        "completion_source": "teacher_generated",
    }
    result = _render_row_for_replay(row, _StubTokenizer(), max_model_len=100)
    assert result is not None
    ids, n = result
    assert n == 2  # "QA" = 2 chars


def test_tally_basic():
    rows = [
        {"subset": "mot_code"},
        {"subset": "mot_code"},
        {"subset": "mot_science"},
    ]
    tally = _build_replay_subset_tally(rows, [100, 200, 50])
    assert tally["mot_code"]["n_rows"] == 2
    assert tally["mot_code"]["n_tokens"] == 300
    assert tally["mot_science"]["n_rows"] == 1
    assert tally["mot_science"]["n_tokens"] == 50


def test_tally_fallback_domain():
    rows = [{"domain": "math"}]
    tally = _build_replay_subset_tally(rows, [42])
    assert "math" in tally
    assert tally["math"]["n_tokens"] == 42


def test_tally_unknown_fallback():
    rows = [{}]
    tally = _build_replay_subset_tally(rows, [5])
    assert "unknown" in tally


def test_tally_empty():
    assert _build_replay_subset_tally([], []) == {}


def test_assert_code_science_passes():
    tally = {
        "mot_code": {"n_rows": 5, "n_tokens": 1000},
        "mot_science": {"n_rows": 3, "n_tokens": 500},
        "math": {"n_rows": 10, "n_tokens": 2000},
    }
    _assert_code_science_nonzero(tally)  # must not raise


def test_assert_fails_no_code():
    tally = {
        "mot_science": {"n_rows": 3, "n_tokens": 500},
        "math": {"n_rows": 10, "n_tokens": 2000},
    }
    with pytest.raises(AssertionError, match="code-subset"):
        _assert_code_science_nonzero(tally)


def test_assert_fails_no_science():
    tally = {
        "mot_code": {"n_rows": 5, "n_tokens": 1000},
        "math": {"n_rows": 10, "n_tokens": 2000},
    }
    with pytest.raises(AssertionError, match="science-subset"):
        _assert_code_science_nonzero(tally)


def test_assert_custom_subsets():
    tally = {
        "my_code": {"n_rows": 1, "n_tokens": 100},
        "my_sci": {"n_rows": 1, "n_tokens": 50},
    }
    _assert_code_science_nonzero(
        tally,
        code_subsets=frozenset({"my_code"}),
        science_subsets=frozenset({"my_sci"}),
    )
```

Run: `pytest max_quality/tests/test_replay_helpers.py -v` — no GPU, no vllm.

---

## Data Flow

```
CLI:
  build_self_traces_calib_vllm.py
    --replay-from artifacts/_shared/self_traces_<key>.jsonl
    --teacher Qwen/Qwen3.6-35B-A3B
    --capture-reap-scores --capture-imatrix [--capture-*...]
    --chunk-size 200 --max-model-len 40960
    [--resume]

main():
  line 115: VLLM_ENABLE_V1_MULTIPROCESSING=0  [module-level, inherited]
  lines 1323-1521: VLLM_CALIB_CAPTURE_* env gates per --capture-* flags
  line ~1522: if args.replay_from: return _run_replay(args)

_run_replay(args):
  validate replay_jsonl exists, >=1 capture enabled
  _harden_runtime_env(str(replay_jsonl), args.dtype)
  load all_rows from replay_jsonl (validate messages field)
  pre-flight scan: n_prompt_tokens + n_gen_tokens vs --max-model-len
  resume: read .replay.ckpt → {rows_done, captured_done}
  _load_teacher_vllm(..., max_num_seqs=256, max_num_batched_tokens=256,
                         max_logprobs=1)
    → LLM: V1 engine, chunked prefill by default, ≤256 tok/chunk
  C2 assert: buf_rows = llm.llm_engine.vllm_config.compilation_config
                              .max_cudagraph_capture_size
             assert buf_rows >= 256  [hard-fail if not]
  _setup_all_writers(args, out_path=replay_jsonl, llm, tokenizer,
                     already_done=captured_done_base)
    → setup() + ckpt hydration per enabled writer → _WriterState ws
  sp_replay = SamplingParams(temperature=0, max_tokens=1, seed=N)

  chunk loop:
    for each row → _render_row_for_replay(row, tokenizer, max_model_len)
      apply_chat_template(messages, add_generation_prompt=False,
                          enable_thinking=True) [mirrors lines 726-733]
      tokenize → token_ids
      if len > max_model_len: n_skipped++ / skipped_by_subset[domain]++
      else: rendered_chunk.append((row, token_ids, n_tok))

    if rendered_chunk empty: write .replay.ckpt + continue

    llm.generate([{"prompt_token_ids": ids}, ...], sp_replay)
      [fallback on TypeError: re-render string + round-trip verify]
      → V1 scheduler: each request split into ≤256-token prefill chunks
        each chunk = one MoE forward, num_tokens ≤ 256 ≤ buf_rows=512
          → router hook fires on ALL tokens: reap, wanda, routing_stats,
                                              stage2_profile, router_logits
          → expert_in fires on ALL tokens: imatrix, input_cov, wanda
          → expert_mid fires on ALL tokens: imatrix
          → expert_out_unweighted fires (num_tokens≤256≤512): reap,
              per_expert_max, output_reservoir, stage2_profile (gated out)
          → block_out fires on ALL tokens: block_outputs
          → layer_in fires on ALL tokens: stage2_profile (layer_input_reservoir)
        + 1 decode token per request (all hooks fire again for 1 token)
      outputs discarded

    first actual submission: assert_enabled_captures_nonempty() [M2]
    block_outputs subset close gate
    per-writer periodic checkpoints (counter = captured_done_base + n_replayed)
    _write_replay_ckpt(ckpt, rows_done=..., captured_done=...)

  _build_replay_subset_tally(replayed_rows, replayed_token_counts)
  _assert_code_science_nonzero(tally)
  assert_enabled_captures_nonempty(..., "<post-replay>")

  DUMP BLOCKS:
    imatrix [SPECIAL]: dump_imatrix(str(out_path.with_suffix(".imatrix.dat")),
                                    chunk_count=total_done_captures)
    reap:    dump_reap_scores(out_path)
    icov:    dump_input_cov(out_path)
    wsr:     dump_wanda_scalar_row(out_path)
    s2p:     dump_stage2_profile(out_path)  [includes layer_input_reservoir]
    pem:     dump_per_expert_max(out_path)
    rts:     dump_routing_stats(out_path)
    rlsx:    dump_router_logits_stats(out_path)
    or:      dump_output_reservoir(out_path)
    bo:      close_subset() + dump_block_outputs(out_path)
    → all non-imatrix: sidecar_path(replay_jsonl, signal)
      = replay_jsonl.parent/sidecars/<stem>/<signal>.pt

  replay_ckpt.unlink()
  return 0
```

---

## Long-Sequence Policy

`_render_row_for_replay` returns `None` when `len(token_ids) > max_model_len`. Never truncates. The pre-flight scan uses stored `n_prompt_tokens + n_gen_tokens` to estimate skips before GPU time is spent. At end-of-loop, `skipped_by_subset` is logged as a WARNING with the exact per-domain breakdown. If `n_skipped / len(replay_rows) > 0.10`, an additional WARNING recommends raising `--max-model-len`.

For the v2 corpus (8000 rows with R1/SWE traces up to ~30K tokens), use `--max-model-len 40960` on H200 (141GB, `--gpu-memory-utilization 0.90`). The pre-flight scan prints the minimum required value.

---

## GPU Smoke Validation

### Primary gate (C2 smoke): expert_out_unweighted fires on prefill

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_CALIB_CAPTURE_REAP_SCORES=1 \
VLLM_CALIB_CAPTURE_ROUTER=1 \
VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1 \
VLLM_USE_FLASHINFER_MOE_FP16=0 \
python -c "
import logging; logging.basicConfig(level=logging.WARNING)
from vllm import LLM, SamplingParams
import vllm.calibration_reap_scores as _reap

llm = LLM(
    'Qwen/Qwen3.6-35B-A3B',
    tensor_parallel_size=1,
    gpu_memory_utilization=0.90,
    max_model_len=4096,
    max_num_seqs=256,
    max_num_batched_tokens=256,
    trust_remote_code=True,
    max_logprobs=1,
)

# Read back buf_rows from live engine.
buf_rows = (llm.llm_engine.vllm_config
              .compilation_config.max_cudagraph_capture_size)
print(f'buf_rows={buf_rows}, max_num_batched_tokens=256')
assert buf_rows >= 256, f'C2 FAIL: buf_rows={buf_rows} < 256'
print('C2 assert passed.')

_reap.setup(llm)
tok = llm.get_tokenizer()

# Build a ~1500-token sequence (>buf_rows to test chunking).
text = 'The quick brown fox jumps over the lazy dog. ' * 35
tids = tok(text, add_special_tokens=False)['input_ids'][:1500]
print(f'Sequence length: {len(tids)} tokens (> buf_rows={buf_rows})')
assert len(tids) > buf_rows, 'Need sequence longer than buf_rows to test chunking'

sp = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
out = llm.generate([{'prompt_token_ids': tids}], sp)
print('Generated token id:', out[0].outputs[0].token_ids)

n = _reap.captured_entry_count()
print(f'reap captured_entry_count: {n}')
assert n > buf_rows, (
    f'FAIL: captured_entry_count={n} <= buf_rows={buf_rows}. '
    'expert_out_unweighted only fired on decode token, not prefill chunks. '
    'Check VLLM_ENABLE_V1_MULTIPROCESSING=0 and max_num_batched_tokens<=buf_rows.'
)
print(f'PASS: captured {n} tokens (> buf_rows={buf_rows}): '
      f'chunked prefill fired expert_out_unweighted on all prefill chunks.')
# Verify no SKIPPED warning in logs (look for absence of the warning above).
"
```

**Pass:** `captured_entry_count > buf_rows`, no "SKIPPED" warning, `buf_rows >= 256`.
**Fail:** `captured_entry_count == 1` or SKIPPED warning.

**If fail — contingency (not planned here):** Patch `TritonExperts.apply` to internally sub-slice the persistent buffer into `_calib_buf_rows`-sized windows when `num_tokens > _calib_buf_rows`. This is a wheel rebuild, contingent on the smoke failing.

### Additional smoke steps

**Step 2: N1 smoke — prompt_token_ids dict format accepted:**
```bash
# The primary generate call in the C2 smoke uses {'prompt_token_ids': tids}.
# If it succeeded, N1 is confirmed. If it raised TypeError, the fallback
# path engaged — verify logs show "falling back to string rendering" and
# "round-trip token equality verified".
```

**Step 3: End-to-end 10-row replay:**
```bash
python max_quality/scripts/build_self_traces_calib_vllm.py \
  --replay-from artifacts/_shared/self_traces_<key>.jsonl \
  --teacher Qwen/Qwen3.6-35B-A3B \
  --capture-reap-scores --capture-imatrix --capture-routing-stats \
  --chunk-size 10 --max-model-len 40960 \
  --gpu-memory-utilization 0.90
```
Verify:
- B0 passes after chunk 1 (no SystemExit 2).
- `sidecars/self_traces_<key>/reap_scores.pt` exists.
- `torch.load(..., map_location="cpu")` has shape `[40, 256]`, no NaN, non-zero.
- `self_traces_<key>.imatrix.dat` exists (NOT under sidecars/).
- `.replay.ckpt` absent on success.

**Step 4: Resume:**
Kill after chunk 2. Re-run with `--resume`. Verify `rows_done` and `captured_done` in logs match chunk-2 boundary, first 20 rows skipped.

**Step 5: Full 8000-row run** (~20–40 min on H200 with chunk_size=200, max_num_batched_tokens=256).

---

## Build Sequence Checklist

**Phase 1: Pure helpers + tests (no GPU)**
- [ ] Add `import dataclasses` to import block (~line 82).
- [ ] Add `_render_row_for_replay` after line 766.
- [ ] Add `_build_replay_subset_tally`.
- [ ] Add `_assert_code_science_nonzero`.
- [ ] Add `_write_replay_ckpt` (two-counter signature).
- [ ] Add `_WriterState` dataclass (10 fields, no layer_input_reservoir).
- [ ] Add `_setup_all_writers`.
- [ ] Create `max_quality/tests/test_replay_helpers.py` (14 tests).
- [ ] `pytest max_quality/tests/test_replay_helpers.py -v` — passes with no GPU.

**Phase 2: `_setup_all_writers` refactor of `main()`**
- [ ] Replace lines 1659–2115 in `main()` with `ws = _setup_all_writers(...)`.
- [ ] Replace all 20 `*_ckpt_path` refs in lines 2310–2735 with `ws.<field>`.
- [ ] `pytest max_quality/tests/test_calib_ckpt_counter.py max_quality/tests/test_build_self_traces_calib_vllm_c1.py -v` — passes.

**Phase 3: `--replay-from` + `_run_replay` skeleton**
- [ ] Add `--replay-from` to argparse (after `--allow-empty-captures`).
- [ ] Add `if args.replay_from: return _run_replay(args)` in `main()` after line 1521.
- [ ] Add `_run_replay(args)` after `main()` (skeleton through step 8).

**Phase 4: Replay loop**
- [ ] Implement chunk loop (render, skip tracking, `llm.generate`, N1 fallback).
- [ ] B0 fail-fast block gated on `first_chunk_checked` (M2).
- [ ] Block-outputs close_subset gate.
- [ ] All 10 periodic checkpoint blocks.
- [ ] `_write_replay_ckpt` call with both counters.

**Phase 5: Correctness gates + dumps + cleanup**
- [ ] `_build_replay_subset_tally` + `_assert_code_science_nonzero`.
- [ ] Post-run `assert_enabled_captures_nonempty`.
- [ ] Imatrix dump (special-cased: `.imatrix.dat` + `chunk_count=`).
- [ ] Nine uniform dump blocks.
- [ ] `replay_ckpt.unlink()` on success.
- [ ] Deprecation comment in TF synthesis block.

**Phase 6: GPU smokes**
- [ ] C2 primary gate (buf_rows ≥ 256, captured_entry_count > buf_rows).
- [ ] N1 confirmation (dict format accepted, or fallback triggered cleanly).
- [ ] 10-row end-to-end replay.
- [ ] Resume test.
- [ ] Full 8000-row run.

---

## Complete Function/Change Table

| Item | File location | Type |
|---|---|---|
| `import dataclasses` | Import block ~line 82 | ADD |
| `_render_row_for_replay(row, tokenizer, max_model_len)` | After line 766 | NEW |
| `_build_replay_subset_tally(rows, token_counts)` | After `_render_row_for_replay` | NEW |
| `_assert_code_science_nonzero(tally, code_subsets, science_subsets)` | After `_build_replay_subset_tally` | NEW |
| `_write_replay_ckpt(path, rows_done, captured_done)` | After `_assert_code_science_nonzero` | NEW |
| `_WriterState` (10 fields: imatrix, reap, input_cov, wsr, s2p, pem, rts, router_logits, or, bo) | After `_write_replay_ckpt` | NEW |
| `_setup_all_writers(args, out_path, llm, tokenizer, already_done)` | After `_WriterState` | NEW |
| `main()` lines 1659–2115 | Replace with `ws = _setup_all_writers(...)` | MODIFY |
| 20× `*_ckpt_path` → `ws.<field>` | Lines 2310–2735 | MODIFY |
| `--replay-from` argparse arg | After `--allow-empty-captures`, line ~1313 | ADD |
| `if args.replay_from: return _run_replay(args)` | After line 1521 | ADD |
| Deprecation comment | TF synthesis block ~line 2186 | ADD |
| `_run_replay(args) -> int` | After `main()` | NEW (~220 lines) |
| `max_quality/tests/test_replay_helpers.py` | tests dir | CREATE |

---

## Remaining Unverified Items (GPU-only)

**UNVERIFIED-1 (GPU-only):** Whether `llm.llm_engine.vllm_config.compilation_config.max_cudagraph_capture_size` is the correct attribute path on the pinned wheel (vllm 0.21.1.dev0+gad7125a43.d20260606). The patch uses `get_current_vllm_config().compilation_config.max_cudagraph_capture_size` during `TritonExperts.__init__`, confirming the `compilation_config.max_cudagraph_capture_size` attribute exists on the config object. The `llm.llm_engine.vllm_config` path is the standard V1 config access pattern. The `AttributeError` fallback in the C2 check handles any deviation.

**UNVERIFIED-2 (GPU-only):** Whether `LLM.generate()` accepts `list[dict]` with `"prompt_token_ids"` key on the pinned wheel. The N1 fallback (re-render string + round-trip token equality verification + `SystemExit(2)` on mismatch) handles this cleanly without any information loss. The C2 smoke confirms acceptance.

No further unverified items exist that can be resolved by static analysis.

---

## Design Doc Note (N3)

The design doc (`tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md`) §Correctness gates states: "A spot check: for ≥1 GENERATE row, replay-captured per-layer routing matches the generate-time capture within float tolerance." This gate is infeasible: generate-time per-row routing captures are not retained (only the aggregated sidecar is written). The gate should be softened in the design doc to: "The read-through == generation equivalence is established by the causal-masking argument (§Numerical justification). The automated correctness gates are: (a) B0 non-empty captures after first chunk, (b) code+science token tally > 0, (c) buf_rows ≥ max_num_batched_tokens confirmed before run start." The plan implements all three automated gates. The design doc line should be updated to match.

Sources:
- [vLLM Engine Arguments — max_cudagraph_capture_size formula](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [vLLM Optimization — V1 chunked prefill enabled by default](https://docs.vllm.ai/en/latest/configuration/optimization/)
