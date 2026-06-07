# PLAN — Calibration v3 forward-only replay (CORRECTED FINAL, verbatim)

Implements tasks/CALIBRATION_V3_CAPTURE_REPLAY_DESIGN.md. Branch feat/calib-v3-replay.
Authoritative spec, full inlined code. Round-2 fixes applied: C-NEW-1 (all-skip
counter) + L-NEW-1 (two-path buf_rows probe).

---

# Calibration v3 — Forward-Only Replay: CORRECTED FINAL Implementation Plan

## Changes in This Revision

**C-NEW-1 (CRITICAL — fixed):** The all-skip-chunk branch in `_run_replay` corrupted `rows_done` by adding `len(chunk)` on top of a `total_rows_consumed` that already included the chunk's skips (which were accumulated per-row during rendering), then incremented `n_skipped` a third time. The fix: write `rows_done=rows_done_base + n_replayed + n_skipped` directly — the per-row `n_skipped += 1` accumulator inside the rendering loop is the sole source of truth. The misleading comment and the spurious `n_skipped += len(chunk) - len(rendered_chunk)` line are deleted. Partial-skip chunks are correct under this formula: the rendering loop accumulates exactly the right `n_skipped` before control reaches the checkpoint write in either branch.

**L-NEW-1 (hardening — applied):** The C2 `AttributeError` fallback now tries two attribute paths before giving up. First: `llm.llm_engine.vllm_config.compilation_config.max_cudagraph_capture_size`. Second (new): `llm.llm_engine.model_executor.driver_worker.model_runner.vllm_config.compilation_config.max_cudagraph_capture_size` (the same model-runner path the generate code uses at line 2277). Only if both paths raise `AttributeError` does it fall back to the formula — and at that point it logs at `ERROR` (not WARNING) to tell the operator the buf_rows guarantee is unverified.

All other content from the previous plan is unchanged and confirmed correct.

---

## Confirmed-Correct Unchanged Items

- All 9 prior findings remain fixed.
- API names, non-imatrix dump signatures, imatrix special-case, sidecar stem, argparse flags, B0/C1 env ordering: unchanged and confirmed correct.
- `_render_row_for_replay`, `_build_replay_subset_tally`, `_assert_code_science_nonzero` helpers: unchanged.
- `_WriterState` (10 fields, no layer_input_reservoir): unchanged.
- `_setup_all_writers`: unchanged.
- Unit test file (14 tests): unchanged.
- Build sequence, implementation map, data flow, smoke plan: unchanged except the C-NEW-1 and L-NEW-1 corrections noted in `_run_replay`.

---

## Patterns & Conventions (file:line refs)

**B0/C1 invariant:** `os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"` module-level line 115. Replay inherits unconditionally.

**Capture env gates:** `VLLM_CALIB_CAPTURE_*` set in `main()` lines 1323–1521. `if args.replay_from: return _run_replay(args)` placed after line 1521.

**`_load_teacher_vllm`** (line 364): signature `(repo, revision, dtype, gpu_memory_utilization, max_model_len, max_num_seqs=None, max_num_batched_tokens=None, max_logprobs=50)`. No change. Replay calls with `max_num_seqs=256, max_num_batched_tokens=256, max_logprobs=1`.

**TF rendering** (lines 726–733): `apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=True)` + `TypeError` fallback. Mirrored exactly by `_render_row_for_replay`.

**Imatrix dump is special:** `dump_imatrix(path: str, chunk_count: int)` writes to `<jsonl>.imatrix.dat` (generate path lines 2540–2559). Every other writer: `dump_<signal>(out_path: Path)`.

**Nine non-imatrix dump signatures (confirmed):** `dump_reap_scores(Path)`, `dump_input_cov(Path)`, `dump_wanda_scalar_row(Path)`, `dump_stage2_profile(Path)`, `dump_per_expert_max(Path)`, `dump_routing_stats(Path)`, `dump_router_logits_stats(Path)`, `dump_output_reservoir(Path)`, `dump_block_outputs(Path)`.

**`layer_input_reservoir` rides `stage2_profile`:** no standalone dump, no 11th `_WriterState` field.

**`_CAPTURE_WRITER_MODULES`** (line 194): `layer_input_reservoir` intentionally absent.

**Model-runner path for model class resolution** (generate path lines 2277–2279): `llm.llm_engine.model_executor.driver_worker.model_runner.model`. L-NEW-1 adds `.vllm_config.compilation_config.max_cudagraph_capture_size` on the same spine as a second buf_rows probe path.

**`_write_replay_ckpt` two-counter contract (M1):** `rows_done` slices the JSONL on resume (includes skips); `captured_done` is passed to `_setup_all_writers` as `already_done` for writer counter checks (excludes skips). Per-row `n_skipped += 1` is the sole accumulator — no other site increments it.

---

## Complete Verbatim Code

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
        "replay: code subsets %s -> %d tokens; "
        "science subsets %s -> %d tokens",
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

    Two counters stored separately (M1 fix):
      rows_done     -- total rows consumed from the JSONL (replayed +
                       skipped). Used to slice the JSONL on resume.
                       Per-row n_skipped += 1 is the SOLE accumulator;
                       no other site may increment n_skipped.
      captured_done -- rows that contributed to captures (no skips).
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
    Dataclass gives AttributeError on typo instead of silent None.

    Ten fields; layer_input_reservoir has no field because it rides
    inside stage2_profile with no separate dump call (H2).
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

    ``out_path`` determines checkpoint file locations:
      generate path: the output JSONL tmp path
      replay path  : the input JSONL (replay_jsonl)
    All per-writer .ckpt files land as siblings of out_path.

    ``already_done`` for the generate path: JSONL rows already written.
    For the replay path: captured_done (rows that contributed to
    captures, EXCLUDING skipped/over-length rows). This is the correct
    counter for _ckpt_counter_check which validates capture coverage.

    The _check_ckpt_counter closure from main() (~line 1659) is
    recreated as a lambda capturing already_done and
    args.allow_counter_divergence from the caller's scope.

    Returns _WriterState with all ten ckpt path fields (None if disabled).
    """
    ws = _WriterState()

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
                log.error(
                    "imatrix: ckpt schema mismatch (%s); deleting", exc)
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
                log.error(
                    "reap-scores: ckpt schema mismatch (%s); deleting", exc)
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
                log.error(
                    "input-cov: ckpt schema mismatch (%s); deleting", exc)
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
                    "routing-stats: ckpt schema mismatch (%s); deleting",
                    exc)
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
                log.info(
                    "block-outputs: hydrated %d-prompt ckpt (closed=%s)",
                    loaded, _bo._SUBSET_CLOSED)
            except ValueError as exc:
                log.error(
                    "block-outputs: ckpt schema mismatch (%s); deleting",
                    exc)
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

    Imatrix dump is special-cased: dump_imatrix(str, chunk_count=int)
    writes <jsonl>.imatrix.dat alongside the JSONL (NOT under sidecars/).
    All other nine writers use dump_<signal>(Path) and compute their
    sidecar path internally.

    Counter contract (M1 fix):
      n_skipped is incremented ONLY in the per-row rendering loop.
      No other site may increment it. rows_done = rows_done_base +
      n_replayed + n_skipped is always the correct JSONL position.
      captured_done = captured_done_base + n_replayed excludes skips
      and is passed to _setup_all_writers for writer counter checks.

    Returns 0 on success, 1 on configuration/validation error.
    SystemExit(2) if B0 hard-fails (assert_enabled_captures_nonempty).
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

    # Runtime env hardening (compile cache, JIT cap).
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
            "Continuing -- may indicate an unexpected corpus."
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
    # 4. Resume handling -- two counters (M1 fix)
    #
    # rows_done    = total rows consumed (replayed + skipped).
    #               Used to slice the JSONL on resume.
    # captured_done = rows that contributed to captures (no skips).
    #               Passed to _setup_all_writers as already_done.
    # ------------------------------------------------------------------
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
    #
    # max_num_seqs=256 set explicitly so:
    #   max_cudagraph_capture_size = min(256*2, 512) = 512 (predictable).
    # max_num_batched_tokens=256 caps each prefill chunk so every MoE
    # forward has num_tokens <= 256 <= buf_rows, making
    # expert_out_unweighted fire on all prefill tokens (not just the
    # single decode token).
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
    # 6. C2 runtime buffer-size assertion (L-NEW-1 hardening)
    #
    # Read back the actual max_cudagraph_capture_size from the live
    # engine. If buf_rows < _REPLAY_MAX_BATCHED_TOKENS the
    # expert_out_unweighted hook would silently skip prefill chunks and
    # reap/per_expert_max/output_reservoir would only capture the single
    # decode token -- the whole chunked-prefill fix would silently fail.
    #
    # Two attribute paths tried before falling back to the formula:
    #   Path 1: llm.llm_engine.vllm_config.compilation_config...
    #   Path 2: llm.llm_engine.model_executor.driver_worker
    #             .model_runner.vllm_config.compilation_config...
    #           (same spine used by the generate path for model_cls,
    #            lines 2277-2279 of the driver)
    # Only if BOTH raise AttributeError: fall back to formula and log
    # ERROR (not WARNING) -- the GPU smoke is then the sole guarantee.
    # ------------------------------------------------------------------
    buf_rows: int
    _buf_rows_source: str
    try:
        buf_rows = (
            llm.llm_engine.vllm_config
            .compilation_config
            .max_cudagraph_capture_size
        )
        _buf_rows_source = "llm.llm_engine.vllm_config"
    except AttributeError:
        try:
            buf_rows = (
                llm.llm_engine.model_executor.driver_worker
                .model_runner.vllm_config
                .compilation_config
                .max_cudagraph_capture_size
            )
            _buf_rows_source = "model_runner.vllm_config"
        except AttributeError:
            # Both paths failed. Fall back to the formula but flag loudly:
            # the assert below becomes best-effort, not authoritative.
            buf_rows = min(_REPLAY_MAX_NUM_SEQS * 2, 512)
            _buf_rows_source = f"formula min({_REPLAY_MAX_NUM_SEQS}*2,512)"
            log.error(
                "C2: both vllm_config attribute paths raised AttributeError. "
                "Falling back to formula buf_rows=%d. "
                "The GPU smoke (captured_entry_count > buf_rows) is now the "
                "SOLE guarantee that expert_out_unweighted fires on prefill. "
                "Run the C2 smoke before trusting any reap/per_expert_max/"
                "output_reservoir sidecar from this replay.",
                buf_rows,
            )

    log.info(
        "C2 check: max_cudagraph_capture_size=%d (source: %s); "
        "must be >= max_num_batched_tokens=%d",
        buf_rows, _buf_rows_source, _REPLAY_MAX_BATCHED_TOKENS,
    )

    if buf_rows < _REPLAY_MAX_BATCHED_TOKENS:
        log.error(
            "C2 HARD FAIL: max_cudagraph_capture_size=%d < "
            "max_num_batched_tokens=%d. expert_out_unweighted would skip "
            "prefill chunks silently. Recovery options: "
            "(a) lower max_num_batched_tokens to %d (= buf_rows), "
            "(b) raise max_num_seqs so min(seqs*2,512) >= %d, "
            "(c) pass compilation_config with cudagraph_capture_sizes "
            "whose max >= %d. Aborting.",
            buf_rows, _REPLAY_MAX_BATCHED_TOKENS,
            buf_rows,
            _REPLAY_MAX_BATCHED_TOKENS,
            _REPLAY_MAX_BATCHED_TOKENS,
        )
        return 1

    log.info(
        "C2 check passed: buf_rows=%d >= max_num_batched_tokens=%d. "
        "expert_out_unweighted fires on all prefill chunks "
        "(vLLM V1 chunked prefill enabled by default).",
        buf_rows, _REPLAY_MAX_BATCHED_TOKENS,
    )

    # ------------------------------------------------------------------
    # 7. Writer setup
    #
    # out_path = replay_jsonl so .ckpt files land next to the input JSONL.
    # captured_done_base passed as already_done (excludes skips -- M1).
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
    #
    # Counter invariant (C-NEW-1 fix):
    #   n_skipped is incremented ONLY by the per-row `n_skipped += 1`
    #   inside the rendering loop. No other site touches it.
    #   rows_done = rows_done_base + n_replayed + n_skipped is always
    #   the correct total rows consumed from the JSONL.
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

        # Render all rows in this chunk. Per-row n_skipped += 1 is the
        # SOLE site that increments n_skipped (C-NEW-1).
        rendered_chunk: list[tuple[dict, list[int], int]] = []
        for row in chunk:
            result = _render_row_for_replay(
                row, tokenizer, args.max_model_len)
            if result is None:
                n_skipped += 1   # SOLE increment site (C-NEW-1)
                subset = str(
                    row.get("subset") or row.get("domain") or "unknown")
                skipped_by_subset[subset] = (
                    skipped_by_subset.get(subset, 0) + 1)
            else:
                tok_ids, n_tok = result
                rendered_chunk.append((row, tok_ids, n_tok))

        # After the rendering loop, n_skipped already reflects ALL skips
        # from this chunk (C-NEW-1). rows_done_base + n_replayed + n_skipped
        # is the correct JSONL position at this point.

        if not rendered_chunk:
            # All rows in this chunk were over max_model_len.
            log.warning(
                "replay chunk %d-%d: all %d rows over max_model_len=%d; "
                "skipping chunk.",
                chunk_start, chunk_start + len(chunk),
                len(chunk), args.max_model_len,
            )
            # Write checkpoint using the already-correct counters.
            # n_skipped already includes this chunk's skips from the
            # per-row loop above. DO NOT add len(chunk) again (C-NEW-1).
            _write_replay_ckpt(
                replay_ckpt,
                rows_done=rows_done_base + n_replayed + n_skipped,
                captured_done=captured_done_base + n_replayed,
            )
            continue

        # Build requests. Primary shape: prompt_token_ids dict.
        # Fallback (N1): if the pinned wheel rejects dicts, re-render
        # as string and verify round-trip token equality before submitting.
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
            # and verify round-trip token equality (no lossy decode).
            log.warning(
                "replay: LLM.generate rejected prompt_token_ids dict "
                "(%s); falling back to string rendering. "
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
        del outputs  # outputs discarded; captures live in writer state

        # Update tally accumulators.
        for row, _, n_tok in rendered_chunk:
            replayed_rows.append(row)
            replayed_token_counts.append(n_tok)
            n_replayed += 1

        total_done_captures = captured_done_base + n_replayed

        # Progress log.
        session_elapsed = time.monotonic() - t0
        log.info(
            "[%d/%d rows consumed] %d replayed, %d skipped -- "
            "%.0fs elapsed (%.2f s/replayed-row avg)",
            rows_done_base + n_replayed + n_skipped,
            rows_done_base + len(replay_rows),
            n_replayed, n_skipped,
            session_elapsed,
            session_elapsed / max(n_replayed, 1),
        )

        # B0 fail-fast (M2: gate on first actual submission only).
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
        total_rows_consumed = rows_done_base + n_replayed + n_skipped
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
        # All counters use total_done_captures (captured rows only, no skips).
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
                    _reap.dump_reap_scores_checkpoint(str(ws.reap_ckpt_path))
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
                    _s2p.dump_stage2_profile_checkpoint(str(ws.s2p_ckpt_path))
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
                    _pem.dump_per_expert_max_checkpoint(str(ws.pem_ckpt_path))
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
        # n_skipped already reflects all skips accumulated so far (C-NEW-1).
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
                "replay: %.1f%% rows skipped -- consider "
                "--max-model-len %d (corpus max ~%d tokens).",
                100.0 * skip_frac, int(max_required * 1.05), max_required,
            )
    log.info(
        "replay loop done: %d replayed, %d skipped.",
        n_replayed, n_skipped,
    )

    # ------------------------------------------------------------------
    # 11. Correctness gates
    #
    # N3 fix: generate-vs-replay routing match removed -- per-row
    # generate-time captures are not retained; the gate is infeasible.
    # Justification is the causal-masking argument in the design doc.
    # Automated gates: (a) B0 non-empty, (b) code+science tokens > 0,
    # (c) buf_rows >= max_num_batched_tokens (asserted before loop).
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
    #
    # IMATRIX IS SPECIAL:
    #   dump_imatrix(path: str, chunk_count: int)
    #   writes <jsonl>.imatrix.dat (sibling of the input JSONL, NOT
    #   under sidecars/). Requires chunk_count argument.
    #
    # ALL OTHER NINE WRITERS (uniform interface):
    #   dump_<signal>(out_path: Path)
    #   each computes sidecar_path(out_path, signal) internally ->
    #   <jsonl>.parent/sidecars/<jsonl.stem>/<signal>.pt
    # ------------------------------------------------------------------

    # -- imatrix (SPECIAL: sibling .dat + required chunk_count arg) ------
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
    # layer_input_reservoir rides inside this sidecar; no separate
    # dump needed (H2).
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
                    "< subset_size=%d -- partial subset shipped).",
                    total_done_captures, args.block_outputs_subset_size,
                )
            _bo.dump_block_outputs(out_path)
            log.info(
                "block-outputs: dumped per-layer sidecars from %d prompts",
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

## Unit Tests (unchanged, 14 tests)

**File:** `/home/lucas/ai/moe_compress/max_quality/tests/test_replay_helpers.py`

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

---

## Implementation Map (changes to existing driver)

### MODIFY: `/home/lucas/ai/moe_compress/max_quality/scripts/build_self_traces_calib_vllm.py`

**Change A** — Add `import dataclasses` to the stdlib import block (~line 82).

**Change B** — Insert after line 766: `_render_row_for_replay`, `_build_replay_subset_tally`, `_assert_code_science_nonzero`, `_write_replay_ckpt`, `_WriterState`, `_setup_all_writers` (in that order, verbatim as above).

**Change C** — Replace `main()` lines 1659–2115 (the `_check_ckpt_counter` closure + all 10 writer setup blocks) with `ws = _setup_all_writers(args, out_path, llm, tokenizer, already_done)`. Replace all 20 bare `*_ckpt_path` variable references in lines 2310–2735 with `ws.<field>`:

| Old | New |
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

**Change D** — Add `--replay-from` argparse argument after `--allow-empty-captures` (after line 1313).

**Change E** — Add `if args.replay_from is not None: return _run_replay(args)` in `main()` after line 1521.

**Change F** — Add deprecation comment in TF synthesis block (~line 2186).

**Change G** — Add `_run_replay(args) -> int` after `main()`, before `if __name__ == "__main__"`.

### CREATE: `/home/lucas/ai/moe_compress/max_quality/tests/test_replay_helpers.py`

14 tests, verbatim as above. Run: `pytest max_quality/tests/test_replay_helpers.py -v` — no GPU required.

---

## GPU Smoke (C2 primary gate, unchanged)

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

# C2: read back buf_rows via both paths.
try:
    buf_rows = (llm.llm_engine.vllm_config
                  .compilation_config.max_cudagraph_capture_size)
    print(f'buf_rows={buf_rows} (via vllm_config)')
except AttributeError:
    buf_rows = (llm.llm_engine.model_executor.driver_worker
                  .model_runner.vllm_config
                  .compilation_config.max_cudagraph_capture_size)
    print(f'buf_rows={buf_rows} (via model_runner.vllm_config)')

assert buf_rows >= 256, f'C2 FAIL: buf_rows={buf_rows} < 256'
print('C2 assert passed.')

_reap.setup(llm)
tok = llm.get_tokenizer()

text = 'The quick brown fox jumps over the lazy dog. ' * 35
tids = tok(text, add_special_tokens=False)['input_ids'][:1500]
print(f'Sequence length: {len(tids)} tokens (> buf_rows={buf_rows})')
assert len(tids) > buf_rows

sp = SamplingParams(temperature=0.0, max_tokens=1, seed=42)
out = llm.generate([{'prompt_token_ids': tids}], sp)
print('Generated token id:', out[0].outputs[0].token_ids)

n = _reap.captured_entry_count()
print(f'reap captured_entry_count: {n}')
assert n > buf_rows, (
    f'FAIL: n={n} <= buf_rows={buf_rows}. '
    'expert_out_unweighted fired only on decode, not prefill chunks.')
print(f'PASS: {n} tokens captured (> buf_rows={buf_rows}).')
"
```

**Pass:** `n > buf_rows`, no SKIPPED warning. **Fail → contingency (not planned here):** patch `TritonExperts.apply` to internally sub-slice into `_calib_buf_rows`-sized windows; requires wheel rebuild.

---

## Remaining Unverified Items (GPU-only, 2 items)

**UNVERIFIED-1:** Whether `llm.llm_engine.vllm_config.compilation_config.max_cudagraph_capture_size` or the model-runner fallback path resolves on the pinned wheel. The L-NEW-1 two-path probe with formula last-resort handles both failure modes; the ERROR log flags the operator when neither resolves.

**UNVERIFIED-2:** Whether `LLM.generate()` accepts `list[dict]` with `"prompt_token_ids"` key on the pinned wheel. The N1 fallback with round-trip token equality verification and `SystemExit(2)` on mismatch handles this without information loss.
