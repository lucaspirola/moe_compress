# PLAN — Stage-4 EoRA opt A1: gate the log-only residual-norm recompute

Branch: `plan/stage4-a1` (from main c6ff6f9)
Scope: ~15-line change. Plan only — do NOT implement.
File under change: `max_quality/src/moe_compress/stage4/plugins/eora_compensation.py`
Tests: `max_quality/tests/test_stage4_multigpu.py` (+ golden snapshot)

## 0. Cite re-verification (done against CURRENT code)

All load-bearing cites confirmed in this checkout. Corrected line drift vs. the analysis narrative:

| Item | Narrative said | ACTUAL (verified) |
|---|---|---|
| `res_before` | :440 | **:440** ✓ |
| `res_after` (extra `Uc@Vc` + `[d_out,d_in]` sub + `.norm()**2`) | :460-462 | **:460-462** ✓ |
| `_compute_eora_factors` call (residual computed strictly AFTER) | :455 | **:455** ✓ (def at :235) |
| `_solve_expert_tile` | "def :691" | **def :389-403**; :691 is the CALL site ✓ |
| residual return tuple | — | `return ... :463` — 6-tuple `(Uc, Vc, int(take_eff), res_before, res_after, gate_spectrum_out)` |
| accumulators | :707-708 | **:707-708** ✓ |
| per-matrix `.item()` sync | :726-727 | **:726-727** ✓ |
| log.info residual line | :731-733 | **:731-733** ✓ |
| `_trackio_log` residual keys | :744-745 | **:744-746** (3 keys: before, after, **rel_drop**) |

**KEY CORRECTION to the threading premise:** residuals are computed ENTIRELY inside `_solve_expert_tile`. `_compute_eora_factors` (:235) returns only `(Uc, Vc, take_eff)` and never touches residuals. Therefore the flag threads into `_solve_expert_tile` ONLY — `_compute_eora_factors` needs **no** change.

**Quality-neutrality (verified):** `res_before`/`res_after` flow ONLY into `res_before_acc`/`res_after_acc` (:707-708) → `.item()` (:726-727) → `log.info` (:731-733) + `_trackio_log` (:744-746). They are NEVER read by `widen_rank` (:722), `rank_map` (:723), `U_corr`/`V_corr` (:703-704), `eff_per_expert` (:705), or the double-widen assert (:716). `res_after` is computed AFTER `Uc,Vc` are finalized (:455 → :460), so gating skips zero productive work. Golden = U/V/ranks, residual-independent ⇒ default-off is byte-identical.

## 1. The config flag

**Name:** `stage4_eora.log_residuals` (bool, DEFAULT `False`).

Rationale: stage4 config is read as `config["stage4_eora"]` at **:587** (`s4 = config["stage4_eora"]`); existing budget key is `stage4_eora.compensation_budget_pct` (:516). New flag lives in the same `stage4_eora` sub-dict. Default false ⇒ production realizes the speedup automatically; golden unaffected.

**Read site:** in `compensate_layer`, right after :587, add:
```python
log_residuals = bool(s4.get("log_residuals", False))
```
Using `s4.get(..., False)` makes it backward-compatible with every config that omits the key. No new ctx slot, no new `reads` tuple entry (:520-526) — the flag rides inside the already-declared `"config"` slot.

## 2. The exact code change — control flow (on AND off)

Chosen design: **skip everything when off** (cleanest — no sentinel arithmetic, no misleading zeros, no NaN to the dashboard). The 6-tuple shape is PRESERVED (slots 3,4 become `None` when off) so callers/tests that do `Uc, Vc, te, *_ = out` and index `out[5]` keep working.

**Sentinel choice:** `None`. The off-path never reaches the accumulators/sync/log (guarded), so the sentinel is never arithmetically consumed; `None` matches the existing `gate_spectrum_out=None` convention in this same tuple.

### 2a. `_solve_expert_tile` — signature (def :389-403)
```python
def _solve_expert_tile(
    name, e, layer_idx, W_orig, U_e, V_e, A, d_in, r_per_expert,
    target_device, a_storage_dtype,
    *,
    gate_spectrum=None,
    log_residuals: bool = False,   # NEW — default off
):
```

### 2b. body — guard the two residual computes
`res_before` at **:440**:
```python
    res_before = delta.norm() ** 2 if log_residuals else None
```
`res_after` at **:460-462**:
```python
    res_after = (
        (delta - (Uc.to(torch.float32) @ Vc.to(torch.float32))).norm() ** 2
        if log_residuals else None
    )
```
Return tuple (:463) unchanged in shape — `res_before`/`res_after` are `None` when off.

### 2c. call site (:691-695) — pass the flag
```python
    Uc, Vc, take_eff, res_before, res_after, gate_spec_out = _solve_expert_tile(
        name, e, ref.layer_idx, W_orig, U_e, V_e, A, d_in,
        r_per_expert, tgt, a_storage_dtype,
        gate_spectrum=gate_spectra.get(e),
        log_residuals=log_residuals,   # NEW
    )
```

### 2d. accumulators (:706-708) — guard
```python
    if log_residuals:
        res_before_acc += res_before.to(dev)
        res_after_acc += res_after.to(dev)
    n_eligible += 1
```
(`n_eligible` stays UNCONDITIONAL — it feeds `eff_rank_*` aggregates and the eligibility count, not residuals.)

### 2e. per-matrix sync + log + trackio residual keys (:726-746) — guard
```python
    residual_fields = {}
    if log_residuals:
        res_before_sum = float(res_before_acc.item())
        res_after_sum = float(res_after_acc.item())
        res_before = (res_before_sum / max(n_eligible, 1)) ** 0.5
        res_after = (res_after_sum / max(n_eligible, 1)) ** 0.5
        rel_drop = (res_before - res_after) / max(res_before, 1e-12)
        log.info("  L%d/%s widened by r=%d → new rank=%d; "
                 "residual %.4e→%.4e (-%.1f%%)",
                 ref.layer_idx, name, r_per_expert, fe.ranks[name],
                 res_before, res_after, 100 * rel_drop)
        residual_fields = {
            f"stage4/{name}_residual_unweighted_before": res_before,
            f"stage4/{name}_residual_unweighted_after": res_after,
            f"stage4/{name}_residual_unweighted_rel_drop": rel_drop,
        }
    else:
        log.info("  L%d/%s widened by r=%d → new rank=%d",
                 ref.layer_idx, name, r_per_expert, fe.ranks[name])
```
Then in the `_trackio_log({...})` call (:735-758), REPLACE the three inline residual keys (:744-746) with `**residual_fields` (empty dict ⇒ no residual keys emitted when off). All other trackio keys (layer_idx, ranks, eff_rank_*, compensated_params, n_eligible_experts) stay unconditional. Satisfies risk item 6: trackio never receives `NaN`, log never formats a `None`.

### Net diff size
~15 changed lines, single file. No change to `_compute_eora_factors`, no change to `reads`/`writes`/ctx slots, no orchestrator file change (flag rides in `config`).

## 3. Threading summary
- Flag read once per layer from `config["stage4_eora"]` (after :587).
- Passed as keyword arg `log_residuals=` into `_solve_expert_tile` at the call site (:691). Pure function arg — the task-parallel band fan-out (:677-695) carries it identically to every band; each invocation gets the same bool. No per-band/IPC plumbing (captured closure local, same as `r_per_expert`).
- `_compute_eora_factors`: **NOT threaded** (computes no residuals — verified).

## 4. Quality-neutrality argument
The golden artifact is the widened U/V tensors + `rank_map`, derived from `Uc`/`Vc`/`take_eff` (:455) + `widen_rank` (:722) — all residual-independent. `res_before`/`res_after` are consumed ONLY by `log.info` (:731-733) and `_trackio_log` (:744-746). Default-off produces byte-identical U/V/ranks to today's default-on-but-discarded output. ONLY behavioral change: the dashboard/log lose the residual metrics unless `stage4_eora.log_residuals: true`.

## 5. Tests (in `max_quality/tests/test_stage4_multigpu.py`)

**(a) Golden snapshot byte-identical (default off).** Re-run the Stage-4 golden snapshot test with no config change (flag defaults false), assert snapshot unchanged. (Residuals aren't in the golden, so it must pass — assert explicitly. Identify via `grep -rn "snapshot\|golden" max_quality/tests/test_stage4*`.)

**(b) New unit `test_solve_expert_tile_residual_flag`:**
```python
out_off = _solve_expert_tile(..., gate_spectrum=None, log_residuals=False)
out_on  = _solve_expert_tile(..., gate_spectrum=None, log_residuals=True)
assert torch.equal(out_off[0], out_on[0])   # Uc
assert torch.equal(out_off[1], out_on[1])   # Vc
assert out_off[2] == out_on[2]              # take_eff
assert out_off[3] is None and out_off[4] is None
assert out_on[3] is not None and out_on[4] is not None
assert float(out_on[3]) >= 0.0 and float(out_on[4]) >= 0.0
```
Reuse the `test_solve_expert_tile_pure` fixture tensors (`d_out,d_in,r = 12,8,2`, seeded). Proves productive output is flag-independent AND slots 3/4 populated-vs-skipped.

**(c) Compatibility / no stale assertion.** Confirm `test_solve_expert_tile_pure` still passes (uses `Uc, Vc, te, *_ = out` and `out[5]`, neither touches slots 3/4). Grep proves no existing test asserts residuals non-sentinel:
```
grep -rn "res_before\|res_after\|residual_unweighted" max_quality/tests/
```
VERIFIED at plan time: only an unrelated `..._before_teardown` function-name match; no test reads slots 3/4 as non-None. Re-run at impl time.

## 6. Risk / edge-cases
- **Multi-GPU task-parallel:** flag is a pure arg at the single call site (:691) into every band; no IPC plumbing.
- **Trackio NaN:** eliminated — off-path `residual_fields = {}`, no residual key (never NaN/0.0).
- **Unconditional log format:** residual `log.info` (:731-733) moved inside `if log_residuals:`; off-path uses a residual-free line. No format string receives `None`.
- **Accumulators init but unused when off:** `res_before_acc`/`res_after_acc` stay zero-init, never `.item()`-synced when off — harmless, avoids the per-matrix device→host sync (secondary speedup).
- **`n_eligible` unconditional** — feeds non-residual aggregates and `/max(n_eligible,1)` guards.
- **`res_before`/`res_after` local rebind at :728-729:** under off-path never assigned (block guarded), never read after (only readers are inside guarded blocks) — no `UnboundLocalError`. Confirm by reading :728-758 at impl time.

## 7. Build sequence
- [ ] Edit `_solve_expert_tile` signature: add `log_residuals: bool = False` (kw-only). (§2a)
- [ ] Guard `res_before` (:440) and `res_after` (:460-462) with `if log_residuals else None`. (§2b)
- [ ] Read flag: `log_residuals = bool(s4.get("log_residuals", False))` after :587. (§1)
- [ ] Pass `log_residuals=log_residuals` at call site (:691). (§2c)
- [ ] Guard accumulators (:707-708) under `if log_residuals:`; keep `n_eligible` unconditional. (§2d)
- [ ] Guard sync+log+trackio residual keys (:726-746); build `residual_fields`, splice `**residual_fields`. (§2e)
- [ ] Add unit `test_solve_expert_tile_residual_flag`. (§5b)
- [ ] Re-run grep (§5c) to re-confirm no test reads residual slots as non-sentinel.

## 8. Test plan (commands)
- [ ] `PYTHONPATH=src python3 -m pytest tests/test_stage4_multigpu.py -q` (pure + new flag test green)
- [ ] Stage-4 golden snapshot test green, snapshot byte-identical (§5a)
- [ ] `grep -rn "res_before\|res_after\|residual_unweighted" tests/` → only the unrelated match
- [ ] Full stage4 suite: `PYTHONPATH=src python3 -m pytest tests/ -k stage4 -q`
