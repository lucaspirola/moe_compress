# Calibration writer resume + sync audit (spot-GPU readiness)

Audit date: 2026-05-29. Scope: all writers behind
`max_quality/src/moe_compress/utils/cached_calibration_signals.py::SCHEMA_VERSIONS`
plus the imatrix `.dat` export. Read-only — no code touched.

Branch: `audit/calib-resume-spot-eviction` (cut from `origin/main` @
`804b1ba fix(vllm build): apt-bootstrap curl in invocation docs`).

## Executive verdict

**YELLOW** — calibration is *operationally* resumable on spot GPUs.
The driver enforces a JSONL-rows-vs-writer-counter cross-check on
resume (F-H-6), the final sidecars are full Pattern O (atomic +
manifest-last), and 8 of 10 vLLM writers ship a two-segment
byte-equality test. **But** four classes of issue keep this from being
GREEN:

1. **Periodic checkpoints are NOT Pattern O.** All 10 vLLM `dump_*_checkpoint`
   functions use bare `tmp + torch.save + os.replace` — NO `fsync(fd)`, NO
   `fsync(parent_dir)`, NO sibling manifest. A power-loss / kernel-panic
   between the writer's last `write()` and the next pdflush cycle (5-30 s on
   ext4) can leave the new checkpoint readable but with stale/garbage blocks.
   The driver fsyncs the JSONL on the same chunk boundary, so on a "clean"
   SIGKILL the data is consistent — but the documented threat model
   (`atomic_io.py:10-19`) is broader than a vanilla SIGKILL.
2. **Operator-adds-writer-on-resume is silently undercounted.** The F-H-6
   counter cross-check fires *only when the writer's checkpoint exists*.
   If a writer is enabled mid-run (e.g. resume with `--capture-wanda-scalar-row`
   that wasn't set in the first session), the writer starts at `_n_prompts_accumulated=0`,
   JSONL has `already_done > 0`, and the final `set_n_prompts_accumulated(already_done + n_new)`
   in the dump path overwrites the accurate `n_new`-only count with the
   inflated total. The per-key `token_counts` (used as the divisor in
   normalization) are correctly `n_new`-only, but the sidecar's
   `n_prompts_accumulated` field will misreport, and any downstream
   consumer that joins on prompt-count will be wrong. No guard exists.
3. **`output_reservoir` Phase 2 resume is NOT byte-identical.** Reservoir
   sampling beyond capacity uses `torch.rand` / `torch.randint` on the
   GLOBAL torch RNG. The checkpoint does NOT serialize torch's global
   generator state. The in-patch test is `test_two_segment_additivity_fill_only`
   — explicitly the *fill-only* regime where Phase 2 is never entered.
   With `--output-reservoir-cap=256` (the default) and 6500 prompts × hundreds
   of tokens routed per (layer, expert), Phase 2 WILL trigger in production,
   and resume-vs-no-resume will differ in reservoir contents (the cap and
   `valid_count` will agree; *which* sampled tokens land in the reservoir
   will not).
4. **Two writers lack a two-segment additivity test.** `router_logits_stats`
   and `block_outputs` ship only a `test_checkpoint_round_trip` (dump-then-load
   preserves state). Neither asserts split-vs-single byte-equality. The
   per-shard accumulator semantics make additivity plausible but unverified.

None of (1)-(4) block spot-GPU usage *today* — but each is a real risk.
A 4-line fix per dump (full Pattern O), a `len(JSONL) > 0 && ckpt absent`
guard in the driver, an `output_reservoir` torch.Generator persistence
patch, and 2 missing tests would close the gap to GREEN.

## Per-signal resume-capability matrix

Legend:
- "Pattern O atomic?" = `fsync(fd) + os.replace + fsync(parent_dir)` (NOT bare `os.replace`)
- "Manifest validated load?" applies to the FINAL sidecar (the `save_*` / `load_*`
  pair in `cached_calibration_signals.py`); the `dump_*_checkpoint` files have NO
  manifest at all.
- "Byte-identical resume test?" = a test that compares `split + resume` to `single run`
  via `torch.equal` (or `torch.allclose(atol=1e-5)` for fp32 sum-of-squares).

| Signal | dump/load present? | Pattern O atomic? (periodic ckpt) | manifest-validated load? (final sidecar) | byte-identical resume test? | Severity |
|---|---|---|---|---|---|
| `phase_b` | load only (legacy compat; no production writer) | n/a | n/a (loader only) | n/a — superseded by per_expert_max + routing_stats + output_reservoir | INFO |
| `stage2_profile` | YES (`dump_stage2_profile_checkpoint`, `load_stage2_profile_checkpoint` in `vllm_calibration_stage2_profile.patch:578,624`) | NO — bare `tmp + torch.save + os.replace` (patch L620-621) | YES — `_validate_manifest_or_warn` in `cached_calibration_signals.py:1120` | YES — `test_checkpoint_resume_byte_identical_after_phase_c` in `tests/test_stage2_profile_layer_in_hook.py:282` (covers RNG-state Phase C correctness) | LOW |
| `covariance` | NO separate periodic ckpt — but Stage 3 / Stage 2 `_stage2_input_covariance.pt` writers exist | n/a | YES — `_validate_manifest_or_warn` in `cached_calibration_signals.py:1449` | n/a (no periodic ckpt) | LOW |
| `router_kd_logits` | Streaming per-attempt-idx .npz shards via `atomic_npz_save` in `build_self_traces_calib_vllm.py:424`; resume = skip already-written shards | YES — `atomic_npz_save` is full Pattern O (fsync fd + parent dir) | NO sidecar manifest (each shard is independent; loader does in-payload schema_version check at `cached_calibration_signals.py:1602`) | implicit — each shard is a self-contained atomic write, no cross-shard accumulation | LOW |
| `block_hidden` | Final sidecar only (per-layer shards via `save_block_hidden`); periodic ckpt is `dump_block_outputs_checkpoint` (covers in-memory accum across whole run) | NO — bare `tmp + torch.save + os.replace` (`vllm_calibration_hooks.patch:5501-5503`) | YES — `_validate_manifest_or_warn` per shard in `cached_calibration_signals.py:1646` | NO — `tests/test_calibration_block_outputs_smoke.py::test_checkpoint_round_trip` only round-trips state, no split-vs-single equality | MEDIUM |
| `teacher_eval` | Separate Stage 6 writer at `stage6/plugins/teacher_provider.py::_save_teacher_cache` | YES — full Pattern O (`teacher_provider.py:233-260`, `fsync(fd) + os.replace + fsync(parent_dir)`) | YES — `format_version` check in `_load_teacher_cache` | n/a (cache key is content-derived; no running accumulation) | LOW |
| `reap_scores` | YES (`dump_reap_scores_checkpoint`, `load_reap_scores_checkpoint` in `vllm_calibration_hooks.patch:8428,8446`) | NO — bare `tmp + torch.save + os.replace` (L8443-8444) | YES — `_validate_manifest_or_warn` in `cached_calibration_signals.py:1215` | YES — `test_two_segment_additivity` in `tests/test_calibration_reap_scores_smoke.py` | LOW |
| `per_expert_max` | YES (`dump_per_expert_max_checkpoint`, `load_per_expert_max_checkpoint` in `vllm_calibration_hooks.patch:8091,8109`) | NO — bare `tmp + torch.save + os.replace` | YES — `_validate_manifest_or_warn` in `cached_calibration_signals.py:1257` | YES — `test_two_segment_additivity` in `tests/test_calibration_per_expert_max_smoke.py` | LOW |
| `routing_stats` | YES (`dump_routing_stats_checkpoint`, `load_routing_stats_checkpoint` in `vllm_calibration_hooks.patch:9209,9229`) | NO — bare `tmp + torch.save + os.replace` | YES — `_validate_manifest_or_warn` in `cached_calibration_signals.py:1299` | YES — `test_two_segment_additivity` in `tests/test_calibration_routing_stats_smoke.py` | LOW |
| `router_logits_stats` | YES (`dump_router_logits_stats_checkpoint`, `load_router_logits_stats_checkpoint` in `vllm_calibration_hooks.patch:8858,8884`) | NO — bare `tmp + torch.save + os.replace` | YES — `_validate_manifest_or_warn` in `cached_calibration_signals.py:1358` | NO — only `test_checkpoint_round_trip` (state-preservation), no `test_two_segment_additivity` | MEDIUM |
| `output_reservoir` | YES (`dump_output_reservoir_checkpoint`, `load_output_reservoir_checkpoint` in `vllm_calibration_hooks.patch:7737,7761`) | NO — bare `tmp + torch.save + os.replace` | YES — `_validate_manifest_or_warn` in `cached_calibration_signals.py:1405` | PARTIAL — `test_two_segment_additivity_fill_only` is byte-identical *only* in the fill-only regime (`<= max_tokens` total per expert). Phase 2 RNG state is NOT in the checkpoint, so post-Phase-2 resume diverges. | HIGH |
| `wanda_scalar_row` | YES (`dump_wanda_scalar_row_checkpoint`, `load_wanda_scalar_row_checkpoint` in `vllm_calibration_hooks.patch:10544,10574`) | NO — bare `tmp + torch.save + os.replace` | YES — manifest-REQUIRED (no back-compat fallback, since green-field) at `cached_calibration_signals.py:1545` | YES (near-byte-equality, `atol=1e-5`) — `tests/test_stage3_wanda_scalar_row_cache.py:540+` (`test_two_segment_additivity_kill_resume`) | LOW |
| `wanda_intra_expert_score` | Final sidecar only (plugin private, `stage3/plugins/wanda_intra_expert_score.py:564`); no periodic ckpt — calibration sweep runs inside a single Stage 3 call | n/a | YES — manifest-validated load in plugin | n/a (single-process plugin sweep, no resume) | LOW |
| (imatrix .dat) | YES (`dump_imatrix_checkpoint`, `load_imatrix_checkpoint` in `vllm_calibration_hooks.patch:6767,6829`) + final `.dat` via `_write_dat` | NO — both periodic ckpt AND final `.dat` use bare `tmp + torch.save + os.replace` / `tmp + os.replace` | n/a (imatrix .dat is a llama.cpp binary format, has its own `m_last_chunk` int but no sibling manifest) | YES — `test_two_segment_additivity` in `tests/test_calibration_imatrix_smoke.py:2107` (`torch.equal`) | LOW |

## Cross-writer sync findings

### Is there a universal counter? Where?

**Yes, but it's the JSONL row count — not a shared in-memory variable.** The
driver computes `already_done` at resume by re-parsing the `.jsonl.tmp` file
(`build_self_traces_calib_vllm.py:1300-1337`) — drop trailing partial lines,
truncate, count valid JSON lines. This is the canonical "how many prompts have
been processed in this run-stream" value, and it is the ONLY value all writers
agree to via the F-H-6 guard.

Each writer keeps its own private global `_n_prompts_accumulated` in its
module scope, set by the driver via `set_n_prompts_accumulated(already_done + n_new)`
at every periodic-checkpoint boundary and at end-of-run. The F-H-6 check
(`build_self_traces_calib_vllm.py:573-620`) compares each writer's loaded
ckpt-counter to `already_done` and either hard-fails (default) or warns
(`--allow-counter-divergence`). This makes JSONL the single source of truth.

### Per-writer counter consistency

- All 10 vLLM writers update `_n_prompts_accumulated` only via the driver's
  `set_n_prompts_accumulated()` call. The writers themselves don't auto-increment.
- The driver always passes `already_done + n_new` (line ~1934, ~1954, ~1975,
  ~1998, ~2020, ~2040, ~2062, ~2084, ~2107, ~2132 for the 10 writers'
  periodic-checkpoint sites). All writers therefore see the SAME prompt count
  at any given checkpoint boundary, as long as they were all enabled.
- The per-key `token_counts` dicts (the actual divisors used in the final
  `dump_*` normalization) are accumulated independently per writer from the
  Triton/MoE callback fire counts. They are restored by `load_*_checkpoint`
  alongside the sum-of-squares / sum-of-x.
- The fsync-ordering invariant in `build_self_traces_calib_vllm.py:1856-1886`
  guarantees "JSONL is durable >= every ckpt counter" — i.e. on a kill
  between JSONL fsync and the next ckpt dump, the writer's ckpt counter
  will be *less than* `already_done` on resume, and F-H-6 will refuse to
  silently undercount.

### Risk scenarios

#### R-1 (CRITICAL): operator adds a NEW writer on resume

Scenario: first session ran `--capture-imatrix --capture-reap-scores`
and produced `out.jsonl.tmp` (1000 rows) + `out.imatrix.ckpt` + `out.reap_scores.ckpt`.
Operator restarts the GPU and adds `--capture-wanda-scalar-row` (forgot it
the first time, or schema-bump made it newly required).

What happens:
- `out.wanda_scalar_row.ckpt` does NOT exist.
- F-H-6 cross-check (`build_self_traces_calib_vllm.py:1506`) only fires
  `if args.resume and wsr_ckpt_path.exists():` — the `exists()` clause skips
  the check entirely.
- Wanda writer starts at `_n_prompts_accumulated=0`.
- Forward pass adds the remaining chunks' tokens to `_WANDA_SCALAR_ROW_SUM`
  and `_WANDA_TOKEN_COUNTS`.
- At dump time, `_wsr.set_n_prompts_accumulated(already_done + n_new)` writes
  the INFLATED count (1000 + new) into `n_prompts_accumulated`, even though
  Wanda only ever observed `n_new` chunks worth of tokens.
- The per-key `sigma_x_g_squared = sumsq / token_counts` is correct (because
  token_counts reflects what was actually seen), BUT the sidecar's
  `n_prompts_accumulated` field lies — and any downstream consumer that uses
  it for cross-writer joins, completeness checks, or schema invariants will
  get wrong answers.

Fix: the driver should refuse to start (or warn LOUDLY) when a `--capture-X`
writer is enabled, `args.resume` is set, `already_done > 0`, and the
`<jsonl>.<writer>.ckpt` is missing. Symmetric fix: emit a "fresh writer
on resume — bias note" provenance bit into the sidecar so consumers can
detect the under-coverage.

#### R-2 (HIGH): operator REMOVES a writer on resume

Scenario: first session ran `--capture-imatrix --capture-output-reservoir`,
wrote both ckpts. Operator resumes WITHOUT `--capture-output-reservoir`.
Result: `out.output_reservoir.ckpt` stays on disk and is never cleared
(the cleanup at `build_self_traces_calib_vllm.py:2307-2308` only fires
inside the dump's `if args.capture_output_reservoir:` block). On the NEXT
resume that re-enables the writer, the stale ckpt's prompt-count will
likely mismatch JSONL (which advanced in between), so F-H-6 will hard-fail
with the actionable "delete the checkpoint" message — but this is
detected at startup, after the operator has already wasted time pulling
the GPU. A startup pre-flight check would shave that cycle.

#### R-3 (HIGH): pre-power-loss kernel-panic blanks the just-renamed ckpt

Scenario: spot eviction takes the host via NMI / kernel panic between
`os.replace(tmp, ckpt_path)` and the next pdflush cycle (5-30 s on
ext4 default `dirty_writeback_centisecs=500`). Per
`utils/atomic_io.py:10-19`, this is the documented threat model. The
JSONL is fsynced — durable — but the new ckpt's data blocks may not be.
On resume: `torch.load(path)` either succeeds (the OS happened to flush
in time), fails with a deserialization error (operator must delete the
ckpt and lose the chunks-since-last-fsync), or — worst case —
deserializes garbage as a valid-looking payload (extremely unlikely with
the torch.save serialization envelope but not impossible).

Fix: full Pattern O for the periodic dumps. The 4-line change per dump:
```python
tmp = ckpt_path + ".tmp"
torch.save(payload, tmp)
fd = os.open(tmp, os.O_RDONLY); os.fsync(fd); os.close(fd)
os.replace(tmp, ckpt_path)
parent_fd = os.open(os.path.dirname(ckpt_path) or ".", os.O_RDONLY)
os.fsync(parent_fd); os.close(parent_fd)
```
(or simply call `utils.atomic_io.atomic_torch_save` — single helper, covers
all 10 writers + the imatrix `_write_dat`).

#### R-4 (MEDIUM): output_reservoir Phase 2 RNG drift

Scenario: 6500 prompts, default `--output-reservoir-cap=256`. By prompt
~500-1000, most (layer, expert) buckets are full and the reservoir
enters Phase 2 (acceptance-prob sampling). The `torch.rand(n_remaining)`
and `torch.randint(0, cap, (n_remaining,))` calls inside
`vllm/calibration_output_reservoir._on_expert_out_unweighted`
(`vllm_calibration_hooks.patch:7653-7655`) use the global torch RNG,
which is **not** part of `dump_output_reservoir_checkpoint`'s payload.

On resume: the resumed process's global RNG is re-seeded by vLLM's
startup (or untouched), drawing a different sequence than the
counterfactual no-resume run. The reservoir's *contents* diverge,
even though `total_seen`, `valid_count`, and shape all match.

Impact: Stage 1 CKADistancePlugin consumes the reservoir as a sample
distribution for CKA scoring. A diverged reservoir changes the CKA
distance numerics. Stochastically this should net out at scale, but
a determinism-required ablation sweep would not pass byte-equality.

Fix: serialize a per-(layer, expert) `torch.Generator` (mirror the
stage2_profile `_LayerInputAccumulator` pattern at
`vllm_calibration_stage2_profile.patch:614`) and use it for the Phase 2
draws. Then the in-patch test can be extended to cover Phase 2.

#### R-5 (LOW): two writers lack split-vs-single tests

`router_logits_stats` and `block_outputs` have `test_checkpoint_round_trip`
but no `test_two_segment_additivity`. Round-trip proves serialization
correctness but does NOT prove the accumulators are additive across a
kill-resume boundary (i.e. that a 3-segment-then-3-segment-with-ckpt run
produces the same final tensors as a single 6-segment run). The
accumulator semantics are simple sums-of-counts so additivity is plausible,
but the contract is unverified. A copy of the existing
`test_two_segment_additivity` template for each of these two modules
would close the gap.

## Findings (5-category)

### CRITICAL
- **C-1: Operator-adds-writer-on-resume silently undercounts.** See R-1.
  The driver's `if args.resume and <writer>_ckpt_path.exists():` guard
  (10× repeated) skips the F-H-6 cross-check whenever a writer was added
  fresh on resume. The sidecar's `n_prompts_accumulated` field will be
  inflated relative to actual coverage. Per-key `token_counts` (the
  divisors) are correct, so the *math* is correct over the chunks
  observed, but the *metadata* lies and downstream consumers may
  mis-attribute the sidecar's coverage. This is a single-line guard fix:
  `if args.resume and args.capture_X: refuse if already_done > 0 and not ckpt.exists()`.

### HIGH
- **H-1: All 10 vLLM periodic checkpoints + imatrix `.dat` final write
  skip fsync.** Both `dump_*_checkpoint` (10 sites) and `_write_dat`
  (1 site) write `tmp + torch.save + os.replace` with no `fsync(fd)` or
  `fsync(parent_dir)`. The 4-line Pattern O dance lives at
  `utils/atomic_io.py:204-229` (`atomic_torch_save`) and is already used
  by every final sidecar in `cached_calibration_signals.py`. Drop-in
  replacement at 11 sites. Without this, the documented power-loss /
  kernel-panic threat model in `atomic_io.py:10-19` is not covered for
  periodic checkpoints — the JSONL/ckpt counter invariant survives a
  clean SIGKILL but not a pre-pdflush eviction.
- **H-2: output_reservoir Phase 2 RNG state is NOT in the checkpoint.**
  See R-4. With default `--output-reservoir-cap=256` and 6500 prompts,
  Phase 2 dominates the late chunks. Resume-vs-no-resume produces
  different reservoir contents. The in-patch test
  (`test_two_segment_additivity_fill_only`) explicitly limits itself
  to the fill-only regime and documents this. Fix is a 6-line
  per-(layer, expert) `torch.Generator` mirror of the stage2_profile
  `_LayerInputAccumulator` pattern.

### MEDIUM
- **M-1: router_logits_stats + block_outputs lack two-segment additivity
  tests.** See R-5. The accumulators are sum-counting, so additivity is
  plausible, but the contract is unverified.
- **M-2: Stale ckpts from a removed writer are never swept on a later
  resume.** See R-2. If the operator drops `--capture-X` on a resume and
  then re-enables it on a later resume, the original stale ckpt may
  pass the prompt-counter check by luck, or hard-fail at startup. Either
  outcome wastes a GPU cycle. A startup pre-flight sweep over all
  10 ckpt paths that warns when a ckpt exists for a writer NOT in the
  current flag-set would help operators catch this.
- **M-3: F-H-6 hard-fail message points operators to delete the
  checkpoint** (`build_self_traces_calib_vllm.py:615-619`) — which means
  the writer restarts accumulation from zero, BUT the JSONL has already
  been written. The writer will then accumulate over the **remaining**
  chunks only. The driver-level dump correctly uses `set_n_prompts_accumulated(already_done + n_new)`,
  but again — the metadata over-states coverage. This is the same
  metadata-vs-math gap as C-1.

### LOW
- **L-1: Stage 5 router_kd `_save_stage5_checkpoint` IS full Pattern O**
  (`router_kd/orchestrator.py:1053-1066`). Stage 6 teacher_eval cache IS
  full Pattern O (`stage6/plugins/teacher_provider.py:233-260`). The
  asymmetry between the well-engineered Stage 5/6 saves and the
  bare-`os.replace` vLLM-side saves is a maintenance hazard — operators
  reading one set of code will assume the other behaves the same.
- **L-2: imatrix periodic checkpoint dedup-checks for `_CAPTURE_IMATRIX`
  but final `.dat` does not check `_n_prompts_accumulated > 0`.** A
  zero-prompt run will write a header-only `.dat` — harmless but the
  log-line claims "wrote N entries". No correctness impact.
- **L-3: `wanda_scalar_row` final-sidecar load is manifest-REQUIRED**
  (no back-compat fallback at `cached_calibration_signals.py:1545`).
  This is intentional (green-field signal) and tighter than the other
  9 sidecars. Worth documenting as the new standard.

### NITPICK
- **N-1: `_check_ckpt_counter` is a closure that captures `already_done`
  and `args.allow_counter_divergence`** from `main()` (build_self_traces_calib_vllm.py:1369-1388),
  but the actual check fn `_ckpt_counter_check` is module-scope and
  takes those as args. The closure is preserved only for backward-compat
  with existing call sites — fine, but adds a hop that could be inlined
  for clarity now that the module-level fn exists.
- **N-2: F-H-6 message text on hard-fail mentions "JSONL has {already_done} rows"
  but doesn't print the file path** — operator has to find it in the
  preceding INFO log. Including `tmp_path` in the message would shave
  10 seconds off the recovery.
- **N-3: The `--allow-counter-divergence` escape hatch is documented as
  "ablation sweeps where minor under-counting is tolerable"** (line 697)
  — but the *what* of the under-counting (the writer integrates over a
  subset of the calibration data, biasing toward later chunks) is not
  spelled out. Worth a sentence in the help text.

## Recommended fixes

In rough priority order:

1. **C-1 fix (1 line × 10 sites):** in `build_self_traces_calib_vllm.py`,
   for each `if args.resume and <ckpt>.exists():` block, add a sibling
   `elif args.resume and args.capture_X and already_done > 0:`
   that hard-fails with "writer X enabled mid-resume with no prior
   checkpoint AND JSONL has rows from a prior session — coverage would
   silently underreport; pass --allow-counter-divergence to override".
2. **H-1 fix (1 line × 11 sites):** replace every periodic-ckpt
   `tmp = path + ".tmp"; torch.save(payload, tmp); os.replace(tmp, path)`
   with `from moe_compress.utils.atomic_io import atomic_torch_save;
   atomic_torch_save(path, payload)`. The 11 sites are all 10 vLLM
   writers' `dump_*_checkpoint` + imatrix `_write_dat`. Note: the
   vLLM writers run inside a patched vLLM wheel — they can't import
   from `moe_compress.utils.atomic_io` at module-import time without
   creating a circular dep. The clean answer is to copy the 4-line
   inline fsync dance into each writer (or vendor the helper into
   `vllm.calibration_hooks`).
3. **H-2 fix (~10 lines):** mirror the `_LayerInputAccumulator` torch.Generator
   pattern from `vllm_calibration_stage2_profile.patch:608-617` into
   `vllm.calibration_output_reservoir`. Replace global `torch.rand` /
   `torch.randint` with `torch.rand(n, generator=g)` /
   `torch.randint(0, cap, (n,), generator=g)` where `g` is a per-(rank, e)
   torch.Generator. Serialize `g.get_state()` in the checkpoint payload.
4. **M-1 fix (~50 lines × 2):** copy the `test_two_segment_additivity`
   template from `test_calibration_reap_scores_smoke.py` into
   `test_calibration_router_logits_stats_smoke.py` and
   `test_calibration_block_outputs_smoke.py`.
5. **M-2 fix (~20 lines):** at driver startup, after `already_done` is
   computed, sweep `out_path.parent` for `<jsonl_stem>.<known_writer>.ckpt`
   files where the corresponding `args.capture_<writer>` flag is False;
   log WARNING listing each stale ckpt and recommending the operator
   delete it.
6. **N-1, N-2, N-3:** docstring/log-message touch-ups, no functional change.

If C-1 + H-1 + H-2 land, the verdict moves to GREEN.

## Wheel-rebuild implication

The vLLM wheel currently building (per the user's note) bundles
`vllm_calibration_hooks.patch` + `vllm_calibration_stage2_profile.patch`.
H-1 (Pattern O for periodic ckpts) and H-2 (output_reservoir RNG)
both require **patch changes** — they cannot be back-fitted from
outside the wheel. C-1 and M-1/M-2 live in the driver script
(`max_quality/scripts/build_self_traces_calib_vllm.py`) and tests
(`max_quality/tests/`) which are NOT inside the wheel — they ship
with the repo. So:

- If the wheel build proceeds as-is, the spot-resume risk surface
  is exactly what's documented above (YELLOW). Operators can run
  calibration; eviction-recovery works in the common case; the
  enumerated holes remain.
- If the wheel build is paused for H-1 + H-2 patches, the verdict
  moves to "GREEN once C-1 + M-1 + M-2 driver/test fixes ship".

The user makes the call. The audit's recommendation is to **NOT block
the wheel** on the audit findings — production calibration is
operationally sound today, and the H-class fixes can land in a
follow-up wheel rev once the current run produces a baseline.

## Evidence

### Pattern O is well-defined and used for finals
`max_quality/src/moe_compress/utils/atomic_io.py:204-229` (`atomic_torch_save`):
```python
def atomic_torch_save(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    try:
        torch.save(payload, tmp)
        durable_rename(tmp, path)         # <- fsync(fd) + replace + fsync(dir)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return path
```

### Final sidecars use it (all 10 in cached_calibration_signals)
`max_quality/src/moe_compress/utils/cached_calibration_signals.py:302-337`
(`_write_payload_and_manifest`): unlink stale manifest, atomic_torch_save
payload, write_manifest_last manifest. This is the per-signal "production"
sidecar path.

### Periodic checkpoints DO NOT use it
`max_quality/patches/vllm_calibration_hooks.patch:5501-5503` (block_outputs):
```
+    tmp = ckpt_path + ".tmp"
+    torch.save(payload, tmp)
+    os.replace(tmp, ckpt_path)
```
Identical bare pattern at L6824-6826 (imatrix), L7314-7315 (input_cov),
L7756-7758 (output_reservoir), L8106-8108 (per_expert_max),
L8443-8444 (reap_scores), L8881-8882 (router_logits_stats),
L9226-9228 (routing_stats), L10572-10573 (wanda_scalar_row),
and `vllm_calibration_stage2_profile.patch:620-621` (stage2_profile).
None of the 10 includes `fsync(fd)` or `fsync(parent_dir)`.

### JSONL fsync establishes durability ordering before ckpt dump
`max_quality/scripts/build_self_traces_calib_vllm.py:1867-1869`:
```python
try:
    f.flush()
    os.fsync(f.fileno())
except OSError as exc:
    ...
```
followed by the per-writer periodic ckpt dumps. The in-line comment at
L1856-1866 documents the invariant "JSONL is durable >= ckpt counter".

### F-H-6 cross-check is conditional on ckpt existence
`max_quality/scripts/build_self_traces_calib_vllm.py:1404-1415` (imatrix
example, 10 sites identical):
```python
if args.resume and imatrix_ckpt_path.exists():
    try:
        loaded_prompts = _im.load_imatrix_checkpoint(...)
        _check_ckpt_counter("imatrix", loaded_prompts, imatrix_ckpt_path)
        ...
```
The `_check_ckpt_counter` body at L598-620 raises ValueError when
`loaded_prompts != already_done`. The guard skips entirely when the ckpt
is missing — this is the C-1 hole.

### output_reservoir Phase 2 uses GLOBAL torch RNG
`max_quality/patches/vllm_calibration_hooks.patch:7643-7661` (the only
RNG calls inside the `_on_expert_out_unweighted` write path are inside
the `n_remaining > 0` Phase-2 branch):
```
+            u = torch.rand(n_remaining)
+            ...
+            slots = torch.randint(0, cap, (n_remaining,))
```
No `generator=` arg. `dump_output_reservoir_checkpoint` at L7745-7755
serializes `reservoir`, `seen`, `valid_count`, `layer_id_to_rank`,
prompt-counter — no torch RNG state.

### Stage 5 and Stage 6 ARE Pattern O (asymmetry)
`max_quality/src/moe_compress/router_kd/orchestrator.py:1053-1066`:
```python
tmp = partial_dir / f"step_{step}.pt.tmp"
final = partial_dir / f"step_{step}.pt"
torch.save(payload, tmp)
fd = os.open(str(tmp), os.O_RDONLY); os.fsync(fd); os.close(fd)
os.replace(tmp, final)
parent_fd = os.open(str(final.parent), os.O_RDONLY); os.fsync(parent_fd); os.close(parent_fd)
```
`max_quality/src/moe_compress/stage6/plugins/teacher_provider.py:233-260`
implements the same dance for `teacher_eval_cache.json`. These two
non-vLLM writers are the spec — the vLLM writers should match.

### Tests
- `tests/test_stage2_profile_layer_in_hook.py:282 test_checkpoint_resume_byte_identical_after_phase_c`
  — full RNG-state byte-equality, exercises Phase C entry.
- `tests/test_stage3_wanda_scalar_row_cache.py:540+ test_two_segment_additivity_kill_resume`
  — near-byte-equality (atol=1e-5) for fp32 sum-of-squares.
- `test_calibration_imatrix_smoke.py:2107`, `test_calibration_reap_scores_smoke.py`,
  `test_calibration_input_cov_smoke.py`, `test_calibration_per_expert_max_smoke.py`,
  `test_calibration_routing_stats_smoke.py` — `test_two_segment_additivity`
  with `torch.equal`.
- `test_calibration_output_reservoir_smoke.py` — `test_two_segment_additivity_fill_only`
  (limited to fill-only regime, per its docstring).
- `test_calibration_router_logits_stats_smoke.py`, `test_calibration_block_outputs_smoke.py`
  — only `test_checkpoint_round_trip`. NO two-segment additivity test.

### F-H-6 message wording
`max_quality/scripts/build_self_traces_calib_vllm.py:615-619`:
```python
raise ValueError(
    f"{msg} Delete the checkpoint file ({ckpt_path}) so the "
    "accumulator restarts from zero and re-walks the prompts "
    "from this run's resume base, OR re-run with "
    "--allow-counter-divergence to tolerate the under-count."
)
```
Deleting the ckpt loses the prior accumulator state — the writer
re-walks only the chunks remaining in `prompts[already_done:]`, so its
final coverage is `n_new` prompts, NOT `already_done + n_new`. The
metadata at dump time will still claim `already_done + n_new`. Same
metadata-vs-math gap as C-1 / M-3.
