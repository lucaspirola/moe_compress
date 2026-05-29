# Plan: H-1 (fsync periodic checkpoints) + H-2 (output_reservoir Phase-2 Generator state) — single patch rev

## Audit refs
- Branch: `audit/calib-resume-spot-eviction` @ `5e485f6`
  - Doc: `tasks/AUDIT_CALIB_RESUME_SPOT_EVICTION.md`
  - Verdict: YELLOW; **H-1 + H-2 are the patch-side blockers to GREEN.**
- This plan branch: `plan/calib-h1-h2-fsync-rng` (cut from `origin/main`).

## Design overview

H-1 and H-2 are both **`max_quality/patches/vllm_calibration_hooks.patch`** changes (with one site
in `vllm_calibration_stage2_profile.patch` for H-1). They ship in ONE coordinated
patch rev → one wheel rebuild → one MANIFEST.md bump. The hooks-patch hunk inside
each writer is independent of every other writer's hunk, so the change is a 11-site
mechanical sweep for H-1 plus a localized 3-site change (init/dump/load) for H-2.

### Why a single coordinated rev
- The wheel is the boundary: both fixes have to land inside the same `.whl` because
  the patched `vllm/calibration_*.py` files are vendored into the wheel.
- MANIFEST.md (line count + MD5) is updated **once** at the end after both fixes
  are in the same patch file.
- Test additions for both go into the patches' `tests/test_calibration_*_smoke.py`
  test files (also vendored into the vLLM tree by these same patches).

### Key constraint surfaced during planning (REQUEST FOR DECISION)

Two pre-existing tests in this codebase use `unittest.mock.patch.object` /
`monkeypatch.setattr` against production module attributes to verify call ORDER:

- `max_quality/tests/test_utils_atomic_io.py:212-214` — patches
  `aio._fsync_file`, `aio.os.replace`, `aio._fsync_dir` to assert
  `fsync_file → replace → fsync_dir` sequence.
- `max_quality/tests/test_stage3_wanda_scalar_row_cache.py:189` —
  patches `_aio.os.replace` to assert payload-rename-before-manifest-rename.

The standing project rule is **no-monkey-patches against production code**
(memory: `feedback_no_monkey_patches.md`, 2026-05-29). The cleanest
no-monkey-patch idiom for verifying that the fsync calls are PRESENT in the
new writers is **AST/source inspection**: `inspect.getsource(writer.dump_X_checkpoint)`
must contain `os.fsync`, and a functional round-trip test on a real (tmpfs is fine)
filesystem confirms no exception is raised. This does NOT verify the call ORDER
the way the existing `mock.patch.object` tests do.

**RAISING per `feedback_raise_dont_substitute.md`**: do you want me to

  **(a)** Match the existing pre-existing test style (`mock.patch.object` to spy
        the fsync/replace sequence — consistent with the file but reinforces
        the prohibited pattern), or

  **(b)** Use AST + functional round-trip only (strictly no-monkey-patch but
        weaker — proves the fsync calls EXIST in the source and the dump
        produces a readable file, but does not prove call order against a
        reordered re-write), or

  **(c)** Use a `strace -e trace=fsync,renameat,renameat2 python -c …` fixture
        (most rigorous — observes real syscalls — but new dependency on
        strace at test-runtime and Linux-only)?

The plan below ASSUMES option (b) until the user picks. Option (a) would be
~10 LoC simpler per test; option (c) would add a `pytest.importorskip("strace")`
or shell-out fixture.

---

## H-1 design

### Inline Pattern O helper per writer

The `vllm/calibration_*.py` files run **inside the patched vLLM wheel**.
They CANNOT import `moe_compress.utils.atomic_io` at module-import time
(creates a circular dep: `moe_compress.stage3.plugins.*` imports vLLM
during routing, vLLM importing `moe_compress.utils.atomic_io` at boot
would close the cycle). The audit recommends inlining the dance.

Each writer already imports `os` + `torch` at module-level (verified
in patch — see imports at L47-52, L1319-1326, etc.). Add ONE module-private
helper at the top of each writer's "checkpoint primitives" section:

```python
def _atomic_torch_save(payload, path: str) -> None:
    """Pattern O atomic + durable: tmp → fsync(fd) → os.replace → fsync(parent_dir).

    Mirrors ``moe_compress.utils.atomic_io.atomic_torch_save``. Inlined here
    because vLLM's runtime modules cannot import from ``moe_compress`` at
    module-load time (circular dep). Best-effort fsync semantics — non-POSIX
    filesystems may EINVAL the parent-dir fsync; swallowed with a debug log.
    """
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    parent = os.path.dirname(path) or "."
    try:
        parent_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError:
        # Non-POSIX FS (tmpfs in CI, FUSE on HF Jobs) may not support
        # fsync on a directory fd; swallow per atomic_io._fsync_dir.
        pass
```

Then replace every periodic-ckpt `tmp + torch.save + os.replace` site with
`_atomic_torch_save(payload, ckpt_path)`. The 3-line write site collapses to 1.

### Why inline-per-writer (not a shared helper inside vllm/calibration_hooks.py)

The 10 writers + imatrix do NOT all import from `vllm.calibration_hooks` (some
are standalone modules registered via the chained-callback registry). To avoid
forcing 10 new cross-module imports inside a wheel-vendored writer set, **copy
the helper verbatim into each writer**. The 10-line LoC duplication is the
cost; the upside is each writer is self-contained for vendoring + maintenance.

The audit acknowledges this trade at `tasks/AUDIT_CALIB_RESUME_SPOT_EVICTION.md:329-331`
("The clean answer is to copy the 4-line inline fsync dance into each writer
(or vendor the helper into `vllm.calibration_hooks`)").

**Alternative considered + rejected**: putting `_atomic_torch_save` in
`vllm/calibration_hooks.py` and importing it from the 9 other writers. Rejected
because vllm/calibration_hooks.py is the "registry" module that the writers
already wire INTO; reversing the dep direction would re-open the same circular
risk that motivated inlining away from `moe_compress.utils.atomic_io`.

### Imatrix `.dat` special case

`_write_dat` in `vllm/calibration_imatrix.py` (patch L6711-6725) is a CUSTOM
binary writer (not `torch.save` — it builds a llama.cpp-format `.dat` via
`struct.pack` inside `with open(tmp, "wb") as f`). The Pattern O dance for this
site is slightly different:

```python
# inside the existing `with open(tmp, "wb") as f:` block, BEFORE close:
f.flush()
os.fsync(f.fileno())
# `with` block exits, closing fh
os.replace(tmp, path)
parent = os.path.dirname(path) or "."
try:
    parent_fd = os.open(parent, os.O_RDONLY)
    try: os.fsync(parent_fd)
    finally: os.close(parent_fd)
except OSError: pass
```

Define a second module-private helper `_durable_close_replace_dat` inside
`vllm/calibration_imatrix.py` to keep `_write_dat` readable (mirrors the
`_durable_close_and_replace + _finalize_atomic_write` pair in
`atomic_io.py:162-200`).

### Site inventory (11 sites, all in 2 patch files)

| # | Patch file | Writer module | Line range (current) | Pattern in patch | Replacement |
|---|---|---|---|---|---|
| 1 | hooks | `vllm/calibration_block_outputs.py` | 5501-5503 | `tmp + torch.save + os.replace` | `_atomic_torch_save(payload, ckpt_path)` |
| 2 | hooks | `vllm/calibration_imatrix.py` (periodic ckpt) | 6824-6826 | same | `_atomic_torch_save(payload, path)` |
| 3 | hooks | `vllm/calibration_imatrix.py` (`_write_dat` final `.dat`) | 6711-6725 | `with open(tmp,'wb') as f: ...; os.replace(tmp, path)` | inline `f.flush() + os.fsync(f.fileno())` then `os.replace + fsync_parent` (helper `_durable_close_replace_dat`) |
| 4 | hooks | `vllm/calibration_input_cov.py` | 7313-7314 | `tmp + torch.save + os.replace` | `_atomic_torch_save(payload, ckpt_path)` |
| 5 | hooks | `vllm/calibration_output_reservoir.py` | 7757-7758 | same | `_atomic_torch_save(payload, ckpt_path)` |
| 6 | hooks | `vllm/calibration_per_expert_max.py` | 8105-8106 | same | `_atomic_torch_save(payload, ckpt_path)` |
| 7 | hooks | `vllm/calibration_reap_scores.py` | 8442-8443 | same | `_atomic_torch_save(payload, ckpt_path)` |
| 8 | hooks | `vllm/calibration_router_logits_stats.py` | 8880-8881 | same | `_atomic_torch_save(payload, ckpt_path)` |
| 9 | hooks | `vllm/calibration_routing_stats.py` | 9225-9226 | same | `_atomic_torch_save(payload, ckpt_path)` |
| 10 | hooks | `vllm/calibration_wanda_scalar_row.py` | 10570-10571 | same | `_atomic_torch_save(payload, ckpt_path)` |
| 11 | stage2_profile | `vllm/calibration_stage2_profile.py` | 620-621 | same | `_atomic_torch_save(state_payload, path)` (helper inlined in same file) |

Note: `vllm/calibration_hooks.py` itself (the registry module) does NOT have its
own periodic checkpoint dump (verified — only `dump_*_checkpoint` calls live in
the per-signal writers). So 10 writers + 1 imatrix `.dat` + 1 stage2_profile = 11
sites = 11 calls to `_atomic_torch_save` (the imatrix `.dat` uses the second
helper variant for the open-write path). Total: **10 writer modules + 1 stage2
module get the `_atomic_torch_save` helper added; 1 module (imatrix) gets BOTH
helpers because of the custom binary `.dat` writer.**

### Tests (H-1)

Per-writer smoke test files already exist inside the hooks patch
(`tests/test_calibration_*_smoke.py`). For each of the 10 writers + imatrix
`.dat`, add ONE new test:

```python
def test_periodic_ckpt_uses_fsync_pattern_o(tmp_path):
    """H-1: dump_<writer>_checkpoint must use the Pattern O fsync dance.

    Verifies (assuming option (b) from the plan's REQUEST FOR DECISION):
    1. inspect.getsource(writer.dump_X_checkpoint) contains 'os.fsync'
       AND 'os.replace' (proves the fsync is in the source path).
    2. A functional round-trip (dump → file exists → torch.load → fields
       match) succeeds and the .tmp file does NOT linger.
    """
    import inspect
    from vllm import calibration_X as writer
    src = inspect.getsource(writer.dump_X_checkpoint)
    assert "os.fsync" in src, (
        "dump_X_checkpoint must inline Pattern O fsync; got source:\n" + src
    )
    assert "os.replace" in src

    # Functional round-trip on a real filesystem.
    _seed_writer(writer, ...)  # writer-specific setup, mirrors existing tests
    ckpt = str(tmp_path / "smoke.X.ckpt")
    writer.dump_X_checkpoint(ckpt)
    assert os.path.exists(ckpt)
    assert not os.path.exists(ckpt + ".tmp")
    loaded = writer.load_X_checkpoint(ckpt)
    assert loaded == writer.get_n_prompts_accumulated()
```

Estimated LoC per writer: ~25 lines. Total: 11 × 25 ≈ 275 lines of new tests
across the 11 `test_calibration_*_smoke.py` files inside the patch.

**If user picks option (a)** (existing project style): replace the
`inspect.getsource` assertion with a `mock.patch.object(os, "fsync", ...)`
spy that records the fsync targets. Net delta vs option (b): -2 LoC per
test, but reintroduces the prohibited pattern.

**If user picks option (c)** (strace): wrap each writer's smoke test in
`@pytest.mark.skipif(not has_strace, …)`, spawn a subprocess that imports the
writer, runs `dump_X_checkpoint`, and exits; parent reads `/proc/<pid>/syscalls`
or strace's CSV output to assert fsync was invoked. Net delta vs (b):
+40 LoC fixture in conftest, +Linux-only, but most rigorous.

---

## H-2 design

### Per-(rank, expert) torch.Generator — mirrors CRITICAL-1's `_LayerInputAccumulator`

Reference pattern (already shipped, proven in production):
- Init: `vllm_calibration_stage2_profile.patch:614` →
  `acc._generator.get_state().clone()` (serialise)
- Restore: same patch L761 →
  `acc._generator.set_state(generator_state.clone())` (deserialise)
- Class def: `max_quality/src/moe_compress/stage2/profiling.py:82-86`:
  ```python
  def __init__(self, max_samples: int = 8192, *, seed: int = 0) -> None:
      self._generator = torch.Generator(device="cpu").manual_seed(int(seed))
  ```

### Apply to `vllm/calibration_output_reservoir.py`

The Phase-2 RNG site (patch L7653-7655):
```python
+            u = torch.rand(n_remaining)
+            ...
+            slots = torch.randint(0, cap, (n_remaining,))
```
uses GLOBAL `torch.rand` / `torch.randint`. Three changes:

1. **Init**: maintain a `_GENERATORS: dict[tuple[int, int], torch.Generator]` at
   module scope (parallel to `_RESERVOIR`, `_SEEN`, `_VALID_COUNT`). Lazily
   create a per-(rank, expert) generator on first use, seeded with a
   deterministic value derived from the key:
   ```python
   def _get_generator(key: tuple[int, int]) -> torch.Generator:
       g = _GENERATORS.get(key)
       if g is None:
           # Deterministic per-(rank, expert) seed. The mix below avoids
           # adjacent-key seed collisions while staying inside int64.
           rank, e = key
           seed = (rank * 1_000_003 + e) & 0x7FFFFFFFFFFFFFFF
           g = torch.Generator(device="cpu").manual_seed(seed)
           _GENERATORS[key] = g
       return g
   ```

2. **Use in Phase-2**: replace the global calls:
   ```python
   g = _get_generator(key)
   u = torch.rand(n_remaining, generator=g)
   ...
   slots = torch.randint(0, cap, (n_remaining,), generator=g)
   ```

3. **Serialise** in `dump_output_reservoir_checkpoint` (patch L7745-7755): add a
   new field to the payload dict:
   ```python
   "generator_states": {
       k: _GENERATORS[k].get_state().clone()
       for k in _GENERATORS
   },
   ```

4. **Deserialise** in `load_output_reservoir_checkpoint` (patch L7761-7801):
   restore generators, with **back-compat for old checkpoints**:
   ```python
   _GENERATORS.clear()
   gs_payload = payload.get("generator_states", None)
   if gs_payload is None:
       # Pre-H-2 checkpoint: no generator state. WARN; downstream
       # Phase-2 draws will diverge from a no-resume run. Generators
       # will be lazy-init'd on next use (deterministic seed) so the
       # FUTURE draws are still reproducible across re-resumes.
       log.warning(
           "calibration_output_reservoir: checkpoint at %s predates the "
           "H-2 RNG-state fix (no 'generator_states' field). Resume "
           "will continue but the resumed Phase-2 reservoir contents "
           "will NOT be byte-identical to a single-run counterfactual.",
           ckpt_path,
       )
   else:
       for k_raw, state in gs_payload.items():
           k = (int(k_raw[0]), int(k_raw[1]))
           g = torch.Generator(device="cpu")
           g.set_state(state.clone())
           _GENERATORS[k] = g
   ```

   Per the `stage2_profile` precedent, we do NOT bump `_CHECKPOINT_SCHEMA_VERSION`
   for this addition (the field is optional + back-compat-loaded). The
   stage2_profile pattern is to keep schema_version stable when adding optional
   payload fields and to use arity / `.get(..., None)` checks at load time.

   **Alternative considered**: bump `_CHECKPOINT_SCHEMA_VERSION` and hard-fail
   old checkpoints. Rejected because in-flight calibration runs from the prior
   wheel rev would be unrecoverable on the wheel upgrade. Soft fallback +
   WARN matches the audit's recovery-first stance.

### Site list (H-2)

| Patch file | Module | Lines (current) | Change |
|---|---|---|---|
| hooks | `vllm/calibration_output_reservoir.py` | new module-scope `_GENERATORS = {}` + `_get_generator()` helper | +12 LoC |
| hooks | `vllm/calibration_output_reservoir.py` (~L7653-7655) | Phase-2 sampling — add `generator=g` kwargs, fetch `g = _get_generator(key)` | ~4 LoC delta |
| hooks | `vllm/calibration_output_reservoir.py` (dump, ~L7745-7755) | add `"generator_states": {...}` field to payload | +4 LoC |
| hooks | `vllm/calibration_output_reservoir.py` (load, ~L7761-7801) | clear `_GENERATORS`, restore from payload with back-compat WARN | +18 LoC |

### Tests (H-2)

Existing test `test_two_segment_additivity_fill_only` (patch L3148-3213) intentionally
stays in the fill-only regime. Add a NEW test alongside it:

```python
def test_two_segment_additivity_phase_2_byte_identical(tmp_path):
    """H-2: Phase-2 reservoir-sampling resume must be byte-identical.

    Mirrors test_two_segment_additivity_fill_only's structure but pushes
    total tokens past `cap` so Phase 2 is dominant. Pre-H-2 this test
    FAILS (resumed reservoir contents differ); post-H-2 it PASSES.
    """
    n_experts = 1
    top_k = 1
    cap = 4               # small cap → easy to overshoot
    hidden = 2
    n_tok = 20            # 20 >> 4 → Phase 2 owns 16 of the 20 draws
    torch.manual_seed(7)
    uw_all = torch.randn(n_tok, top_k, hidden, dtype=torch.float32)
    ids_all = torch.zeros(n_tok, top_k, dtype=torch.int64)

    env = {
        "VLLM_CALIB_CAPTURE_OUTPUT_RESERVOIR": "1",
        "VLLM_CALIB_OUTPUT_RESERVOIR_CAP": str(cap),
        "VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED": "1",
    }
    # Reference: single 20-token run.
    ref = _reload_or(env)
    _seed_layer(ref, layer_idx=0, rank=0, n_experts=n_experts)
    ref._on_expert_out_unweighted(0, uw_all, ids_all)
    expected = ref._RESERVOIR[(0, 0)].clone()

    # Split: 8 tokens + checkpoint + reload + 12 tokens.
    seg = _reload_or(env)
    _seed_layer(seg, layer_idx=0, rank=0, n_experts=n_experts)
    seg._on_expert_out_unweighted(0, uw_all[:8], ids_all[:8])
    seg.set_n_prompts_accumulated(1)
    ckpt = str(tmp_path / "phase2.or.ckpt")
    seg.dump_output_reservoir_checkpoint(ckpt)

    seg2 = _reload_or(env)
    seg2.load_output_reservoir_checkpoint(ckpt)
    seg2._on_expert_out_unweighted(0, uw_all[8:], ids_all[8:])

    assert torch.equal(seg2._RESERVOIR[(0, 0)], expected), (
        "H-2 fix did NOT preserve byte-identical Phase-2 resume."
    )
```

Plus a back-compat regression test:

```python
def test_load_pre_h2_checkpoint_warns_and_proceeds(tmp_path, caplog):
    """H-2: a pre-H-2 checkpoint (missing 'generator_states') must load
    with a WARNING, not a hard-fail. Downstream Phase-2 draws will
    diverge from a single-run counterfactual but the load itself
    survives so in-flight runs from the prior wheel rev are recoverable.
    """
    # ... fabricate a payload dict lacking 'generator_states', torch.save it,
    # call load_output_reservoir_checkpoint, assert WARN was emitted and
    # _RESERVOIR was hydrated correctly.
```

The `caplog` fixture is the env-independent capture path (`_attach_capture_handler`
helper if needed per the `no-monkey-patches` lesson; bare `caplog` works in
properly-configured pytest envs).

Estimated LoC: ~60 lines new tests in `tests/test_calibration_output_reservoir_smoke.py`
inside the hooks patch.

---

## Affected patch files + estimated LoC delta

| Patch file | H-1 LoC | H-2 LoC | Test LoC | Total Δ (lines) |
|---|---|---|---|---|
| `vllm_calibration_hooks.patch` | +110 (10 helpers × ~11 lines, minus 2 lines per replaced site) | +38 (H-2 init + dump + load) | +275 (H-1 per-writer tests) + 60 (H-2 byte-identical + back-compat) | **~+483 lines** (current 10920 → ~11403) |
| `vllm_calibration_stage2_profile.patch` | +11 (1 helper inlined + 2-line site collapse) | n/a | 0 (stage2_profile's existing `test_checkpoint_resume_byte_identical_after_phase_c` already exercises the resume path; new H-1 fsync test is identical-shape to the per-writer ones above, ~25 LoC) | **~+36 lines** (current 802 → ~838) |

Net wheel-payload code delta: ~+520 lines across both patches. The MANIFEST.md
line-counts + MD5s update once at the end of implementation.

---

## Risk

1. **Pattern O parent-dir fsync** can raise `OSError` (EINVAL/ENOTSUP) on
   tmpfs / FUSE filesystems (CI sandboxes, HF Jobs ephemeral mounts). The
   inline helper swallows the exception with a `try/except OSError: pass`
   (matches `atomic_io._fsync_dir`). The functional round-trip test will
   pass on tmpfs because the swallow is intentional; this is the same
   trade-off the production `atomic_io` helper already makes.

2. **On-disk checkpoint shape changes for `output_reservoir`** (new
   `generator_states` field). Back-compat is via `.get("generator_states", None)`
   + WARN. This protects in-flight runs from the prior wheel rev but means
   the *first* resume after the wheel upgrade will produce a non-byte-identical
   Phase-2 reservoir (the resumed segment will not have the original Phase-2
   draws). Operators running an ablation-sweep that DEPENDS on byte-equality
   should delete the pre-H-2 checkpoint and restart calibration on the new wheel.

3. **MANIFEST.md MD5 + line count must re-bump after both fixes ship.**
   Skipping this step makes the wheel-build script's identity check fail
   and the patched wheel will be re-built incorrectly. The plan's commit
   sequence ends with the MANIFEST.md bump explicitly.

4. **Test strategy decision (option a/b/c above) gates the test-LoC
   estimates.** The 275 + 60 LoC test estimates above assume option (b) (AST
   inspection + functional round-trip). Option (a) shaves ~20 LoC total.
   Option (c) adds ~40 LoC fixture and a strace dependency.

5. **The `_atomic_torch_save` helper is duplicated 10 times** (one per
   writer module). This is deliberate (avoids cross-module imports inside
   the wheel — see "Why inline-per-writer" above). Future drift between
   the copies is mitigated by: (a) all 10 copies pinned identical at
   commit time; (b) a smoke test per writer asserts the helper produces
   a readable file (catches a typo in any copy); (c) the source-of-truth
   `atomic_io._fsync_dir` / `durable_rename` semantics are documented at
   `atomic_io.py:122-159` and referenced in each inlined helper's
   docstring. If a future Pattern-O change is needed, all 11 inlined
   sites must be edited together — same maintenance cost as the existing
   bare `tmp + torch.save + os.replace` 11-site sweep that this plan
   replaces.

---

## Commit structure (for the implementer)

Each step is one focused commit. The implementer should run the full
`max_quality/tests/` pytest suite after each commit to catch regressions
early.

1. **`patch(vllm): inline _atomic_torch_save helper in 10 writer modules + stage2_profile`**
   — purely additive (defines the helper at the top of each writer's "checkpoint
   primitives" section; no call-site changes yet). Lets the test harness
   import the helper standalone for the next commit's tests. ~110 LoC
   (10 × ~11 LoC).

2. **`patch(vllm): H-1 fsync periodic checkpoints — 11 sites use _atomic_torch_save`**
   — replaces every `tmp + torch.save + os.replace` triple with a single
   `_atomic_torch_save(payload, path)` call. The imatrix `_write_dat` site uses
   the parallel `_durable_close_replace_dat` helper variant. ~33 LoC delta (11
   × 3-line block → 11 × 1-line call) + 11 LoC for the imatrix `.dat` helper.

3. **`patch(vllm): H-2 output_reservoir per-(rank,expert) torch.Generator`**
   — adds `_GENERATORS` dict + `_get_generator(key)` helper, replaces the
   2 global `torch.rand` / `torch.randint` calls in Phase-2 with `generator=g`
   variants, adds `generator_states` to the dump payload, adds back-compat
   restore in load. ~38 LoC.

4. **`patch(vllm): tests — H-1 fsync round-trip per writer (11 tests)`**
   — adds `test_periodic_ckpt_uses_fsync_pattern_o` to each
   `tests/test_calibration_*_smoke.py` inside the patch. ~275 LoC. **Assumes
   option (b) from the REQUEST FOR DECISION above; switch to (a) or (c) per
   user's choice.**

5. **`patch(vllm): tests — H-2 Phase-2 byte-identical + back-compat WARN`**
   — adds `test_two_segment_additivity_phase_2_byte_identical` and
   `test_load_pre_h2_checkpoint_warns_and_proceeds` to
   `tests/test_calibration_output_reservoir_smoke.py` inside the patch. ~60 LoC.

6. **`patch(manifest): bump line count + MD5 for both patches`** — final
   commit, runs `wc -l max_quality/patches/vllm_calibration_*.patch` and
   `md5sum` and updates the MANIFEST.md table.

Per commit: tests are validated by re-applying the patch against a fresh
v0.21.0 vLLM checkout and running `pytest tests/test_calibration_*_smoke.py`
inside the patched tree. The wheel-build script (out of scope for this
implementation but unblocked once commits 1-6 land) will rebuild and
re-upload to `pirola/vllm-patched-calib`.

---

## Out of scope (for separate tickets)

- **C-1** (operator-adds-writer-on-resume undercount) — driver-side, no patch
  change. Lives in `max_quality/scripts/build_self_traces_calib_vllm.py`.
- **M-1** (router_logits_stats + block_outputs lack two-segment additivity
  tests) — test-side, lives in `tests/test_calibration_*_smoke.py` BUT not
  blocking the wheel rev. Could be folded into the H-1 test commit if the
  user wants ONE bigger test push, but the audit calls them separate findings.
- **M-2** (stale-ckpt-from-removed-writer pre-flight sweep) — driver-side.
- **M-3** (F-H-6 hard-fail message metadata-vs-math gap) — driver-side wording.
- **L-1 / L-2 / L-3** (documentation asymmetries, zero-prompt log line, etc.)
  — non-functional, defer.
- **N-1 / N-2 / N-3** (nitpicks: closure inlining, log message paths, help text)
  — non-functional.
- **Pre-existing monkey-patch tests** in `test_utils_atomic_io.py:212-214` and
  `test_stage3_wanda_scalar_row_cache.py:189` — these violate the project's
  `no-monkey-patches` standing rule but are out of scope for THIS plan
  (separate refactor ticket; raised in the REQUEST FOR DECISION above so the
  user is aware of the test-style asymmetry the H-1 tests will create).
