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

### Decision-fold (v2): test strategy is (c) strace fixture, both violator tests migrated

The v1 planner raised a 3-way choice for verifying fsync call ORDER without
re-introducing the prohibited `mock.patch.object` / `monkeypatch.setattr`
pattern (memory: `feedback_no_monkey_patches.md`, 2026-05-29). The user picked:

- **Test strategy: (c) strace fixture** — observe real `fsync` / `rename` /
  `renameat` / `renameat2` syscalls and assert order. Most rigorous;
  Linux-only (already a project constraint per
  `feedback_compare_against_actual_upstream.md`'s clone-and-read workflow).
- **Migrate the 2 pre-existing violators**: `test_utils_atomic_io.py:212-214`
  and `test_stage3_wanda_scalar_row_cache.py:189` get converted from
  `mock.patch.object` to the same strace fixture. Closes the prohibited-pattern
  surface in this directory entirely (rather than letting H-1's new tests be
  the only fsync-order tests not using `mock.patch.object`).

This is reflected in:
- H-1 test section below (strace fixture spec + per-test usage).
- New commits #8 + #9 in the commit sequence (violator migrations).
- LoC tables and risk section updated for the strace dependency.
- "Out of scope" section: the two violator migrations are MOVED out of
  scope and INTO this plan.

See "Plan-v2 ledger" at the bottom of this doc for the audit trail.

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

### Tests (H-1) — strace fixture (decision (c))

Per-writer smoke test files already exist inside the hooks patch
(`tests/test_calibration_*_smoke.py`). For each of the 10 writers + imatrix
`.dat`, add ONE new test that uses a shared `strace_syscalls` pytest fixture
to assert the exact `fsync → replace → fsync(parent_dir)` ordering by reading
real syscalls.

#### Fixture location + spec

The fixture lives in a NEW file vendored inside the patches' test tree:
`tests/conftest_strace.py` (imported into the patch's `tests/conftest.py`
via `from .conftest_strace import strace_syscalls`). It lives at the patch's
test-tree root so all 11 `test_calibration_*_smoke.py` files can use it
without per-file boilerplate.

```python
# tests/conftest_strace.py
"""Strace-based syscall observer for Pattern O fsync ordering tests.

NOTE on fsync predicates: strace records `fsync(N)` with a bare numeric
file descriptor — there is NO path argument to match against. The
canonical Pattern O trace is

    openat(..., "/.../target.tmp",  O_RDONLY|O_CLOEXEC) = 7
    fsync(7) = 0
    rename("/.../target.tmp", "/.../target") = 0
    openat(..., "/.../parent_dir",  O_RDONLY|O_DIRECTORY|...) = 7
    fsync(7) = 0

so the file vs. parent-dir fsync is disambiguated by the most-recent
preceding `openat` (path-bearing) — NOT by anything in the fsync's own
args. See "Predicate authoring guide (fd disambiguation)" below for
the full FD-correlation pattern.

Usage:
    def test_periodic_ckpt_pattern_o(strace_syscalls):
        with strace_syscalls() as recorder:
            writer.dump_X_checkpoint(ckpt_path)
        # recorder.syscalls is a list of (syscall_name, args_str) tuples
        # in observed order, filtered to fsync/rename*/openat events.
        recorder.assert_order(
            ("openat", lambda a: ".tmp" in a),               # opens payload tmp
            ("fsync",  lambda a: True),                      # payload fsync (fd from prev openat)
            ("rename", lambda a: ".tmp" in a),               # tmp → final
            ("openat", lambda a: "O_DIRECTORY" in a),        # opens parent dir
            ("fsync",  lambda a: True),                      # parent-dir fsync (fd from prev openat)
        )
"""
import contextlib
import os
import re
import subprocess
import sys

import pytest


# Each strace line we care about looks like:
#   <pid> fsync(7)                                 = 0
#   <pid> rename("/tmp/xxx.tmp", "/tmp/xxx")       = 0
#   <pid> renameat2(AT_FDCWD, "/.../a.tmp", ...)   = 0
#   <pid> openat(AT_FDCWD, "/some/dir", O_RDONLY...) = 7
_SYSCALL_RE = re.compile(
    r"^(?:\[pid\s+\d+\]\s+)?"
    r"(?P<name>fsync|rename|renameat|renameat2|openat)"
    r"\((?P<args>[^)]*)\)\s*=\s*(?P<ret>-?\d+|0x[0-9a-fA-F]+)"
)


class _StraceRecorder:
    def __init__(self, syscalls):
        # list of (name, args_str)
        self.syscalls = syscalls

    def assert_order(self, *predicates):
        """Assert the recorded syscalls match the given ordered predicates.

        Each predicate is a (syscall_name, args_predicate_callable) tuple.
        The recorded syscalls are scanned in order; each predicate must
        match SOME subsequent syscall (gaps OK — e.g. openat between
        fsync and rename is fine).
        """
        i = 0
        for syscall_name, args_pred in predicates:
            while i < len(self.syscalls):
                name, args = self.syscalls[i]
                i += 1
                if name == syscall_name and args_pred(args):
                    break
            else:
                raise AssertionError(
                    f"strace order check failed: did not find {syscall_name} "
                    f"matching predicate after position {i}. "
                    f"Recorded syscalls: {self.syscalls}"
                )


@pytest.fixture
def strace_syscalls(tmp_path):
    """Yield a context-manager factory that records fsync/rename/openat.

    The factory runs the body inside a `strace -e trace=...` subprocess
    that re-imports the writer and invokes the dump. Stdout/stderr of
    the subprocess is captured + parsed.

    Skipped if `strace` is not on PATH (e.g. macOS, BSD CI).
    """
    if subprocess.run(
        ["which", "strace"], capture_output=True
    ).returncode != 0:
        pytest.skip("strace not available; H-1 fsync-order test requires Linux + strace")

    @contextlib.contextmanager
    def _runner(python_code: str):
        # The test body emits python source that does the dump; we
        # exec it under strace and parse the trace log.
        trace_log = tmp_path / "strace.log"
        cmd = [
            "strace",
            "-f",                                          # follow forks
            "-e", "trace=fsync,rename,renameat,renameat2,openat",
            "-o", str(trace_log),
            sys.executable, "-c", python_code,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise AssertionError(
                f"strace subprocess failed (rc={res.returncode}):\n"
                f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
            )
        syscalls = []
        for line in trace_log.read_text().splitlines():
            m = _SYSCALL_RE.search(line)
            if m:
                syscalls.append((m.group("name"), m.group("args")))
        yield _StraceRecorder(syscalls)

    yield _runner
```

#### Per-writer test shape

```python
def test_periodic_ckpt_uses_fsync_pattern_o(tmp_path, strace_syscalls):
    """H-1: dump_<writer>_checkpoint must issue fsync → rename → fsync(parent).

    Uses the strace_syscalls fixture (option (c) from the v1 planner's
    REQUEST FOR DECISION) to observe real syscalls in a subprocess.
    """
    ckpt = tmp_path / "smoke.X.ckpt"
    body = f'''
import torch
from vllm import calibration_X as writer
# writer-specific seed (mirrors existing smoke tests in this file)
writer._RESERVOIR[(0, 0)] = torch.zeros(2, 3)
writer.set_n_prompts_accumulated(1)
writer.dump_X_checkpoint(r"{ckpt}")
'''
    with strace_syscalls(body) as recorder:
        pass  # body ran inside the strace subprocess
    # Assert: fsync(payload_tmp) → rename(tmp → final) → fsync(parent_dir)
    recorder.assert_order(
        ("fsync",  lambda a: True),                          # payload fd fsync
        ("rename", lambda a: ".tmp" in a and str(ckpt) in a),# tmp → final
        ("fsync",  lambda a: True),                          # parent dir fd fsync
    )
    assert ckpt.exists()
    assert not (ckpt.with_suffix(ckpt.suffix + ".tmp")).exists()
    # Functional round-trip check (still useful)
    loaded = torch.load(ckpt, weights_only=False)
    assert "n_prompts_accumulated" in loaded
```

#### Predicate authoring guide (fd disambiguation)

The per-writer test shape above uses lambdas like
`lambda a: ".tmp" in a` against the `args` string of a recorded syscall.
This works for `rename` and `openat` because they take path arguments
directly. **`fsync(N)` is different** — strace records it as a numeric
file descriptor with no path. The same trace prefix
`openat(path, O_RDONLY) + fsync(fd) + close(fd)` is emitted by BOTH the
file-fsync helper AND the parent-dir-fsync helper in
`utils/atomic_io.py`. At the syscall level the trace for a normal
durable-rename therefore looks like:

```
openat(AT_FDCWD, "/.../target.tmp",  O_RDONLY|O_CLOEXEC) = 7
fsync(7)                                                  = 0
close(7)                                                  = 0
rename("/.../target.tmp", "/.../target")                  = 0
openat(AT_FDCWD, "/.../parent_dir", O_RDONLY|O_DIRECTORY) = 7
fsync(7)                                                  = 0
close(7)                                                  = 0
```

Predicates that need to distinguish the *file* fsync from the *parent
dir* fsync MUST correlate each `fsync(fd)` with the most-recent
`openat(path, ...)` that returned that `fd`. Two options for the
implementer:

1. **(preferred)** Extend `_StraceRecorder` with an
   `fd_path(fd) -> str` helper. The recorder walks the recorded
   syscalls forward, maintaining a `dict[int, str]` of live fds keyed
   by the path captured from the matching `openat`'s args, removing
   entries on `close(fd)`. The fixture would then accept predicates
   shaped as `lambda ev: ev.path == str(target_dir)` where `ev`
   carries the resolved path. This requires changing `_SYSCALL_RE` to
   also capture `close`, and extending `assert_order` to pass the
   live-fd dict into each predicate.

2. **(simpler, may be enough)** Keep the recorder as-is and write
   per-test predicates that walk the trace themselves. For
   `test_durable_rename_call_order`:
   ```python
   recorder.assert_order(
       ("openat",  lambda a: '.tmp' in a),                     # path-only, robust to future flag-set changes
       ("fsync",   lambda a: True),                            # file fd
       ("rename",  lambda a: ".tmp" in a),
       ("openat",  lambda a: 'O_DIRECTORY' in a),
       ("fsync",   lambda a: True),                            # parent dir fd
   )
   ```
   This works because `assert_order` matches predicates in sequence
   with gaps allowed, and the `openat(..., O_DIRECTORY)` flag
   reliably disambiguates the parent-dir open from the file open.
   The path-only `'.tmp' in a` predicate (instead of an
   `a.endswith(..., O_RDONLY|O_CLOEXEC')` flag-set match) survives
   future Python / libc default-flag changes — the previous strict
   form was tightly coupled to CPython 3.4+'s `O_CLOEXEC`-by-default
   on `os.open`. (L-v3-1 fold.)

The implementer should start with option (2) — it stays within the
existing `_StraceRecorder` API and, **after commit #2 in the
"Commit structure" section below lands** (the new
`feat(atomic_io): _fsync_dir uses O_DIRECTORY` change), the
`O_DIRECTORY` flag is a stable disambiguator written by
`os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))` in
`utils/atomic_io._fsync_dir`. Promote to option (1) only if a test
arises where `O_DIRECTORY` is not present after that commit — which
would be either a regression in `_fsync_dir` or a non-Linux platform
where `getattr(os, "O_DIRECTORY", 0)` evaluates to `0` (the production
fallback). Both cases warrant raising back to the user rather than
silently switching test idioms.

(Note: previously, prior to the v4 fold of H-v3-1, `_fsync_dir` opened
the directory with bare `os.O_RDONLY` — strace decoded the parent-dir
openat as `O_RDONLY|O_CLOEXEC`, making the `'O_DIRECTORY' in a`
predicate FALSE in practice. The new `_fsync_dir` commit makes the
predicate factually correct on Linux.)

**RAISE-not-fallback**: per `feedback_raise_dont_substitute.md`, if
predicate authoring proves harder than either pattern above — e.g.,
the strace trace shows the `openat` happening in a forked child whose
pid `strace -f` does not interleave with the parent's syscalls in the
log, or `O_DIRECTORY` is absent on the test box's `_fsync_dir`
codepath, or the `_StraceRecorder` API needs a refactor wider than a
single-method addition — the implementer SHOULD RAISE back to the user
rather than fall back to a weaker test idiom (e.g., monkey-patching
`_fsync_file`, asserting only call-count rather than order, etc.).
Option (2)'s assertion shape is the LOAD-BEARING contract for H-1; any
deviation goes through the user.

Estimated LoC:
- Fixture (one-time): ~110 LoC in `tests/conftest_strace.py` + ~2 LoC
  import in `tests/conftest.py`.
- Per-test usage: ~20 LoC (subprocess-body string + assertion block).
- Total: ~110 fixture + 11 × ~20 = **~330 lines** of new test code across
  the 11 `test_calibration_*_smoke.py` files + conftest.

(The v1 plan's pre-decision estimate of ~275 LoC under option (b) is
replaced by this updated ~330 LoC figure: the per-test cost dropped from
~25 → ~20 LoC because the assertion is `recorder.assert_order(...)` instead
of an AST grep + round-trip block, but the one-time fixture cost is ~110
LoC — net +55 LoC vs option (b).)

#### Strace dependency note

`strace` is part of `procps`-adjacent tooling on every Linux distro the
project's CI/dev boxes use (verified `/usr/bin/strace` present on the
v6.8 dev host). For non-Linux developers (macOS) the fixture issues
`pytest.skip(...)` so the rest of the smoke suite still runs. The wheel
build pipelines all run on Linux containers; the strace requirement does
NOT affect production runtime, only test execution. See risk note #6 below.

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
| `vllm_calibration_hooks.patch` | +110 (10 helpers × ~11 lines, minus 2 lines per replaced site) | +38 (H-2 init + dump + load) | +330 H-1 (110 fixture + 11×20 per-writer) + 60 H-2 (byte-identical + back-compat) | **~+538 lines** (current 10920 → ~11458) |
| `vllm_calibration_stage2_profile.patch` | +11 (1 helper inlined + 2-line site collapse) | n/a | +20 (1 strace-fixture-based fsync test in the stage2 smoke file; the fixture itself is imported from the hooks-patch conftest, not duplicated) | **~+31 lines** (current 802 → ~833) |

Migrations of the 2 pre-existing violators (separate non-patch files, do
NOT touch the wheel-vendored test tree):

| File | Lines (current) | Δ | Notes |
|---|---|---|---|
| `max_quality/tests/test_utils_atomic_io.py` | L212-214 use `mock.patch.object` | ~+10 / -8 | Replace 3-line `with mock.patch.object(...) as ...` block with `with strace_syscalls(body) as recorder: ...; recorder.assert_order(...)`. The fixture is imported from `max_quality/tests/conftest.py` (NOT the patch-vendored conftest — needs a sibling copy). |
| `max_quality/tests/test_stage3_wanda_scalar_row_cache.py` | L189 uses `patch.object` | ~+10 / -5 | Same migration pattern. |
| `max_quality/tests/conftest.py` | new | +5 | `from .conftest_strace import strace_syscalls` (sibling copy of the same fixture used inside the patches' test tree) |
| `max_quality/tests/conftest_strace.py` | new | +110 | Sibling copy of the fixture vendored at `<patch>/tests/conftest_strace.py`. Kept in sync by hand at commit time; future change to one must update both. |

Net wheel-payload code delta: ~+569 lines across both patches.
Net repo code delta (non-patch): ~+125 lines for the violator migrations
+ the second fixture copy.
The MANIFEST.md line-counts + MD5s update once at the end of implementation.

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

4. **Test strategy decision is folded (option (c) chosen).** The
   ~330 H-1 + ~60 H-2 + ~125 violator-migration LoC reflect the strace
   fixture path. Original v1 estimates (option (b)) are superseded.

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

6. **Strace must be on PATH at test time.** Verified present
   (`/usr/bin/strace`, v6.8) on the dev host and standard on every Linux
   container the project uses (apt: `strace`, alpine: `strace`). The
   fixture issues `pytest.skip` if absent — so non-Linux contributors
   (macOS) still run the rest of the smoke suite cleanly, but they
   cannot validate the fsync-order assertions locally. CI MUST run on
   Linux + ensure `strace` is in the image. Document the requirement
   in `tests/conftest_strace.py`'s module docstring and in the patches'
   README hunk so future maintainers see it before debugging a "test
   skipped" puzzle.

7. **The `conftest_strace.py` fixture exists in two locations**: once
   vendored inside each patch's test tree (`<patch>/tests/conftest_strace.py`),
   once in the repo-level test tree (`max_quality/tests/conftest_strace.py`).
   The two copies MUST stay byte-identical. Mitigation: a CI sanity check
   `diff max_quality/tests/conftest_strace.py max_quality/patches/.../conftest_strace.py`
   in pre-commit / pre-push. Alternative considered: symlink the second
   copy — rejected because the patch-vendored copy must survive the
   `git apply` of the patch into a fresh vLLM checkout, where symlinks
   to outside-the-tree paths break.

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

2. **`feat(atomic_io): _fsync_dir uses O_DIRECTORY for portable, fail-fast directory fsync`**
   — single-line production change at
   `max_quality/src/moe_compress/utils/atomic_io.py:125`:
   ```python
   # before:
   fd = os.open(str(directory), os.O_RDONLY)
   # after:
   fd = os.open(str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
   ```
   Mirrors the existing portability pattern at
   `max_quality/src/moe_compress/router_kd/plugins/early_stop.py:168`
   (same `getattr(os, "O_DIRECTORY", 0)` idiom). Strict improvement:
   - On Linux, `O_DIRECTORY` raises `ENOTDIR` if the path is not a
     directory — fail-fast for a class of caller bugs that previously
     fsync-ed silently against the wrong fd.
   - On platforms where `O_DIRECTORY` is absent from `os` (rare; macOS
     does have it, BSD does, the `getattr(..., 0)` fallback covers any
     hypothetical platform where it isn't exposed), the OR-with-0
     yields the original `O_RDONLY` behavior — zero behavior change.
   - Decodes in strace as `O_RDONLY|O_DIRECTORY|O_CLOEXEC` (or similar
     superset), making the `'O_DIRECTORY' in a` predicate in the
     fixture authoring guide (above) factually correct. This is the
     v3 reviewer's HIGH (H-v3-1) folded via user-chosen path (b).

   **Test impact**: `max_quality/tests/test_utils_atomic_io.py` has
   existing `_fsync_dir` exercises (the `real_fsync_dir`/`rec_fsync_dir`
   spy at L197/L208/L210/L214, and the order assertion at L217). None
   of these assert on the openat flag-set; they assert on call-order
   and on the helper returning successfully against a real directory.
   `getattr(os, "O_DIRECTORY", 0)` evaluates to a non-zero int on
   Linux+macOS (where the existing tests run) and `O_DIRECTORY` already
   implies `O_RDONLY`-compatible read semantics for the parent-dir
   fsync, so the existing tests continue to pass unchanged. A new
   one-line assertion that `_fsync_dir(<non-dir path>)` raises
   `NotADirectoryError` / `OSError(ENOTDIR)` may be added in the same
   commit to lock in the fail-fast contract (optional; out of the LoC
   budget but cheap if the implementer wants to harden the change).

   LoC: 1 production line changed + ~5 LoC optional new test = ~6 LoC.

3. **`patch(vllm): H-1 fsync periodic checkpoints — 11 sites use _atomic_torch_save`**
   — replaces every `tmp + torch.save + os.replace` triple with a single
   `_atomic_torch_save(payload, path)` call. The imatrix `_write_dat` site uses
   the parallel `_durable_close_replace_dat` helper variant. ~33 LoC delta (11
   × 3-line block → 11 × 1-line call) + 11 LoC for the imatrix `.dat` helper.

4. **`patch(vllm): H-2 output_reservoir per-(rank,expert) torch.Generator`**
   — adds `_GENERATORS` dict + `_get_generator(key)` helper, replaces the
   2 global `torch.rand` / `torch.randint` calls in Phase-2 with `generator=g`
   variants, adds `generator_states` to the dump payload, adds back-compat
   restore in load. ~38 LoC.

5. **`test(infra): strace_syscalls fixture — foundation for fsync-order tests`**
   — adds `tests/conftest_strace.py` inside the hooks patch's vendored
   test tree AND a sibling copy at `max_quality/tests/conftest_strace.py`.
   Wires both into their respective `conftest.py`. ~115 LoC each location
   (~230 LoC total in this commit). This commit lands BEFORE the fsync-order
   tests in commits #6 / #8 / #9 so they all reference an existing fixture.

6. **`patch(vllm): tests — H-1 fsync ORDER per writer (11 strace tests)`**
   — adds `test_periodic_ckpt_uses_fsync_pattern_o` to each
   `tests/test_calibration_*_smoke.py` (10 writers) + the imatrix `.dat`
   smoke + the stage2_profile smoke. Each test uses the `strace_syscalls`
   fixture from commit #5. ~330 LoC across 12 test files (110 fixture
   already shipped in #5, so this commit adds 11 × ~20 LoC).

7. **`patch(vllm): tests — H-2 Phase-2 byte-identical + back-compat WARN`**
   — adds `test_two_segment_additivity_phase_2_byte_identical` and
   `test_load_pre_h2_checkpoint_warns_and_proceeds` to
   `tests/test_calibration_output_reservoir_smoke.py` inside the patch. ~60 LoC.

8. **`test(refactor): migrate test_utils_atomic_io.py off mock.patch.object`**
   — converts `max_quality/tests/test_utils_atomic_io.py:212-214`'s
   `with mock.patch.object(aio, "_fsync_file", ...) ...` block to the
   `strace_syscalls` fixture from commit #5. Closes one of the two
   pre-existing `no-monkey-patches` violators. ~+10 / -8 LoC.

9. **`test(refactor): migrate test_stage3_wanda_scalar_row_cache.py off patch.object`**
   — converts `max_quality/tests/test_stage3_wanda_scalar_row_cache.py:189`'s
   `with patch.object(_aio.os, "replace", side_effect=_spy_replace):` block
   to the `strace_syscalls` fixture. Closes the second violator. ~+10 / -5 LoC.
   (Note: the `monkeypatch.setattr` / `patch.object` calls at L299, L437,
   L438, L458 in the same file stub `instrument_experts` /
   `build_calibration_tensor` / `save_compressed_checkpoint` / `walk_phases`
   — these are NOT fsync-order spies and are OUT OF SCOPE for this plan;
   they're a separate test-design question.) Uses the `strace_syscalls`
   fixture from commit #5.

   **Implementer note (payload reconstruction)**: the existing
   `test_writer_emits_manifest_after_payload` (around L170-218 of
   `test_stage3_wanda_scalar_row_cache.py`) builds a tensor `payload`
   via `_make_payload(...)`. To migrate, the subprocess body MUST
   reconstruct the payload via the deterministic `_make_payload(seed=...)`
   call (mirrors the existing pattern at L170-218 — `_make_payload` is
   deterministic via `torch.manual_seed`). Do NOT serialise the payload
   to a `tmp_path` file in the parent test and re-load it inside the
   subprocess body — that defeats the point of the strace approach
   (the extra `openat`/`read` syscalls from the parent's dump and the
   subprocess's reload pollute the trace, plus it couples the test to a
   serialisation side-channel). The ~20 LoC delta in the estimate above
   assumes the in-body `_make_payload(seed=...)` reconstruction approach.

10. **`patch(manifest): bump line count + MD5 for both patches`** — final
    commit, runs `wc -l max_quality/patches/vllm_calibration_*.patch` and
    `md5sum` and updates the MANIFEST.md table.

Per commit: tests are validated by re-applying the patch against a fresh
v0.21.0 vLLM checkout and running `pytest tests/test_calibration_*_smoke.py`
inside the patched tree. After commits #8 / #9 the repo-level
`max_quality/tests/test_utils_atomic_io.py` and `test_stage3_wanda_scalar_row_cache.py`
must also pass. The wheel-build script (out of scope for this
implementation but unblocked once commits 1-10 land) will rebuild and
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
- **Other `monkeypatch.setattr` / `patch.object` sites in the migrated
  files** that do NOT spy fsync/replace order (e.g.,
  `test_stage3_wanda_scalar_row_cache.py` L299, L437, L438, L458 —
  production function stubbing for `instrument_experts` /
  `build_calibration_tensor` / `save_compressed_checkpoint` /
  `walk_phases`). These are a separate test-design refactor question
  and stay out of scope; commit #9 only migrates the fsync/replace spy
  at L189.

(Moved IN-SCOPE in v2: the two `mock.patch.object` violators at
`test_utils_atomic_io.py:212-214` and `test_stage3_wanda_scalar_row_cache.py:189`
— commits #8 and #9 above.)

---

## Plan-v2 ledger

This section documents the folds from v1 → v2 with reviewer attribution
+ design rationale, per `feedback_subagent_files_need_git_persistence.md`
(audit trail) and `feedback_raise_dont_substitute.md` (user-decision
provenance).

### Fold 1: Test strategy (c) strace fixture

- **Source**: v1 planner (commit `bfb40941d`) explicitly raised a 3-way
  choice (a/b/c) in §"Key constraint surfaced during planning (REQUEST
  FOR DECISION)" with default (b) AST + readback. Raised because the
  pre-existing `mock.patch.object` style (option (a)) violates the
  project's `no-monkey-patches` rule and option (b) cannot prove call
  ORDER (only call presence).
- **User decision** (folded here, 2026-05-29): chose **(c) strace fixture**
  — observe real syscalls. Rationale: most rigorous; catches
  helper-reorder bugs that AST or readback can't; the Linux constraint
  is already baked into the project.
- **Cost**: +110 LoC fixture (one-time), +5 LoC per test (per-test
  invocation), strace must be on PATH at test time (verified
  `/usr/bin/strace` v6.8 on dev host).
- **Updated in v2**: §"H-1 design / Tests", §"Risk" #6 (strace
  availability), §"Commit structure" #4 (fixture as foundation commit;
  renumbered to #5 in v4 after the O_DIRECTORY production fix was
  inserted at #2).

### Fold 2: Migrate the 2 pre-existing violators

- **Source**: v1 planner listed the 2 violators in §"Out of scope" with
  a note that the H-1 tests would create a test-style asymmetry (new
  tests use strict no-monkey-patch path; old tests still use
  `mock.patch.object`).
- **User decision** (folded here, 2026-05-29): chose **yes, migrate both**.
  Rationale: close the prohibited-pattern surface entirely in
  `max_quality/tests/` so reviewers can't accidentally cite the
  pre-existing violators as precedent for new monkey-patches.
- **Cost**: ~+20 LoC across 2 test files + a sibling fixture copy at
  `max_quality/tests/conftest_strace.py` (~115 LoC, byte-identical to
  the patch-vendored copy; risk #7 above covers the dual-copy
  maintenance).
- **Updated in v2**: §"Out of scope" (entry removed + back-reference
  added), §"Commit structure" #7 + #8 (renumbered to #8 + #9 in v4),
  §"Affected patch files + estimated LoC delta" (new "Migrations"
  table), §"Risk" #7 (dual fixture copy).

### What did NOT change from v1

- The H-1 site inventory (11 sites) is unchanged.
- The H-1 helper inlining strategy (per-writer `_atomic_torch_save`,
  imatrix's special `_durable_close_replace_dat`) is unchanged.
- The H-2 design (per-(rank, expert) `torch.Generator`, payload
  `generator_states` field, soft back-compat WARN at load) is
  unchanged.
- The MANIFEST.md re-bump is still the final commit.
- All "Risk" entries #1, #2, #3, #5 from v1 are unchanged.

### Raises (per `feedback_raise_dont_substitute.md`)

None pending. v1's single open decision-point is resolved by user input;
no new ambiguity surfaced during the fold. If the implementer hits an
unexpected blocker (e.g., `strace` parsing edge-case on a writer whose
dump_X path takes a `pathlib.Path` instead of a `str`), they should raise
back to the user rather than silently switch to a different test
strategy.

---

## Plan-v3 ledger

### Fold 1 (NIT, v2 → v3): out-of-scope L299 enumeration + implementer notes

- **Source**: plan-reviewer-v2 NIT.
- **Changes**: §"Out of scope" enumerated `L299, L437, L438, L458`
  (added `L299` — `instrument_experts`) plus matching parenthetical
  in commit #8's body (renumbered to #9 in v4). Two material
  implementer notes added: (a) the `_make_payload(seed=...)`
  reconstruction approach for commit #8's subprocess body
  (renumbered to #9 in v4; ~20 LoC budget already accounted),
  (b) the `feedback_raise_dont_substitute.md` block in the
  predicate authoring guide.

### Raises (per `feedback_raise_dont_substitute.md`)

None.

---

## Plan-v4 ledger

### Fold 1 (HIGH, H-v3-1): user picked path (b) — add O_DIRECTORY to `_fsync_dir`

- **Source**: plan-reviewer-v3 HIGH (H-v3-1).
- **Verbatim reviewer**:
  > The plan asserts at L378-380 that the recommended disambiguator
  > (`'O_DIRECTORY' in a`) is "stable" because `_fsync_dir` opens with
  > `os.O_RDONLY | os.O_DIRECTORY`. Verification:
  > `max_quality/src/moe_compress/utils/atomic_io.py:125` uses bare
  > `os.open(str(directory), os.O_RDONLY)` — NO `O_DIRECTORY`.
  > Empirical strace run confirms the parent-dir openat decodes as
  > `O_RDONLY|O_CLOEXEC`, not `O_RDONLY|O_DIRECTORY`.
- **Path picked** (user, 2026-05-29): **(b)** — add the production
  flag rather than complicate the test predicate.
- **Rationale**: cheapest fix; the codebase already uses the exact
  `getattr(os, "O_DIRECTORY", 0)` idiom at
  `max_quality/src/moe_compress/router_kd/plugins/early_stop.py:168`
  (verified present 2026-05-29 via
  `grep -n 'O_DIRECTORY' max_quality/src/`). Strict improvement:
  fail-fast on non-directory paths via `ENOTDIR`. Makes the strace
  disambiguation predicate (option (2) in the fd-disambiguation guide)
  factually correct.
- **Changes**:
  1. New commit #2 in §"Commit structure":
     `feat(atomic_io): _fsync_dir uses O_DIRECTORY for portable, fail-fast directory fsync`.
     1-line production change at
     `max_quality/src/moe_compress/utils/atomic_io.py:125`. Commit
     count: 9 → 10.
  2. All subsequent commits renumbered (#2→#3, #3→#4, ..., #9→#10).
     Cross-references updated at lines 43, 785, 791-792, 802, 814,
     836, 864, 869, and the v2-ledger §"Updated in v2" annotations.
  3. §"Predicate authoring guide (fd disambiguation)" — the
     "stable disambiguator" claim now references the new commit and
     notes the prior-state correction (old behavior: bare `O_RDONLY`,
     `'O_DIRECTORY' in a` was FALSE in practice).

### Fold 2 (LOW, L-v3-1): option (2) example predicate tightened

- **Source**: plan-reviewer-v3 LOW (L-v3-1).
- **Verbatim reviewer**:
  > Option (2)'s example predicate at L366
  > `lambda a: a.endswith('.tmp", O_RDONLY|O_CLOEXEC')` is brittle to
  > Python's default flag-set on `os.open`. Empirically it works on
  > Python 3.8+ (O_CLOEXEC is the default since Python 3.4), but a
  > more conservative predicate like `lambda a: '.tmp' in a` (matching
  > the path only) would survive future Python or libc changes.
- **Change**: example predicate switched from
  `a.endswith('.tmp", O_RDONLY|O_CLOEXEC')` to `'.tmp' in a`
  (path-only match) with an explanatory paragraph after the code
  block citing the L-v3-1 origin.

### Fold 3 (NIT, N-v3-1): fixture docstring fsync-predicate examples corrected

- **Source**: plan-reviewer-v3 NITPICK (N-v3-1).
- **Verbatim reviewer**:
  > The pre-existing fixture docstring at L194-197 uses
  > `("fsync", lambda args: "tmp" in args)` and
  > `("fsync", lambda args: "parent_dir_fd" in args)`. The new
  > authoring guide at L329-330 correctly observes that `fsync(N)`
  > has no path arg — so the fixture's own docstring example is wrong
  > in the same way the guide warns against.
- **Change**: fixture docstring now leads with an explanatory NOTE
  about `fsync(N)` having no path argument, and the Usage block
  example now mirrors the canonical 5-step pattern from the predicate
  authoring guide (openat-tmp → fsync → rename → openat-parent →
  fsync), using `lambda a: True` for the FD-based fsync predicates
  and path-only/flag-only matches for the surrounding openats.

### What did NOT change from v3

- The H-1 site inventory (11 sites).
- The H-1 helper inlining strategy (per-writer `_atomic_torch_save`).
- The H-2 design (per-(rank, expert) `torch.Generator`, payload
  `generator_states`, soft back-compat WARN).
- The MANIFEST.md re-bump as the final commit.
- All "Risk" entries (#1–#7).
- The LoC totals for H-1 / H-2 / fixture / migrations (the new
  commit adds ~6 LoC outside the patch wheel; this is below the
  table's rounding granularity but called out in the new commit's
  body).
- The two violator migrations (commits #8 + #9).

### Raises (per `feedback_raise_dont_substitute.md`)

None pending. The user's choice of path (b) over (a) for H-v3-1 was
explicit; L-v3-1 and N-v3-1 are mechanical folds with no engineering
ambiguity. **One nit to flag, not a raise**: per the user's protocol
"between commit #1 and commit #2 — or appended at the end if cleaner",
this v4 picked the **between #1 and #2** position because (a) the
strace tests in commit #6 reference `O_DIRECTORY` in their predicates,
so the production fix must land before the tests for the loop-close
verification step (`pytest tests/test_calibration_*_smoke.py` after
every commit) to succeed; (b) the new commit is production code, not
patch code, so grouping it before the wheel-patch sweep keeps the
commits' subsystems contiguous. The alternative "appended at the end"
position would have made the test-as-evidence step fail at commit #6
until the very last commit landed, breaking per-commit validation.
