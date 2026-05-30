# PLAN — Calibration durability rebuild: H-1 (fsync) + H-2 (output_reservoir RNG determinism)

Status: PLAN ONLY. Rebuild of work that lived in a `/tmp` scratch dir, never committed,
destroyed in the 2026-05-29 systemd-oomd crash. No production code is changed by this doc.

All line numbers below are **patch line numbers** in the canonical patch
`max_quality/patches/vllm_calibration_hooks.patch`
(MD5 `a8da5e321ac7fb30f1648fba3476bea6`, 10101 lines) unless prefixed with `[stage2]`,
in which case they are in the sibling patch
`max_quality/patches/vllm_calibration_stage2_profile.patch`
(MD5 `1bc773701efd5c4339e9a499519fe564`, 679 lines).

Both patches are unified diffs applied on top of upstream vLLM. The functional change
lives in the patched vLLM source files — **no monkey-patching** (project law). For each
H-1/H-2 edit we modify the `+` body inside the existing hunk, which grows the hunk's
`+N` line count in the `@@` header; the hunk header counts must be recomputed when the
edit is applied (see "Hunk-header arithmetic" below).

---

## 0. Verified inventory of the write sites (read against the real patch)

### 0.1 Writer modules present in the hooks patch
Eight calibration-writer source files are ADDED by the hooks patch (each has its own
`diff --git a/vllm/calibration_*.py` header):

| Module | header @ | imports `os` @ | imports `struct` |
|---|---|---|---|
| `calibration_block_outputs.py` | 5057 | 5124 | no |
| `calibration_imatrix.py` | 5949 | 6074 | yes (6076) |
| `calibration_input_cov.py` | 6933 | 7011 | no |
| `calibration_output_reservoir.py` | 7353 | 7439 | no |
| `calibration_per_expert_max.py` | 7800 | (os used) | no |
| `calibration_reap_scores.py` | 8134 | (os used) | no |
| `calibration_router_logits_stats.py` | 8471 | 8544 | no |
| `calibration_routing_stats.py` | 8922 | 8983 | no |

`calibration_hooks.py` (header @ 5553) is the shared registry; it has NO checkpoint
writer of its own (verified: zero `os.replace`/`torch.save` inside its hunks).
Every writer module already imports `os`, so the directory-fsync calls (`os.open`,
`os.fsync`, `os.close`) need NO new import.

### 0.2 The 11 durability sites — exact `os.replace(...)` patch lines

There are **10 distinct atomic-rename writers** plus the imatrix module owning **2** of
them, giving the "11 sites" framing (imatrix counted twice: its `.dat` final writer and
its `.ckpt` periodic writer). Enumerated:

**A. Periodic-checkpoint writers (`dump_*_checkpoint`, `torch.save(payload, tmp)` → `os.replace`).** These are the highest-risk sites: a torn/zero-length `.ckpt` that survives a rename is silently loaded on `--resume` and corrupts the whole run.

| # | Module | function (def @) | `torch.save` @ | `os.replace` @ |
|---|---|---|---|---|
| 1 | block_outputs | `dump_block_outputs_checkpoint` @ 5478 | 5502 | **5503** |
| 2 | imatrix | `dump_imatrix_checkpoint` @ 6765 | 6823 | **6824** |
| 3 | input_cov | `dump_input_cov_checkpoint` @ 7294 | 7311 | **7312** |
| 4 | output_reservoir | `dump_output_reservoir_checkpoint` @ 7735 | 7755 | **7756** |
| 5 | per_expert_max | `dump_per_expert_max_checkpoint` @ 8089 | 8103 | **8104** |
| 6 | reap_scores | `dump_reap_scores_checkpoint` @ 8426 | 8440 | **8441** |
| 7 | router_logits_stats | `dump_router_logits_stats_checkpoint` @ 8856 | 8878 | **8879** |
| 8 | routing_stats | `dump_routing_stats_checkpoint` @ 9207 | 9223 | **9224** |

**B. imatrix `.dat` final writer (SPECIAL CASE — raw binary, not torch.save).**

| # | Module | function (def @) | write path | `os.replace` @ |
|---|---|---|---|---|
| 9 | imatrix | `_write_dat(...)` @ ~6688 | `with open(tmp,"wb") as f: f.write(...)` lines 6710–6722 | **6723** |

`_write_dat` already holds the file object `f` in a `with` block, so its fsync is the
cleanest of all sites (the fd is in scope). Body verified at 6709–6723:
```
tmp = path + ".tmp"
with open(tmp, "wb") as f:
    f.write(struct.pack("<i", len(entries)))
    ... struct/tobytes writes ...
    f.write(dataset_b)
os.replace(tmp, path)
```

**C. stage2_profile periodic checkpoint (sibling patch).**

| # | Module | function (def @) | `torch.save` @ | `os.replace` @ |
|---|---|---|---|---|
| 10 | stage2_profile | `dump_stage2_profile_checkpoint` @ `[stage2]`536 | `[stage2]`562 | `[stage2]`**563** |

stage2 uses a `Path` tmp: `tmp = path.with_suffix(path.suffix + ".tmp")` (`[stage2]`540),
and does `path.parent.mkdir(parents=True, exist_ok=True)` (`[stage2]`561) before the save.
It imports `os` (`[stage2]`76) and `torch` (`[stage2]`81).

**Out of scope (NOT durability sites):** the final `dump_*` sidecar writers
(`dump_block_outputs` @ 5338, `dump_input_cov` @ 7250, `dump_output_reservoir` @ 7666,
`dump_per_expert_max` @ 8035, `dump_reap_scores` @ 8381, `dump_router_logits_stats` @ 8790,
`dump_routing_stats` @ 9156) delegate the actual disk write to
`moe_compress.utils.cached_calibration_signals.save_*`, which lives in the moe_compress
repo, NOT in these patches. Their durability is a separate concern in that module and is
explicitly EXCLUDED from this plan. `dump_imatrix` @ 6582 is in-scope only via the
`_write_dat` helper it calls (site #9).

---

## 1. H-1 — fsync durability design

### 1.1 The bug
Current pattern at every site: `write tmp` → `os.replace(tmp, final)` with **no fsync**.
On power-loss / OOM-kill between the kernel buffering the `tmp` write and the rename
reaching disk, two failure modes survive:
- a **torn** `final` (rename committed, data pages not yet flushed) — looks valid, loads garbage;
- a **zero-length / partial** `final` after the directory entry is durable but file data is not.

`torch.save` returns once the Python buffer is handed to the OS; it does **not** fsync.
`open(...,"wb")` + `f.write` likewise only reaches the page cache. The rename's metadata
and the file's data can be reordered by the filesystem across a crash.

### 1.2 The fix (durable atomic write)
Required ordering (project law): **`tmp → fsync(file) → os.replace → fsync(dir)`**.

For each site:
1. After fully writing `tmp` and BEFORE `os.replace`: open/obtain the tmp fd, `f.flush()`,
   `os.fsync(f.fileno())`. This forces the tmp file's data+metadata to stable storage.
2. `os.replace(tmp, final)` (atomic rename, unchanged).
3. After the replace: fsync the **containing directory** so the rename itself is durable:
   ```
   dir_fd = os.open(os.path.dirname(final) or ".", os.O_RDONLY)
   try:
       os.fsync(dir_fd)
   finally:
       os.close(dir_fd)
   ```

`torch.save(payload, tmp)` closes its file internally, so for the torch.save sites we
cannot fsync the same open handle. Two viable shapes:
- **(preferred) pass a file object to torch.save**: `with open(tmp,"wb") as f: torch.save(payload, f); f.flush(); os.fsync(f.fileno())`. `torch.save` accepts a file-like; this keeps the fd in scope for the fsync and is the minimal, idiomatic change.
- (rejected) reopen `tmp` read-only after `torch.save` solely to fsync — an extra open and racier; only use if a torch version rejects file-object args (it does not for the pinned vLLM torch).

For `_write_dat` (site #9): the fd is already in the `with open(tmp,"wb") as f:` block —
add `f.flush(); os.fsync(f.fileno())` as the last statements **inside** the `with`, before
the dedent to `os.replace`.

### 1.3 Inline-per-writer vs shared helper — DECISION
A prior design chose **inline-per-writer** fsync (no shared helper in `calibration_hooks.py`).
**Confirm and KEEP inline, with one refinement.** Rationale:

- The eight checkpoint writers + stage2 each already construct `tmp`, call `torch.save`,
  and `os.replace` inline. A shared helper would have to abstract over (a) torch.save vs
  raw binary, (b) `str` vs `Path` tmp construction, (c) the per-module env-gate guards.
  That abstraction is wider than the duplication it removes.
- A shared helper in `calibration_hooks.py` would create a **new import edge**
  (`calibration_imatrix` → `calibration_hooks`, etc.) for several writers that today do
  NOT import the hooks registry for their dump path. That widens the patch's blast radius
  and the risk surface for a follow-up rebase against upstream vLLM.
- The fsync sequence is ~4 lines; inlining keeps each writer's dump function
  self-contained and independently reviewable against its own existing kill-mid-write test.

**Refinement (improvement on the lost design):** factor ONLY the directory-fsync into a
tiny **module-private** helper *within each writer file* is rejected (would re-duplicate
the helper 9×). Instead, define a single 5-line helper `_fsync_dir(path)` **once per
module is also rejected**. Final decision: **pure inline at all 10 sites**, identical
4-line block, because (a) it is the smallest diff, (b) it keeps each module import-graph
unchanged, (c) the block is trivial enough that duplication carries no real maintenance
cost, and (d) it matches the lost design the user already reviewed. The directory-fsync
swallows `OSError` from filesystems that don't support directory fsync (rare; e.g. some
network FS) by logging at debug and continuing — the file-fsync already gave durability;
a dir-fsync failure must not abort a calibration run.

### 1.4 Per-site hunk shape

For the eight `torch.save` checkpoint writers (sites 1–8) and stage2 (site 10), replace:
```
    tmp = <ckpt_path + ".tmp"  |  path.with_suffix(...)>
    torch.save(payload, tmp)
    os.replace(tmp, <final>)
```
with:
```
    tmp = <same tmp expression>
    with open(tmp, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, <final>)
    _dir_fd = os.open(os.path.dirname(<final>) or ".", os.O_RDONLY)
    try:
        os.fsync(_dir_fd)
    except OSError:
        pass
    finally:
        os.close(_dir_fd)
```
(stage2 site uses `os.path.dirname(str(path))`; `path.parent.mkdir(...)` already runs
above at `[stage2]`561 so the dir exists before the dir-fsync.)

For `_write_dat` (site #9), the existing `with open(tmp,"wb") as f:` block already holds
the fd. Insert, as the LAST two statements inside the `with` (after the final
`f.write(dataset_b)` at 6722, before the dedent):
```
        f.flush()
        os.fsync(f.fileno())
```
then after the existing `os.replace(tmp, path)` (6723) append the same directory-fsync
block keyed off `path`.

### 1.5 Hunk-header arithmetic
Each writer is a `new file mode 100644` diff, so its single hunk header is
`@@ -0,0 +1,N @@`. Adding `k` lines to a module bumps that hunk to `@@ -0,0 +1,N+k @@`:
- torch.save sites: the new block is **+12 lines net** over the old 3 (old: `tmp=`, `torch.save`, `os.replace` → kept `tmp=`+`os.replace`, added `with`/`torch.save(f)`/`flush`/`fsync` = 4, added 6-line dir-fsync, +`with` indent of torch.save). Recompute exactly when applying; the count below is the editing target, the `@@` arithmetic is mechanical.
- For each affected module file, after editing, set `+1,N` to the new physical added-line count. Because these are whole-new-file diffs, `git apply --check` will reject a wrong count — that is the verification gate (see §4).
- stage2 is ALSO a whole-new-file diff — a single hunk `@@ -0,0 +1,673 @@` adding
  `vllm/calibration_stage2_profile.py` (there is NO `-A,B +C,D` context header). Bump its
  `+1,N` count by the number of added lines exactly as sites 1–9; verify with `git apply --check`.

---

## 2. H-2 — output_reservoir Phase-2 RNG determinism across resume

### 2.1 Current (buggy) state — verified
`calibration_output_reservoir.py` `_on_expert_out_unweighted` (def @ 7575) does Phase-2
sampling with the **global** generator (lines 7651, 7653):
```
u = torch.rand(n_remaining)                # GLOBAL stream
accept_mask = u < probs
slots = torch.randint(0, cap, (n_remaining,))   # GLOBAL stream
```
On kill+resume the global `torch.rand`/`torch.randint` stream restarts (or is wherever
the resumed process's global RNG happens to be), so the resumed segment's accept/slot
decisions diverge from the single-run counterfactual. The checkpoint
(`dump_output_reservoir_checkpoint` @ 7735, payload dict @ 7743) currently carries NO RNG
state; `load_output_reservoir_checkpoint` (@ 7759) restores reservoirs/seen/valid_count
but nothing RNG-related.

### 2.2 Design — per-(rank,expert) `torch.Generator(device="cpu")`

**New module-level state** (next to `_RESERVOIR`/`_SEEN`/`_VALID_COUNT`, declared near
patch lines 7472–7474):
```
_GEN: dict[tuple[int, int], torch.Generator] = {}   # (rank, e) -> CPU generator
```

**Seeding (deterministic, per cell).** When a `(rank, e)` cell is first allocated in
`_on_expert_out_unweighted` (alongside the lazy `_RESERVOIR.get(key) is None` branch at
7622–7627), create and seed its generator:
```
gen = _GEN.get(key)
if gen is None:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_cell_seed(rank, e))
    _GEN[key] = gen
```
`_cell_seed(rank, e)` is a pure function of the cell index plus a fixed run salt so two
different cells never share a stream and the seed is reproducible:
```
_RESERVOIR_RNG_BASE_SEED = int(os.getenv("VLLM_CALIB_OUTPUT_RESERVOIR_SEED", "0x5eed"), 0)  # base-0: auto-detects 0x prefix; int("0x5eed") without base would raise
def _cell_seed(rank, e):
    # 64-bit mix; deterministic, cell-unique, independent of dispatch order.
    return (_RESERVOIR_RNG_BASE_SEED * 0x9E3779B97F4A7C15
            + (rank << 20) + e) & 0x7FFFFFFFFFFFFFFF
```

**Draw sites (7651, 7653)** become per-cell:
```
u = torch.rand(n_remaining, generator=gen)
...
slots = torch.randint(0, cap, (n_remaining,), generator=gen)
```
Order within a dispatch is preserved EXACTLY (`rand` then `randint`) — this ordering is
load-bearing for byte-identity (see §3).

**Serialize into the checkpoint** (`dump_output_reservoir_checkpoint`, payload dict @
7743–7752). Add a key holding each live generator's state. `generator.get_state()` returns
a CPU `uint8` ByteTensor (~5056 bytes for the MT19937 CPU engine). Clone it so a later
draw cannot mutate the serialized snapshot:
```
    "generator_state": {k: g.get_state().clone() for k, g in _GEN.items()},
    "rng_base_seed": _RESERVOIR_RNG_BASE_SEED,
```
`torch.save` already serializes tensors, so the ByteTensor state rides along natively.

**Restore on load** (`load_output_reservoir_checkpoint` @ 7759; the `_RESERVOIR.clear()` /
hydrate block @ 7789–7798). Add a `_GEN.clear()` and a restore loop:
```
    _GEN.clear()
    saved_base = int(payload.get("rng_base_seed", _RESERVOIR_RNG_BASE_SEED))
    if saved_base != _RESERVOIR_RNG_BASE_SEED:
        raise ValueError(
            f"output_reservoir checkpoint rng_base_seed={saved_base} != "
            f"current {_RESERVOIR_RNG_BASE_SEED}; the Phase-2 stream would "
            f"diverge. Delete the checkpoint or restore VLLM_CALIB_OUTPUT_RESERVOIR_SEED."
        )
    for k, state in payload.get("generator_state", {}).items():
        rk = (int(k[0]), int(k[1]))
        g = torch.Generator(device="cpu")
        g.set_state(state.to(torch.uint8))   # state is the serialized ByteTensor
        _GEN[rk] = g
```
The `rng_base_seed` cross-check joins the existing `max_tokens` cross-check (@ 7773–7782)
and `schema_version` check (@ 7766–7772). Bump `_CHECKPOINT_SCHEMA_VERSION` (@ 7482) from
`1` → `2` because the payload schema changes; the existing schema guard at 7767 then
forces a clean restart against any v1 checkpoint instead of silently loading one with no
RNG state.

**Cells that never reached Phase-2** still get a generator at first allocation, so a cell
that is fill-only at checkpoint time but crosses into Phase-2 after resume continues from
its seeded-but-undrawn stream — identical to the single run, where that same generator
would also be undrawn until the first Phase-2 dispatch. (A generator that exists but has
never been drawn has state == freshly-seeded state; `get_state`/`set_state` round-trips it
exactly.)

### 2.3 Why per-(rank,expert) and not one global Generator
Empirically confirmed (see §3, experiment 5): a per-cell generator isolates each expert's
stream so cross-expert dispatch interleaving (which CAN differ run-to-run if batching
differs) does not perturb a given cell's draws. A single shared generator would make
cell A's stream position depend on how many draws cells B, C, … made before it — i.e. on
global dispatch order — which is NOT guaranteed stable across a resume. Per-cell isolation
is what makes the byte-identity invariant in §3 actually hold.

---

## 3. C-1 — the torch CPU Generator stream-semantics analysis (DESIGN GATE)

### 3.1 The disputed claim
A prior reviewer flagged a test `test_two_segment_additivity_phase_2_byte_identical`
(asserting kill+resume in the Phase-2 sampling regime is byte-identical to a single run)
as "mathematically wrong." Note: the **current** patch does NOT contain that test — its
output_reservoir smoke suite (added @ 2868) only has `test_two_segment_additivity_fill_only`
(@ 3148) and `test_accumulator_math_fill_only` (@ 2965). The disputed test was part of the
lost `/tmp` work. We must decide what the rebuilt Phase-2 test should assert.

### 3.2 Empirical findings (run on this repo's torch, CPU)
Five experiments (commands reproduced in §3.4):

1. **Single batched draw split via get_state/set_state is byte-identical.**
   `rand(10)` == `rand(4)` then `[get_state→fresh gen→set_state]` `rand(6)`. → **True**
2. **Batched == per-element.** `rand(5)` == five successive `rand(1)`. → **True.**
   (CPU MT19937 consumes the stream one element at a time; batch size is irrelevant to
   which numbers come out, only to how many.)
3. **rand-then-randint split AT A DISPATCH BOUNDARY, SAME grouping, via get/set_state is
   byte-identical.** Dispatch seq `[3,5]` (rand+randint per dispatch), checkpoint between
   the two dispatches, restore, continue → **True.**
4. **Same op type splits contiguously.** `rand(8)` == `rand(3)+rand(5)`; `randint(8)` ==
   `randint(3)+randint(5)`. → **True** for each.
5. **Per-cell generator isolates streams.** With a per-(rank,e) generator, expert-0's
   draws depend ONLY on expert-0's dispatch sequence, regardless of interleaving with
   other experts. Same expert-0 dispatch seq `[4,7]`, checkpoint between, restore →
   **True.**
6. **THE FAILURE MODE.** `rand(8),randint(8)` (one dispatch of 8 tail tokens) ≠
   `rand(3),randint(3),rand(5),randint(5)` (the SAME 8 tail tokens regrouped into two
   dispatches of 3+5). → **False.** Reason: the interleaving ORDER of `rand` vs `randint`
   draws differs. Single dispatch draws all 8 u's then all 8 slots; the regrouped version
   draws 3 u's, 3 slots, 5 u's, 5 slots. Different consumption order ⇒ different numbers.

### 3.3 VERDICT
**Phase-2 sampling resume IS byte-identical to the single-run counterfactual — IF AND
ONLY IF (a) each cell uses its own CPU `torch.Generator` whose `get_state` is serialized
and restored via `set_state`, AND (b) the per-cell sequence of `n_remaining` values
(i.e. the dispatch grouping for that expert) is identical between the split run and the
single run.** A checkpoint is always taken BETWEEN dispatches (at a prompt/chunk
boundary), never mid-dispatch, so condition (a)+(b) is exactly the resume scenario when
the same calibration prompts are replayed in the same order — which is the contract the
driver already guarantees (it resumes by re-feeding the not-yet-processed tail of the
fixed prompt list). Under that contract, byte-identity holds (experiments 3 + 5).

The reviewer was RIGHT about the *original* implementation: with the **global** RNG
(current code) the stream diverges and byte-identity is false. The reviewer's "mathematically
wrong" applies to the global-RNG version and to ANY claim of byte-identity under *changed
dispatch grouping* (experiment 6). It does NOT hold against the corrected per-cell-Generator
design under the same-grouping resume contract. So the resolution is **neither** "assert the
false claim" **nor** option (c) "drop the byte-identity guarantee" (which the user rejected):
it is to **make the guarantee true** (H-2 per-cell Generator) and test it under its actual
precondition (same dispatch grouping), while ALSO keeping the fill-only test for the
no-randomness regime.

This is the merits-based reconstruction of the user's "option b": keep a byte-identity
assertion, but scoped to the regime where it is provably true (per-cell generator restored
+ identical per-cell dispatch grouping), not the unconditional cross-regrouping claim the
reviewer correctly rejected.

### 3.4 Test plan (rebuilt output_reservoir smoke suite)

Keep the three existing always-true tests unchanged:
- `test_accumulator_math_fill_only` (@ 2965) — Phase-1 fill, byte-equal to input slice.
- `test_checkpoint_round_trip` (@ 3076) — extend to also assert `_GEN` round-trips
  (assert `or2._GEN[(rank,e)].get_state()` equals the dumped state for a populated cell).
- `test_two_segment_additivity_fill_only` (@ 3148) — fill-only split, exact, no RNG.

ADD two new tests:

**T-A `test_two_segment_additivity_phase_2_byte_identical` (the corrected version).**
Construct a single expert that DOES cross into Phase-2 (e.g. cap=4, feed 10 tokens all
routed to expert 0 across a dispatch sequence `[3, 4, 3]`). Run once as the reference
(three `_on_expert_out_unweighted` calls). Then run the SAME three dispatches split by a
checkpoint+reload after dispatch 2 (so the split is at a dispatch boundary, grouping
identical). Assert:
```
torch.equal(seg2._RESERVOIR[(0,0)], expected_reservoir)   # byte-identical
seg2._SEEN[(0,0)] == expected_seen
seg2._VALID_COUNT[(0,0)] == expected_valid
torch.equal(seg2._GEN[(0,0)].get_state(), ref._GEN[(0,0)].get_state())
```
This assertion is TRUE by §3.3 (per-cell generator + identical dispatch grouping +
get_state/set_state). The test docstring MUST state the precondition explicitly ("byte
-identity holds because the per-cell generator state is serialized and the split is taken
at a dispatch boundary so the per-expert dispatch grouping is unchanged"), so a future
reader does not mistake it for the false unconditional claim.

**T-B `test_phase_2_resume_diverges_under_regrouping` (negative / guard test).**
Document the boundary of the guarantee: take the same total tail tokens but REGROUP the
dispatches across the split (single run `[8]`, resume `[3]+[5]`). Assert the reservoir is
NOT required to be byte-identical (it generally differs). This pins experiment-6 behavior
so nobody later "fixes" it by asserting false byte-identity. (Assert a structural
invariant instead: `seen` and `valid_count` ARE additive/equal even though the sampled
slot contents differ.)

Optionally ADD **T-C `test_generator_isolation_across_experts`**: two experts both in
Phase-2; assert expert-0's reservoir is unchanged whether or not expert-1 received any
dispatches in between (per-cell isolation, experiment 5).

**Reproduction commands (verified outputs):**
- EXP1 split-batched byte-identical → True
- EXP2 batched == per-element → True
- EXP3 dispatch-boundary same-grouping rand+randint via get/set_state → True
- EXP4 / EXP6 regrouped one-dispatch(8) vs two-dispatch(3+5) → False
- EXP5 per-cell isolation, same grouping `[4,7]` → True
- `rand(8)`==`rand(3)+rand(5)` → True; `randint(8)`==`randint(3)+randint(5)` → True
- `torch.rand` default dtype → `torch.float32`

### 3.5 Determinism caveats (must be documented in code)
- **CPU only.** `torch.Generator(device="cpu")` + reservoirs on CPU. Do NOT use a CUDA
  generator: cross-GPU-arch (H200 vs RTX 6000 Pro) CUDA RNG streams are NOT guaranteed
  bit-identical, and the checkpoint is explicitly device-agnostic (it is dumped/hydrated
  across machines per the imatrix comment @ 6770–6771). CPU MT19937 is portable.
- **dtype.** `torch.rand(..., generator=gen)` yields float32 by default (verified). The
  comparison `u < probs` with `probs` float32 (built @ 7646–7650) is consistent. Do not
  introduce a float64 path.
- **Op order is load-bearing.** `rand` MUST precede `randint` within a dispatch and the
  pattern must be identical on the resume side (it is the same function). Any future
  reordering or fusing of the two draws breaks byte-identity (experiment 6 logic).
- **No mid-dispatch checkpoints.** The byte-identity guarantee assumes checkpoints are
  taken only between `_on_expert_out_unweighted` calls. This is true today (the driver
  checkpoints at prompt/chunk boundaries, never inside a kernel callback). Document this
  as an invariant the driver must preserve.

---

## 4. Build sequence (ordered commits) + verification

Work on branch `plan/calib-h1-h2-durability` is PLAN ONLY. The IMPLEMENTATION (separate
session) proceeds in this order; each commit ends with the Co-Authored-By trailer.

**Commit 1 — H-1 fsync, the 8 torch.save checkpoint writers (sites 1–8).**
Edit each `dump_*_checkpoint` body to the §1.4 shape. Recompute each whole-file hunk's
`@@ -0,0 +1,N @@`.
Verify: `git apply --check max_quality/patches/vllm_calibration_hooks.patch` against a
clean upstream vLLM checkout (the hunk-count gate). Then apply and run the per-writer
kill-mid-write tests already in the patch (e.g. @ 2181 "torch.save failure mid-checkpoint
must NOT corrupt a prior", @ 308/2052/2539/3108/3547/3936/4525/4933 the
`assert not os.path.exists(ckpt + ".tmp")` tmp-cleanup asserts).

**Commit 2 — H-1 fsync, imatrix `.dat` writer (site 9).**
Add `f.flush(); os.fsync(f.fileno())` inside the `_write_dat` `with` block + dir-fsync
after `os.replace`. Recompute imatrix hunk count.
Verify: `git apply --check`; run imatrix smoke tests (the imatrix test block in the patch).

**Commit 3 — H-1 fsync, stage2_profile (site 10, sibling patch).**
Edit `dump_stage2_profile_checkpoint`; recompute the whole-new-file hunk `@@ -0,0 +1,673 @@` → `+1,N` by the added-line count (same as sites 1–9; NOT a context hunk).
Verify: `git apply --check max_quality/patches/vllm_calibration_stage2_profile.patch`;
run stage2_profile smoke/round-trip tests.

**Commit 4 — H-2 per-cell Generator (output_reservoir).**
Add `_GEN`, `_cell_seed`, `_RESERVOIR_RNG_BASE_SEED`; seed at lazy alloc; switch draws to
`generator=gen`; serialize/restore in dump/load checkpoint; bump
`_CHECKPOINT_SCHEMA_VERSION` 1→2; add the `rng_base_seed` cross-check.
Recompute output_reservoir hunk count.
Verify: `git apply --check`; run the existing output_reservoir smoke suite (must still
pass: env-gate, clone-safety, payload-shape, fill-only additivity).

**Commit 5 — H-2 tests (corrected C-1 suite).**
Add T-A (`..._phase_2_byte_identical`, corrected/scoped), T-B
(`..._diverges_under_regrouping`), optional T-C (isolation); extend
`test_checkpoint_round_trip` to assert `_GEN` round-trip. Recompute the test-file hunk
count.
Verify: run the full output_reservoir smoke suite; T-A must pass (proving the guarantee),
T-B must pass (proving the boundary).

**Final full sweep** (proves each commit, in the same pytest rootdir / plugin-set as the
code under test — never a bare /tmp shell, per the reproduction-env-must-match lesson):
```
git apply --check max_quality/patches/vllm_calibration_hooks.patch
git apply --check max_quality/patches/vllm_calibration_stage2_profile.patch
# apply to a clean upstream vLLM checkout, then:
pytest tests/test_calibration_output_reservoir_smoke.py -v
pytest tests/test_calibration_imatrix*.py -v
pytest tests/test_calibration_*_smoke.py -v          # all writer smoke suites
pytest tests/test_calibration_stage2_profile*.py -v  # via the stage2 patch's tests
```
The `git apply --check` on BOTH patches is the primary gate: a wrong hunk-header count
fails it immediately, catching the single most likely mechanical error.

---

## 5. Risks / halt-triggers

- **R1 — wrong hunk-header counts.** Editing whole-new-file diffs requires recomputing
  `+1,N`. HALT and recompute if `git apply --check` reports "patch does not apply" /
  "corrupt patch at line N". Do not hand-fudge; recount added lines.
- **R2 — torch version rejects file-object torch.save.** If `torch.save(payload, f)` errors
  on the pinned torch, fall back to the reopen-and-fsync shape (§1.2 alt). Verify the
  pinned torch first; do not assume.
- **R3 — directory fsync unsupported on the target FS.** The `except OSError: pass` guard
  prevents a run abort; file-level fsync already gives the core durability. If the
  calibration output lives on a network FS where even file fsync is a no-op, durability is
  weaker than on local disk — note this to the operator, do not silently assume safety.
- **R4 — schema bump strands existing checkpoints.** `_CHECKPOINT_SCHEMA_VERSION` 1→2
  means any in-flight v1 output_reservoir `.ckpt` from before this change will be REJECTED
  on resume (clean restart of that signal only). This is intended (v1 has no RNG state and
  cannot satisfy the byte-identity guarantee). Confirm no production run is mid-flight on a
  v1 output_reservoir checkpoint before merging; if one is, let it finish first.
- **R5 — driver checkpoints mid-dispatch.** The byte-identity guarantee assumes checkpoints
  land only between `_on_expert_out_unweighted` calls. If a future driver change checkpoints
  inside a callback, T-A breaks and the guarantee is void. The invariant is documented in
  §3.5; a driver change that violates it must update this analysis.
- **R6 — monkeypatch creep.** The existing `test_dump_payload_shape...` (@ 3253) uses
  `monkeypatch.setattr(ccs, "save_output_reservoir", ...)`. Per project law (no monkey
  patches on production code) the NEW tests T-A/T-B/T-C MUST NOT introduce monkeypatch;
  they drive the public functions (`_on_expert_out_unweighted`,
  `dump/load_output_reservoir_checkpoint`) and inspect module state directly, as the
  existing additivity tests already do. Do not "fix" the pre-existing monkeypatch as part
  of this work (out of scope) but do not propagate the pattern.
- **R7 — never use a CUDA generator** (§3.5). A reviewer or future contributor might
  "optimize" by moving the generator to GPU; that silently breaks cross-arch checkpoint
  portability. The CPU constraint is load-bearing, not incidental.

---

## Appendix — exact patch line references (quick index)
- Reservoir module: header 7353; `_RESERVOIR/_SEEN/_VALID_COUNT` 7472–7474; schema ver
  7482; `_on_expert_out_unweighted` 7575; lazy alloc 7622–7627; Phase-2 draws 7651
  (`torch.rand`) + 7653 (`torch.randint`); `dump_..._checkpoint` 7735 (payload 7743–7752,
  torch.save 7755, os.replace 7756); `load_..._checkpoint` 7759 (schema check 7766,
  max_tokens check 7773, clear/hydrate 7789–7798).
- imatrix: `_write_dat` 6688 (open 6710, last write 6722, os.replace 6723);
  `dump_imatrix_checkpoint` 6765 (torch.save 6823, os.replace 6824).
- Other checkpoint os.replace sites: 5503, 7312, 8104, 8441, 8879, 9224.
- stage2: `dump_stage2_profile_checkpoint` [stage2]536 (tmp 540, mkdir 561, torch.save
  562, os.replace 563).
- Existing tests: output_reservoir smoke header 2868; fill-only math 2965; checkpoint
  round-trip 3076; fill-only additivity 3148; clone-safety 3221; payload-shape 3253.
