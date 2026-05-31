# PLAN S-1 — Pattern O (manifest-last) Roll-out for the 10 Calibration Sidecars

**Status**: PLANNER deliverable. No code changes here.
**Repo**: `/home/lucas/ai/moe_compress` (main @ `6a31963` at planner read time).
**Branch the implementer will work on**: `fix/calib-sidecar-pattern-o-rollout` (new — feature branch off `main`).
**Author**: Planner (S-1).
**Last updated**: 2026-05-29.

## Note on the source audit

The orchestrator instruction references `tasks/AUDIT_CALIBRATION_COMPLETENESS_V2.md`.
That filename does NOT exist in this tree (and in no branch / no remote — verified by
`git log --all --oneline -- '*CALIBRATION_COMPLETENESS*'`). The semantically equivalent
docs that DO exist:

* `tasks/AUDIT_CALIBRATION_DURABILITY.md` (on branch `audit/calibration-durability`,
  commit `017e21b`) — the F-C-1 / F-S3-1 / F-RK-1 durability audit that motivated
  Pattern O in the first place.
* `tasks/PLAN_PLUGIN_14_sidecar_audit.md` — read-only audit of the sidecar inventory
  + per-plugin consumption; this is the canonical "what sidecars exist" reference.
* `tasks/calib_v2_writers_todo.md` — the campaign that landed the 10 writers we are
  about to rewrite.

The orchestrator instruction also names this scope concretely ("10 calibration-sidecar
writers in `cached_calibration_signals.py` are out-of-sync with Pattern O posture")
and lists the 10 pairs verbatim. The plan proceeds against that explicit list and
those source docs; the missing V2 audit filename is treated as a label, not a blocker.

If the implementer wants the V2 audit doc to literally exist, the obvious sequence
is (a) rename `AUDIT_CALIBRATION_DURABILITY.md` to `AUDIT_CALIBRATION_COMPLETENESS_V2.md`,
or (b) ship a thin V2 doc that points at this plan. NEITHER is on this plan's critical
path; flagging only.

---

## 1. Goal

Promote every calibration-sidecar writer in
`max_quality/src/moe_compress/utils/cached_calibration_signals.py` from
**"atomic-write only"** to the **full Pattern O posture: atomic-write + manifest-last
+ manifest-validated read**, so that:

* A SIGKILL between `torch.save(tmp)` and `os.replace(tmp, final)` leaves a stale
  `.tmp` (already handled).
* A SIGKILL between `os.replace(...)` and the next "I am done" signal leaves a
  payload **without** its sibling `MANIFEST.json` — readers detect that immediately
  (presumed torn) instead of silently consuming a partial file.
* A successful write produces `<payload>.<ext>` **and** `<payload>.<ext>.MANIFEST.json`,
  and the manifest is the LAST thing written.
* Older sidecars (written by the current "no manifest" code path) still load with
  a single WARNING via the back-compat fallback already used in `eora_inputs.py:199-243`
  and `router_kd/plugins/teacher.py:270-314`.

Pattern O is already implemented in `max_quality/src/moe_compress/utils/atomic_io.py`
(`write_manifest_last` + `read_and_validate_manifest` + `ManifestMismatchError`).
This plan threads it through the 10 outstanding pairs.

---

## 2. Scope — the 10 writer / reader pairs

Verified against `cached_calibration_signals.py` (functions, with file:line at planner
read time):

| # | Writer (save_*) | Reader (load_*) | Sidecar layout | Notes |
|---|---|---|---|---|
| 1 | `save_covariance` (1040) | `load_covariance` (1063) | `sidecars/<stem>/covariance.pt` | schema v2 |
| 2 | `save_routing_stats` (919) | `load_routing_stats` (935) | `sidecars/<stem>/routing_stats.pt` | schema v1; unwired consumer |
| 3 | `save_per_expert_max` (886) | `load_per_expert_max` (902) | `sidecars/<stem>/per_expert_max.pt` | schema v1 |
| 4 | `save_reap_scores` (853) | `load_reap_scores` (869) | `sidecars/<stem>/reap_scores.pt` | schema v1 |
| 5 | `save_router_logits_stats` (952) | `load_router_logits_stats` (985) | `sidecars/<stem>/router_logits_stats.pt` | schema v1 |
| 6 | `save_output_reservoir` (1002) | `load_output_reservoir` (1023) | `sidecars/<stem>/output_reservoir.pt` | schema v1; ~10-15 GB |
| 7 | `save_stage2_profile_v3` (683) | `load_stage2_profile_v3` (768) | `sidecars/<stem>/stage2_profile.pt` | schema v3; large (~40 GB cov + reservoir) |
| 8 | `save_block_hidden` (1137) | `load_block_hidden` (1148) | `sidecars/<stem>/block_hidden/layer_NNNN.pt` | per-layer shard |
| 9 | *(deleted, NIT-3)* | `load_phase_b` (664) | `sidecars/<stem>/phase_b.pt` | LEGACY back-compat only — no writer left |
| 10 | *(deleted, NIT-4)* | `load_router_kd_logits` (1083) | `sidecars/<stem>/router_kd_logits/NNNNNNN.npz` | LEGACY back-compat only — per-attempt-idx `.npz` |

Excluded from this plan (per orchestrator instruction): `save_teacher_eval` writer
was already deleted under NIT-5; its loader (`load_teacher_eval` at 1175) is back-
compat-only and treated identically to rows #9 and #10.

---

## 3. Per-pair migration plan

The general transform is the same for every "writer still alive" row (#1-#8). The
"loader-only" rows (#9, #10, plus `load_teacher_eval`) get a smaller transform —
they just learn to accept the new manifest opportunistically, with a fallback that
matches today's behaviour.

### 3.1 Writer template (applies to rows #1-#8)

**Today** (`save_routing_stats` exemplifies; every save_* follows the same shape):
```python
def save_routing_stats(payload: RoutingStatsPayload, jsonl_path: Path) -> None:
    cpu_payload = RoutingStatsPayload(...)  # CPU-cast every tensor field
    _atomic_torch_save(cpu_payload, sidecar_path(jsonl_path, "routing_stats"))
```

**After**:
```python
def save_routing_stats(payload: RoutingStatsPayload, jsonl_path: Path) -> None:
    cpu_payload = RoutingStatsPayload(...)  # unchanged
    path = sidecar_path(jsonl_path, "routing_stats")
    manifest_path = _manifest_path_for(path)  # see §3.3 helper
    _atomic_torch_save(cpu_payload, path)
    write_manifest_last(
        path,
        manifest_path,
        schema_version=SCHEMA_VERSIONS["routing_stats"],
        extra_meta={"artifact": "routing_stats"},
        compute_sha256=False,  # see §3.4 sha256 policy
    )
```

Key points:
* **Always pre-`unlink(missing_ok=True)` the manifest before the atomic_torch_save**
  (mirrors `stage3/orchestrator.py:474-478` and `wanda_intra_expert_score.py:565-567`).
  Rationale: if a previous run left a stale `<path>.MANIFEST.json` from a different
  payload (because we changed the calibration mix but the artifacts directory was
  shared), an interrupted re-write here would briefly look "good" to a reader after
  the new payload lands but before the new manifest is written. The unlink-first
  invariant means a torn re-write leaves NO manifest at all → reader fails loudly.
* **Manifest written LAST** — atomic_torch_save fsync's the payload before
  os.replace; write_manifest_last is itself atomic (delegates to atomic_json_save)
  and runs strictly after the payload's data blocks are durable.
* **`extra_meta`** carries the signal name as `artifact` so logs and forensic dumps
  can attribute a stray sidecar without parsing the path. Two existing call sites
  do this (stage3 originals, wanda_intra_expert_score); we copy the convention.
* **`compute_sha256=False`** is the default for the calibration sidecars (see §3.4
  for the policy). The stage3-originals (50 GB) currently uses `True`; this plan
  does NOT change that — only the new manifests written by `cached_calibration_signals.py`.

### 3.2 Reader template (applies to rows #1-#8)

**Today** (`load_routing_stats` exemplifies):
```python
def load_routing_stats(jsonl_path: Path) -> RoutingStatsPayload | None:
    path = _resolve_sidecar_for_load(jsonl_path, "routing_stats")
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("routing_stats", payload.schema_version, path)
    return payload
```

**After**:
```python
def load_routing_stats(jsonl_path: Path) -> RoutingStatsPayload | None:
    path = _resolve_sidecar_for_load(jsonl_path, "routing_stats")
    if path is None:
        return None
    _validate_manifest_or_warn(
        path,
        expected_schema_version=SCHEMA_VERSIONS["routing_stats"],
        signal_name="routing_stats",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _check_schema("routing_stats", payload.schema_version, path)
    return payload
```

Where `_validate_manifest_or_warn` is a new module-local helper (see §3.3).

* **In-payload schema check is PRESERVED** (`_check_schema` stays).
  Manifest-cross-check is an *additional* guard against torn writes; it does NOT
  replace the schema check. Together they defend against both
  (a) wrong-schema sidecars (caught by the dataclass version field) and
  (b) torn-write sidecars (caught by manifest absence or size mismatch).
* **Back-compat WARN fallback** (the eora_inputs.py:230-243 pattern):
  - Manifest exists → call `read_and_validate_manifest`; on `ManifestMismatchError`,
    raise a RuntimeError with the "Delete <payload> and <manifest> and re-run
    calibration" actionable message.
  - Manifest absent → log a one-shot WARNING per (signal, sidecar parent dir)
    keyed in a module-level dedupe set (mirror `_already_warned_legacy_paths`),
    then proceed with the in-payload schema check only. Tag the warning with
    `MEDIUM-8 TODO(post-2026-Q3)` exactly like the existing back-compat shims
    in stage4/router_kd so the removal cadence is uniform.

### 3.3 Module-local helpers

Add the following helpers to `cached_calibration_signals.py` (near
`_resolve_sidecar_for_load`, not in `atomic_io.py` — they encode this module's
back-compat policy):

```python
def _manifest_path_for(payload_path: Path) -> Path:
    """Sibling manifest path for a calibration sidecar.

    Convention: append ``.MANIFEST.json`` AFTER the payload's full suffix so
    the manifest sorts alphabetically right after the payload (matches
    Stage 3 originals + Stage 5 teacher logits + wanda_intra_expert_score
    sidecar conventions). Pattern O Hub-upload ordering relies on this.
    """
    return Path(str(payload_path) + ".MANIFEST.json")


_already_warned_missing_manifest: set[tuple[Path, str]] = set()


def _validate_manifest_or_warn(
    payload_path: Path,
    *,
    expected_schema_version: int,
    signal_name: str,
) -> None:
    """Pattern O validation with one-shot back-compat WARN fallback.

    Behavior:
    * Manifest exists → ``read_and_validate_manifest``. On failure, raise
      RuntimeError with the canonical "delete + re-run calibration"
      message and chain the ManifestMismatchError as cause.
    * Manifest absent → log a one-shot WARNING (deduped via
      ``_already_warned_missing_manifest`` keyed on
      ``(payload_path, signal_name)``) and return; the caller's existing
      in-payload schema check is the only torn-write guard for these
      legacy sidecars.

    The fallback exists so this fix does NOT invalidate calibration
    artifacts already on disk (~tens of GB per artifacts dir). Operators
    re-running calibration will get a manifest automatically on the next
    save_*; on the run after that, the warning will stop firing.

    MEDIUM-8 TODO(post-2026-Q3): drop the fallback once all in-flight
    sidecars under ``/opt/output/*`` have been regenerated. The horizon
    is intentionally identical to the Stage 4 / Stage 5 fallbacks
    (eora_inputs.py:230, teacher.py:301).
    """
    manifest_path = _manifest_path_for(payload_path)
    if manifest_path.exists():
        try:
            read_and_validate_manifest(
                payload_path,
                manifest_path,
                expected_schema_version=expected_schema_version,
            )
        except ManifestMismatchError as exc:
            raise RuntimeError(
                f"Calibration sidecar manifest validation FAILED for "
                f"signal={signal_name!r}: {exc}. Delete both "
                f"{payload_path.name} and {manifest_path.name} from "
                f"{payload_path.parent} and re-run calibration."
            ) from exc
        return
    _key = (payload_path, signal_name)
    if _key not in _already_warned_missing_manifest:
        log.warning(
            "Pattern O back-compat: sidecar %s has no MANIFEST.json "
            "(pre-S1 calibration run?). Proceeding with in-payload schema "
            "check only; next save_* call will emit a manifest. "
            "MEDIUM-8 TODO(post-2026-Q3): remove this fallback once all "
            "in-flight calibration sidecars have been regenerated.",
            payload_path,
        )
        _already_warned_missing_manifest.add(_key)
```

Notes:
* `_already_warned_missing_manifest` mirrors the existing
  `_already_warned_legacy_paths` set. Tests that need fresh process state can
  call `.clear()` between cases (same hook the F-H-7 tests already use).
* Imports needed at the top of the module: `ManifestMismatchError`,
  `read_and_validate_manifest`, `write_manifest_last` — all from
  `.atomic_io`.

### 3.4 SHA-256 policy

Stage 3 originals (~50 GB) uses `compute_sha256=True` because Stage 4 is the
correctness-critical reader and the operator may opt into a deep validation
flag. The 10 calibration sidecars are different:

* `output_reservoir.pt` (~10-15 GB) and `stage2_profile.pt` (~40 GB) — too large
  for a default SHA-256; the size + schema_version cross-check is the validation
  budget. Set `compute_sha256=False`.
* Everything else (`reap_scores`, `per_expert_max`, `routing_stats`,
  `router_logits_stats`, `covariance`, `block_hidden/layer_*.pt`) — small (MB to
  ~hundreds of MB). SHA-256 is cheap (~seconds at most). Could go either way.
  **Default: `compute_sha256=False`** for uniformity with the large sidecars and
  to keep the writer fast in the common case; the size + schema_version cross-
  check is sufficient torn-write detection. Operators with a security-sensitive
  flow can pass `require_sha256=True` to the loader and add a follow-up
  improvement to enable `compute_sha256=True` on the writer side (out of scope
  for this plan).

### 3.5 Loader-only rows (rows #9, #10, plus `load_teacher_eval`)

These have NO writer in this module (deleted in NIT-3/NIT-4/NIT-5). They MUST
detect manifest absence gracefully because every legacy file on disk lacks one:

* `load_phase_b` (row #9): no writer → no fresh sidecars are ever produced under
  the new posture. The loader stays as-is; we DO NOT add manifest validation
  because every disk hit is by definition pre-Pattern-O. If a future writer ever
  re-emerges, it MUST emit a manifest, and that future PR adds the validation
  call to this loader.
* `load_router_kd_logits` (row #10): same reasoning. The production `.npz`
  shards are written directly by `build_self_traces_calib_vllm.py` (see audit
  row C.3); F-C-1 fix landed there. Manifest-last on those is **out of scope for
  this plan** — they have their own atomic-write path inside the vLLM script.
  This loader stays as-is.
* `load_teacher_eval` (excluded per orchestrator note): same as #9 — Stage 6 has
  its own teacher-eval cache mechanism that's distinct from this sidecar layout.
  Loader stays as-is.

**Net effect for rows #9/#10/teacher_eval**: zero code change in this plan.

### 3.6 Per-pair table

| # | Pair | Writer change | Reader change | Test additions |
|---|---|---|---|---|
| 1 | `covariance` | + write_manifest_last (compute_sha256=False, extra_meta artifact="covariance") | + `_validate_manifest_or_warn` call | 4 tests (§7) |
| 2 | `routing_stats` | + write_manifest_last | + `_validate_manifest_or_warn` | 4 tests |
| 3 | `per_expert_max` | + write_manifest_last | + `_validate_manifest_or_warn` | 4 tests |
| 4 | `reap_scores` | + write_manifest_last | + `_validate_manifest_or_warn` | 4 tests |
| 5 | `router_logits_stats` | + write_manifest_last | + `_validate_manifest_or_warn` | 4 tests |
| 6 | `output_reservoir` | + write_manifest_last (compute_sha256=False) | + `_validate_manifest_or_warn` | 4 tests |
| 7 | `stage2_profile_v3` | + write_manifest_last (compute_sha256=False) — see §9 risk note | + `_validate_manifest_or_warn` | 4 tests |
| 8 | `block_hidden` | + write_manifest_last (per-layer manifest) | + `_validate_manifest_or_warn` (per-layer) | 4 tests + 1 multi-layer test |
| 9 | `phase_b` (loader-only) | — | — | — |
| 10 | `router_kd_logits` (loader-only) | — | — | — |

---

## 4. Manifest schema — fields each manifest carries

Decided by `write_manifest_last` itself; this plan does NOT change the schema.
Each calibration-sidecar manifest will contain:

```json
{
  "schema_version": <int — copied from SCHEMA_VERSIONS[signal]>,
  "payload_name": "<basename of sidecar.pt>",
  "size_bytes": <int — payload's size on disk at write time>,
  "sha256": null,                          // compute_sha256=False default
  "write_timestamp_iso": "<UTC ISO-8601>",  // forensics, never validated
  "extra": {
    "artifact": "<signal_name — covariance / routing_stats / ...>"
  }
}
```

`schema_version` is the calibration-signal's existing version (already declared
in `SCHEMA_VERSIONS` at lines 111-129). The reader cross-checks the manifest's
`schema_version` against `SCHEMA_VERSIONS[signal]` AND `_check_schema` continues
to cross-check the loaded payload's in-dataclass `schema_version` field; both must
agree with the expected version.

---

## 5. Backward-compat strategy

Older sidecars without a sibling `.MANIFEST.json` must still load. The contract is:

* **First read after upgrade**: manifest absent → one-shot WARNING per
  `(payload_path, signal_name)` (deduped in module-level set), then proceed with
  the existing in-payload `schema_version` check. Same posture as
  `eora_inputs.py:229-243` and `router_kd/plugins/teacher.py:300-314` — proven
  pattern, identical TODO horizon.
* **Subsequent runs**: any `save_*` call re-emits the sidecar AND its manifest.
  After that point the back-compat path is never taken for that artifacts dir.

**Removed writers (rows #9, #10, plus teacher_eval)**: their loaders are LEGACY-
ONLY by construction. Adding a manifest check here would deadlock the back-compat
read (every disk file is pre-Pattern-O). These loaders stay unchanged.

**Why NOT delete the old sidecars on upgrade?** The user runs calibration on
rented GPUs; an artifacts directory may hold 30-60 GB of sidecars representing
hours of teacher forward passes (per the audit's wall-clock numbers in
`PLAN_PLUGIN_14_sidecar_audit.md`). Forcing a re-capture would destroy live work.
The back-compat WARN preserves the artifact and lets the next write upgrade it.

---

## 6. Removed-writer hygiene check

The orchestrator instruction calls out that `save_phase_b` (NIT-3) and
`save_router_kd_logits` (NIT-4) are GONE; the loaders are retained for
back-compat reads of any legacy sidecars an operator may still have on disk.

* `load_phase_b` (664): no change. The function already detects missing files
  and returns None; for legacy hits, no manifest is expected.
* `load_router_kd_logits` (1083): no change. The function reads .npz shards via
  `np.load` and checks `schema_version` field inside the .npz. Adding manifest
  validation here would require writing one shard at a time, which the production
  writer (in `build_self_traces_calib_vllm.py`) doesn't do. Out of scope.
* `load_teacher_eval` (1175): no change (NIT-5).

**Implementer check**: confirm by `grep -n "^def save_phase_b\|^def save_router_kd_logits\|^def save_teacher_eval" max_quality/src/moe_compress/utils/cached_calibration_signals.py` returns nothing. (Verified by planner: returns nothing.)

---

## 7. Files to modify

### 7.1 Source changes (1 file)

* `max_quality/src/moe_compress/utils/cached_calibration_signals.py` — the entire
  scope of this plan. Touches:
  - Add 3 imports at the top: `ManifestMismatchError`,
    `read_and_validate_manifest`, `write_manifest_last` (already imports two
    helpers from `.atomic_io` — just extend the import group).
  - Add `_manifest_path_for` + `_already_warned_missing_manifest` +
    `_validate_manifest_or_warn` (~50 LoC of new helpers).
  - Modify 8 writers (rows #1-#8): add `path = ...; manifest_path = ...;
    manifest_path.unlink(missing_ok=True); ...; write_manifest_last(...)`.
  - Modify 8 readers (rows #1-#8): add `_validate_manifest_or_warn(path, ...)`
    immediately after `_resolve_sidecar_for_load` succeeds and before
    `torch.load`.
  - **NO change** to rows #9, #10, `load_teacher_eval`.
  - **NO change** to `SCHEMA_VERSIONS` — manifest carries the signal's existing
    version verbatim.
  - **NO change** to module docstring's atomic-write paragraph (it already
    references Pattern O; verify lines 19-31 still read correctly after the
    edit — if anything, this fix actually closes the gap the docstring already
    advertises).
  - Add 1 small docstring tweak in lines 19-31 to flip language from "the
    atomic dance is implemented" → "the atomic dance + manifest-last is
    implemented" (single-line edit; consistency only).

### 7.2 Test changes (1 file)

* `max_quality/tests/test_cached_calibration_signals.py` — extend the existing
  roundtrip / crash-safety tests to cover the new manifest pathway:
  - 8 new "manifest written sibling-to-payload" assertions (one per writer).
  - 8 new "torn-payload detected by size mismatch" tests (one per writer; mirror
    `test_stage3_originals_torn_payload_fails_loudly` in
    `test_stage3_originals_manifest.py`).
  - 8 new "missing-manifest WARN fallback still loads" tests (one per writer;
    confirm the legacy back-compat path works).
  - 1 new "warn-deduped" test confirming the WARNING fires at most once per
    `(payload, signal)`.
  - 1 new "block_hidden per-layer manifest" test confirming each layer shard has
    its own manifest.
  - PRESERVE every existing test (262-1099). The roundtrip + schema-version-
    mismatch tests stay green and now exercise the manifest cross-check
    incidentally.

### 7.3 Docs

* `max_quality/patches/MANIFEST.md` — append a one-line "Pattern O roll-out
  complete for 10 calibration sidecars (planner ref `tasks/PLAN_S1_PATTERN_O_CALIB_SIDECARS.md`)"
  entry under the existing schema-bumps section (no schema bump, but the policy
  change deserves a one-liner).
* `tasks/lessons.md` — capture the pattern-extension lesson (see §10 rollback
  plan; this is the post-merge update).

### 7.4 Files NOT touched

* `max_quality/src/moe_compress/utils/atomic_io.py` — already complete. No edits.
* `max_quality/src/moe_compress/stage3/orchestrator.py`,
  `max_quality/src/moe_compress/stage3/plugins/wanda_intra_expert_score.py`,
  `max_quality/src/moe_compress/router_kd/plugins/teacher.py`,
  `max_quality/src/moe_compress/stage4/plugins/eora_inputs.py`,
  `max_quality/hf_jobs/precompute_teacher_logits.py` — already Pattern O. No edits.
* `max_quality/patches/vllm_calibration_hooks.patch`,
  `max_quality/patches/vllm_calibration_stage2_profile.patch` — the production
  calibration writers live here, but they all dispatch through
  `cached_calibration_signals.save_*` (confirmed by `grep "save_block_hidden\|
  save_covariance\|save_reap_scores\|save_per_expert_max\|save_routing_stats\|
  save_router_logits_stats\|save_output_reservoir" max_quality/patches/vllm_calibration_hooks.patch`).
  **No patch regeneration needed** — the writer functions stay at the same module
  path, with the same signatures. Operators with the existing patch wheel pick up
  the new behavior the next time they `pip install` the moe_compress package.
* `max_quality/src/moe_compress/calibration/stage2_profile_writer.py` — the
  caller of `save_stage2_profile_v3`. The function signature is unchanged; the
  caller doesn't move. No edit.

---

## 8. Commit chunking strategy

**Recommendation**: ONE commit for the helper + all 8 writers, plus ONE commit for
all 8 readers, plus ONE test-only commit. Rationale:

* Per-pair commits (10 commits or 16 if we split each pair into writer+reader)
  would be a lot of `git log` noise for a 50-LoC mechanical pattern.
* But splitting writer-side and reader-side gives a clean revert lever: a writer
  bug (manifest fails to land) is isolatable from a reader bug (validation throws
  on a healthy artifact) by reverting the relevant commit.
* The test commit is separated because test failures should NOT block the
  source commit from landing — but they should land in the same merge and
  reviewers benefit from seeing the tests in isolation.

Concrete sequence (3 commits on `fix/calib-sidecar-pattern-o-rollout`):

1. **`feat(calib): Pattern O writers — manifest-last for 8 calibration sidecars`**
   - Add helpers (`_manifest_path_for`, `_already_warned_missing_manifest`,
     `_validate_manifest_or_warn`) — the validate helper is included here
     because import-time linting otherwise complains about an unused private.
   - Modify all 8 writers in rows #1-#8.
   - DO NOT touch the readers yet — they'll quietly ignore the new manifest
     until the next commit lands. This intermediate state is safe: existing
     `_check_schema` still runs, and a torn write at this intermediate version
     produces the same "no manifest, no fallback yet" behavior as today.
   - Update the module docstring's atomic-write paragraph (one-line tweak).
   - `tasks/lessons.md` skip (until end-of-loop).

2. **`feat(calib): Pattern O readers — validate manifest with back-compat WARN`**
   - Modify all 8 readers in rows #1-#8 to call `_validate_manifest_or_warn`.
   - This is the activation commit — once it lands, fresh writes produce a
     manifest and fresh reads validate it; legacy reads fall back with WARN.

3. **`test(calib): Pattern O coverage for 8 calibration sidecars`**
   - 8 × {roundtrip-with-manifest, torn-payload-detected, missing-manifest-WARN}
     = 24 new tests, plus 2 extra (warn-dedupe, block_hidden per-layer manifest).
   - Preserve every existing test as-is.

(Alternative — one commit per pair — would be 10 commits + 1 test commit. The
implementer may choose this if reviewer feedback explicitly asks for per-pair
commits, but the 3-commit shape is the default.)

---

## 9. Test plan

For each of the 8 writer/reader pairs, add the following four tests (mirror the
test names already in `test_stage3_originals_manifest.py` and
`test_stage5_teacher_logits_manifest.py` — keep the convention identical):

1. **`test_<signal>_manifest_roundtrip`** — write, then assert:
   - `<sidecar>.pt` exists and reloads byte-identical to the input payload.
   - `<sidecar>.pt.MANIFEST.json` exists.
   - The manifest's `schema_version` equals `SCHEMA_VERSIONS[<signal>]`.
   - The manifest's `extra.artifact` equals the signal name.
   - The manifest's `size_bytes` equals `<sidecar>.pt.stat().st_size`.

2. **`test_<signal>_torn_payload_detected_by_size`** — write, truncate the .pt
   to half size, attempt `load_<signal>`, assert RuntimeError raised with
   "manifest validation FAILED" + "Delete" + signal-specific filenames.

3. **`test_<signal>_missing_manifest_warn_fallback_loads`** — write payload
   only (skip the manifest), confirm `load_<signal>` returns the payload and
   emits the WARN at most once.

4. **`test_<signal>_missing_manifest_warn_deduped`** — after the warn fires on
   the first load, a second load on the same path emits NO additional WARN.
   Confirm `len([r for r in caplog.records if 'Pattern O back-compat' in r.message])`
   stays at 1 across two consecutive loads.

For row #8 (`block_hidden`), add ONE extra:

5. **`test_block_hidden_per_layer_manifests_independent`** — write layers 0..2,
   truncate layer 1's .pt, assert layers 0 and 2 still load and layer 1 fails
   loudly. Confirms per-layer torn-write isolation.

For row #7 (`stage2_profile_v3`), add ONE extra:

6. **`test_stage2_profile_manifest_with_cross_validation`** — confirm the
   manifest validation runs BEFORE the existing cross-validation
   (`expected_cov_storage_dtype`, `expected_n_layers`, etc.) so a torn .pt
   surfaces a manifest error not a cross-validation error. The order matters
   because the cross-validation messages currently say "delete the sidecar"
   without mentioning the manifest; a torn-write should produce the manifest-
   specific message.

**Total new tests**: 24 + 2 = 26 new tests across the 8 pairs.

**Preserved existing tests** (every existing test stays green):
* `test_sidecar_path_atomic_and_sharded` (262)
* `test_stage2_profile_roundtrip` (292) — now also exercises the manifest path
* `test_reap_scores_roundtrip` (336) — likewise
* ... (every roundtrip test from lines 336-661)
* `test_schema_version_mismatch_raises` (662) — unchanged behavior
* `test_atomic_write_crash_safety` (700) — extend with a manifest-aware variant
  in the new tests; the original still verifies the .tmp → final atomic dance
* `test_provider_pair_dispatch_first` (835) — unchanged
* The 4 F-H-7 legacy-path back-compat tests (946-1051) — unchanged

**Independent verification commands** (the implementer must run all three):

```bash
# Unit tests (this module only):
pytest max_quality/tests/test_cached_calibration_signals.py -v

# Adjacent atomic-io tests (must stay green — we don't touch this file but
# the new helpers depend on it):
pytest max_quality/tests/test_utils_atomic_io.py -v

# The two existing Pattern O integration tests must stay green (different
# artifacts, but same write_manifest_last / read_and_validate_manifest API):
pytest max_quality/tests/test_stage3_originals_manifest.py \
       max_quality/tests/test_stage5_teacher_logits_manifest.py -v
```

Plus a final full-suite smoke (`pytest max_quality/tests/ -x`) before merge to
catch any cross-cutting break (the calibration sidecar tests are read by half a
dozen plugin tests transitively).

---

## 10. Rollback plan

If a regression surfaces post-merge, the revert sequence is straightforward
because of the 3-commit chunking:

### Symptom A: writer crashes mid-calibration → calibration job fails to land sidecars

Most likely cause: `write_manifest_last` raised on a CPU-only / FUSE-mounted FS
where the helper's `_fsync_dir` errors. Pattern O's atomic_io already swallows
those at DEBUG level (lines 96-138 of atomic_io.py), so the failure would have
to come from somewhere else (e.g. `payload_path.stat().st_size` on a
non-POSIX FS).

**Revert sequence**:
1. `git revert <commit-1-feat-writers>` — also reverts commit 3 (tests) by
   `git revert <commit-3-tests>` first. Net: returns to today's
   "atomic-write only, no manifest emitted" posture.
2. Push.
3. Operators with in-flight calibration runs see no behavior change because:
   - their on-disk sidecars are pre-Pattern-O (back-compat-WARN path),
   - the writer revert removes the manifest emission,
   - the reader revert removes the manifest validation call,
   - the cache hit / live recompute decision is unchanged.

### Symptom B: reader crashes on a healthy artifact

Most likely cause: a logic bug in `_validate_manifest_or_warn` (e.g.
back-compat fallback misfires and rejects a fresh write).

**Revert sequence**:
1. `git revert <commit-2-feat-readers>` only.
2. Push.
3. Writers continue to emit manifests (commit 1 stays in), readers stop
   validating them — equivalent to "ahead-of-policy" state. Safe because the
   manifest is a CHECK, not a contract; ignoring it == today's behavior.
4. Open a follow-up to fix the reader bug.

### Symptom C: A reviewer / operator complains about disk-pressure from the
~tiny manifest files (worst case: 8 × n_layers × ~200 bytes ≈ 0.2 MB per
artifacts dir for the block_hidden shards). Not a credible scenario but listed
for completeness.

**Revert sequence**: `git revert <commit-1-feat-writers>` only. Readers tolerate
missing manifests by design (back-compat WARN).

### Pre-merge guard

Before merging, the implementer MUST verify on a real artifacts-dir snapshot
(planner suggests `/opt/output/calibration_smoke_sm9.0a/` or any small calibration
output the user has on the rental box) that:

1. Existing sidecars (no manifest) STILL LOAD with a single WARN.
2. After one `save_*` call, the sidecar AND its manifest are present.
3. After a follow-up `load_*` call, no WARN is emitted.
4. A `truncate -s $((size/2))` on the new sidecar makes the subsequent
   `load_*` raise the RuntimeError with the "delete + re-run" message.

This is a 4-step manual smoke; ~5 minutes. Strongly recommended before pushing
to a branch that the user will merge.

---

## 11. Risk hotspots — which pair is most likely to break

Ranked from highest to lowest implementer-attention-priority:

### 11.1 (HIGHEST) — Row #7: `stage2_profile_v3`

* **Why riskiest**: largest payload (~40 GB cov + reservoir on Qwen3.6-35B-A3B),
  most cross-validation (5 optional `expected_*` kwargs), most callers
  (Stage 2 `LayerMergePlugin` early-return depends on this). A bad manifest
  posture here breaks the single biggest cache lever in the pipeline (per
  `PLAN_PLUGIN_14_sidecar_audit.md` §3, Plugin #12 review).
* **Implementer prioritization**: write the per-pair test #6 (the cross-
  validation order test) FIRST as a guard. The reader's manifest validation
  MUST run BEFORE the existing cross-validation else operators see the wrong
  error message under torn-write.
* **Bonus risk**: `compute_sha256=False` is mandatory; computing SHA-256 on
  a 40 GB payload would add minutes per calibration run. Confirm.

### 11.2 (HIGH) — Row #8: `block_hidden`

* **Why risky**: PER-LAYER shard pattern means N manifests per artifacts dir
  (typically 40+ for Qwen3.6-35B-A3B). The pre-`unlink(missing_ok=True)` of
  the manifest must run per-shard, and the sidecar_path for the manifest must
  use the same per-shard layout
  (`sidecars/<stem>/block_hidden/layer_NNNN.pt.MANIFEST.json`).
* **Implementer prioritization**: write per-pair test #5 (per-layer torn-write
  isolation) FIRST.
* **Bonus risk**: the back-compat WARN dedupe key MUST be per-shard
  (`(payload_path, signal_name)` where `payload_path` includes the layer
  number) so legacy 40-layer artifacts don't dedupe to one global warning
  (which would obscure which layer is legacy). The helper signature in §3.3
  uses `payload_path` so this is already correct — but verify.

### 11.3 (MEDIUM) — Row #6: `output_reservoir`

* **Why risky**: large (~10-15 GB). `compute_sha256=False` is mandatory.
  Otherwise low risk — single payload, no cross-validation, straightforward
  reader.

### 11.4 (MEDIUM) — Row #1: `covariance`

* **Why risky**: ~40 GB at typical configs. Same SHA-256 policy as #6.
  Otherwise low risk.

### 11.5 (LOW) — Rows #2, #3, #4, #5

* `routing_stats`, `per_expert_max`, `reap_scores`, `router_logits_stats` are
  all small (~MB), single-payload, no cross-validation. Mechanical
  application of the writer/reader templates. Lowest risk.
* `routing_stats` has the wrinkle that its consumer is unwired today (per
  `cached_calibration_signals.py:395-400`), so a reader-side regression here
  has no production impact — but the cache provider DOES still call
  `load_routing_stats` and deposit it on ctx, so a reader crash would surface
  immediately as a Stage 1 STEP 4.6 failure.

### Cross-cutting risk

* **The `_already_warned_missing_manifest` set is process-local.** Calibration
  runs on rented GPUs that get killed and restarted; each fresh restart re-emits
  the WARN once. This is by design (matches `_already_warned_legacy_paths`) but
  worth documenting in the helper's docstring — the implementer should mention
  it explicitly so operators don't file noise as a regression.

---

## 12. Effort estimate

Best-case (no surprises, no rework loops):
* **Source diff**: ~50 LoC helpers + 8 × ~5 LoC writer edits + 8 × ~3 LoC reader
  edits + docstring tweak ≈ 120-150 LoC net. ~2 hours.
* **Test diff**: 26 new tests; each follows an existing template (the
  test_stage3_originals_manifest.py shape). ~3 hours.
* **Manual pre-merge smoke** (§10): ~30 min.
* **Review / fix loop**: per the user's review/fix protocol, expect ~1-2 fix
  iterations (nitpicks + 1 medium). ~2 hours.

**Total: ~8 hours / 1 working day.**

Worst-case (a fix-loop catches a real bug — e.g. per-shard manifest path bug
in row #8 surfaces only after the smoke against real artifacts):
* Add ~4 hours debug + retest.

**Worst-case total: ~12 hours / 1.5 working days.**

---

## 13. Riskiest pair — the implementer should prioritize tests for

**Row #7: `stage2_profile_v3`** is the highest-risk pair. Reasoning recap:

1. Largest payload (~40 GB) — torn writes the most likely to surface here.
2. Most cross-validation (5 `expected_*` kwargs) — manifest validation must
   run BEFORE these or operators get misleading error messages.
3. Highest blast radius — broken cache here defeats the single biggest cache
   lever in the pipeline (`PLAN_PLUGIN_14_sidecar_audit.md` §7, top-3 action #1
   line 479).
4. Largest test surface — the existing
   `test_stage2_profile_roundtrip(tmp_path)` is 44 lines (lines 292-335);
   adding manifest-aware variants requires careful attention to the
   gate_logit_profiles dict-of-list-of-tuples preservation contract.

Recommend the implementer:
* Write `test_stage2_profile_manifest_with_cross_validation` FIRST.
* Then write the writer side for row #7 against that test.
* Then the reader side.
* Then proceed to the easier 7 pairs in any order.

---

## 14. Deliverable summary

* **Plan doc path**: `tasks/PLAN_S1_PATTERN_O_CALIB_SIDECARS.md` (this file).
* **Estimated effort**: ~8 hours best-case / ~12 hours worst-case (1.0-1.5 working
  days for a single implementer).
* **Riskiest pair to prioritize**: **`stage2_profile_v3` (row #7)** — largest
  payload, most cross-validation, largest blast radius.
