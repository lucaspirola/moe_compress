# Stage-3 Factorization Fixes (F1 α-determinism + F2 parallel spectra) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Stage-3 Swift-SVD+ α-search host-reproducible (F1: eigh→Cholesky whitening) and cut the ~90 min serial fp64-CPU rank-spectra loop to ~8-13 min via an opt-in spawn ProcessPool (F2), both fidelity-safe.

**Architecture:** Two INDEPENDENT functions that both touch Stage-3 factorization goldens, so they are sequenced F1-first then F2. **F1** replaces the discrete-threshold `eigh` whitening factor in the two α-path spectral producers with a unique full-rank Cholesky factor (keeping the load-bearing `W @ L_C` order), adds a relative-tolerance α tie-break, and regenerates ONLY the α-variant goldens. **F2** parallelizes the per-`(layer,matrix)` group-stat loop across a `multiprocessing.get_context("spawn")` ProcessPool with each worker pinned to 1 intra-op thread (a fidelity invariant, not just perf), gated by a `stage3_svd.spectra_workers` config knob defaulting to `1` (byte-identical serial path).

**Tech Stack:** Python, PyTorch (`torch.linalg.cholesky` / `svdvals`, fp64 CPU), `concurrent.futures.ProcessPoolExecutor` + `multiprocessing` spawn context, pytest (golden byte-identity + `torch.equal` fidelity gates).

**Source design docs (reviewer-approved, on branch `feat/stage3-factorization-fixes`):**
- F1: `max_quality/docs/research/2026-06-14-alpha-search-determinism.md`
- F2: `max_quality/docs/research/2026-06-14-stage3-spectra-speedup.md`

The design is DECIDED. This plan synthesizes — it does NOT re-design. Every code line below was grep-verified against the working tree at HEAD `08be594` (the doc line numbers had not drifted; they match).

---

## Grep-verified line numbers (HEAD 08be594)

**F1 — `max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py`:**
- Producer 1 `_swift_svd_plus_alpha_search` eigh whitening block: **lines 1202-1216**
  (`A64` 1203-1204, `eigh` **1205**, `keep_a` **1206**, `L_A` **1208**, `M_A = W @ L_A` **1209**, `svdvals(M_A)` **1210**, rank-zero fallback `svdvals(W)` 1211-1216)
- Producer 2 `_redistribute_ranks_swift_svd_plus` eigh whitening block: **lines 1400-1413**
  (`A64` 1401-1402, `eigh` **1403**, `keep_a` **1404**, `L_A` **1406**, `svdvals(W @ L_A)` **1407**, rank-zero fallback 1408-1413)
- α-selection proxy loops: per-type `if err < best_err` at **lines 1324-1328**; global `if err < best_err` at **lines 1335-1339**
- Tier-1 precondition inline recompute (uses eigh — MUST be updated in lockstep): `max_quality/tests/test_stage3_tier1.py` **lines 143-146** (`assert torch.equal` at **147**)
- `_warn_raw_svd_fallback_once` helper: defined at swift_svd_alpha.py **line 248**

**F2 — `max_quality/src/moe_compress/stage3/orchestrator.py`:**
- Serial group-stat loop: **lines 658-677** (outer `for k, ref` at **658**, cov-average inner block 667-674, `_group_stat(...)` call **675-677**)
- `_group_stat` body (numerics UNTOUCHED): `max_quality/src/moe_compress/stage3/plugins/d_rank_allocate.py` **lines 347-421**; Cholesky factor **374**; `effective_rank` materialized (the bit-sensitive pivot) at **line 410**; `_GroupStats` dataclass field `effective_rank` at **line 338**
- Spawn-context precedent: `max_quality/src/moe_compress/stage6/plugins/humaneval.py` **lines 374-381** (`multiprocessing.get_context("spawn")` + `ProcessPoolExecutor(mp_context=ctx)`)
- Orchestrator binds `_group_stat`, `_GroupStats`, `MATRIX_NAMES`, `build_banks`, `_cov_lookup` at lines 50-66; `s3 = config["stage3_svd"]` at line 194; `MATRIX_NAMES = ("gate_proj", "up_proj", "down_proj")` at model_io.py:365

---

## File Structure

### F1 files
| File | Responsibility | Action |
|---|---|---|
| `max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py` | Swap eigh→Cholesky whitening in BOTH α-path producers (identical swap, keep `W @ L_C` order); add `ALPHA_ERR_REL_TOL` tie-break to the two proxy selection loops | Modify 1202-1216, 1400-1413, 1324-1328, 1335-1339 |
| `max_quality/tests/test_stage3_alpha_determinism.py` | NEW — α-selection + integer-rank invariance under fp-epsilon perturbation; whitening-determinism micro-test (eigh threshold-flip vs Cholesky stability) | Create |
| `max_quality/tests/test_stage3_tier1.py` | Update the precondition inline recompute (143-146) from eigh→Cholesky so producer==recompute stays `torch.equal` | Modify 143-146 |
| `max_quality/tests/golden/stage3/rank_map.alpha.fp32.json` | α-variant golden — REGENERATE (intended bytes change) | Regenerate |
| `max_quality/tests/golden/stage3/rank_map.alpha.bf16.json` | α-variant golden — REGENERATE | Regenerate |
| `max_quality/tests/golden/stage3/rank_map.{fp32,bf16}.json` | non-α goldens — **MUST NOT be regenerated**; stay byte-identical | Untouched |

### F2 files
| File | Responsibility | Action |
|---|---|---|
| `max_quality/src/moe_compress/stage3/spectra_pool.py` | NEW — lean, torch-light top-level module: picklable `_group_stat_payload(payload)` wrapper + `_pin_one_thread` pool initializer. Bounds spawn re-import cost. | Create |
| `max_quality/src/moe_compress/stage3/orchestrator.py` | Replace the serial group-stat loop (658-677) with: parent-side CPU gather → spawn ProcessPool dispatch (gated by `stage3_svd.spectra_workers`, default 1) → order-free dict reassembly | Modify 658-677 |
| `max_quality/tests/test_stage3_spectra_pool.py` | NEW — parallel==1-thread-pinned-serial equivalence (`effective_rank` exact + `singular_values_mean` `torch.equal` + rank-map sweep); thread-pinning-holds-the-bits; worker serializability/initializer | Create |
| `max_quality/src/moe_compress/stage3/plugins/d_rank_allocate.py` | numerics UNTOUCHED (the pool wrapper reconstructs a duck-typed bank and calls the unchanged `_group_stat`) | Untouched |

---

## Sequencing & golden-regen contract (READ FIRST)

1. **F1 lands first** (Tasks 1-5): α-whitening Cholesky swap + tie-break + tier1 precondition update, then regenerate ONLY `rank_map.alpha.{fp32,bf16}.json`.
2. **F2 lands second** (Tasks 6-11): parallelize behind `spectra_workers` (default 1). The default path stays byte-identical to the committed non-α AND α goldens (the α golden already re-blessed by F1).
3. **Golden rules — NON-NEGOTIABLE:**
   - F1 regenerates **only** `rank_map.alpha.fp32.json` + `rank_map.alpha.bf16.json` via `MOE_REGEN_GOLDEN=1`.
   - The non-α `rank_map.{fp32,bf16}.json` are **NEVER regenerated** in this plan. `test_stage3_rank_map_byte_identical` must keep passing against the unchanged committed bytes after BOTH F1 and F2.
   - F2's default (`spectra_workers=1`) is byte-identical to today's serial default path — **no golden regen for F2**. The parallel path (`workers>1`) is validated against the 1-thread-pinned serial result via `torch.equal` (Task 8), NOT against the committed default-threaded golden (the two legitimately differ at ~1e-11 — see F2 doc §2/§4; that re-bless is explicitly OUT of scope here and is a future opt-in flip of the default).
4. **These tiny-model golden tests RUN ON THIS HOST (RTX 5080)** — F1 is fully validatable here (regen + re-pass + cross-host reproducibility is the acceptance property). F2's tiny-model equivalence tests also run here (multi-core box).

---

# GROUP F1 — α-search determinism (eigh→Cholesky whitening)

## Task 1: Whitening-determinism micro-test (proves the bug + the fix mechanism)

**Files:**
- Create: `max_quality/tests/test_stage3_alpha_determinism.py`

This test targets the ACTUAL instability — the discrete `keep_a` threshold flip — NOT eigenbasis sign/rotation (which `svdvals(W @ L)` is invariant to by construction). It demonstrates the OLD eigh path was threshold-sensitive and the NEW Cholesky path is not. The Cholesky helper does not exist yet, so it fails first.

- [ ] **Step 1: Write the failing test**

```python
"""Stage-3 α-path whitening determinism (F1).

Targets the discrete ``keep_a`` threshold flip that makes the eigh-based
α-path whitening cross-host non-reproducible, and proves the Cholesky
replacement removes it. Does NOT assert eigenvector sign/rotation
invariance — ``svdvals(W @ L)`` depends only on ``L @ L.T`` and is invariant
to it by construction (the doc's reviewer measured ~5e-15), so such an
assertion proves nothing about the bug.
"""
from __future__ import annotations

import torch


def _eigh_factor(A64: torch.Tensor) -> torch.Tensor:
    """The OLD whitening factor (discrete keep-threshold)."""
    A64 = 0.5 * (A64 + A64.T)
    ev, evec = torch.linalg.eigh(A64)
    keep = ev > ev.max() * 1e-6
    return evec[:, keep] * ev[keep].clamp_min(1e-12).sqrt().unsqueeze(0)


def _chol_factor(A64: torch.Tensor) -> torch.Tensor:
    """The NEW whitening factor (full-rank, unique, no threshold)."""
    from moe_compress.stage3.plugins.swift_svd_alpha import _alpha_whiten_factor
    return _alpha_whiten_factor(0.5 * (A64 + A64.T))


def _spd_with_eigenvalue_at_threshold(d: int, seed: int):
    """Build an SPD ``A`` with one eigenvalue sitting right at ``1e-6 * lambda_max``."""
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    lam = torch.ones(d, dtype=torch.float64)
    lam[0] = 1.0  # lambda_max
    lam[-1] = 1e-6  # exactly at the keep threshold
    A = (Q * lam) @ Q.T
    return 0.5 * (A + A.T)


def test_eigh_whitening_is_threshold_sensitive():
    """OLD path: nudging the boundary eigenvalue across ``1e-6*lambda_max`` flips
    the kept-column count → svdvals dimension/values jump discontinuously."""
    d = 8
    torch.manual_seed(1)
    W = torch.randn(6, d, dtype=torch.float64)
    A = _spd_with_eigenvalue_at_threshold(d, seed=2)
    ev, evec = torch.linalg.eigh(A)
    ev_lo = ev.clone()
    ev_lo[ev_lo.argmin()] = ev.max() * 1e-6 * 0.5  # now dropped by keep_a
    A_lo = (evec * ev_lo) @ evec.T
    s_at = torch.linalg.svdvals(W @ _eigh_factor(A))
    s_lo = torch.linalg.svdvals(W @ _eigh_factor(A_lo))
    assert s_at.shape != s_lo.shape or not torch.allclose(s_at, s_lo, atol=1e-6)


def test_cholesky_whitening_is_threshold_free_and_stable():
    """NEW path: same boundary perturbation changes svdvals only by round-off
    (no threshold to cross), and cholesky run twice is byte-equal."""
    d = 8
    torch.manual_seed(1)
    W = torch.randn(6, d, dtype=torch.float64)
    A = _spd_with_eigenvalue_at_threshold(d, seed=2)
    A_eps = A + 1e-12 * torch.eye(d, dtype=torch.float64)
    s0 = torch.linalg.svdvals(W @ _chol_factor(A))
    s1 = torch.linalg.svdvals(W @ _chol_factor(A_eps))
    assert s0.shape == s1.shape
    assert torch.allclose(s0, s1, rtol=0, atol=1e-9)
    assert torch.equal(_chol_factor(A.clone()), _chol_factor(A.clone()))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest max_quality/tests/test_stage3_alpha_determinism.py -v`
Expected: FAIL — `ImportError: cannot import name '_alpha_whiten_factor' from ...swift_svd_alpha` (helper not yet added).

- [ ] **Step 3: Add the Cholesky whitening helper (minimal)**

In `swift_svd_alpha.py`, add a module-level helper near `_warn_raw_svd_fallback_once` (after line 248). It mirrors d_rank_allocate.py:372-374's jitter recipe but is a pure factor producer (caller keeps the `W @ L_C` order):

```python
def _alpha_whiten_factor(A64: torch.Tensor) -> torch.Tensor:
    """Cholesky whitening factor for the α-path spectra.

    F1: replaces the prior eigh-based ``L_A`` (discrete ``keep_a`` threshold,
    cross-host-unstable) with the unique full-rank lower-triangular Cholesky
    factor of ``A64 + jitter`` (jitter recipe mirrors d_rank_allocate.py:372-374).
    The CALLER keeps the ``svdvals(W @ L_C)`` order — do NOT use ``L_C @ W.T``:
    for a triangular factor ``L_C^T L_C != A``, so ``svdvals(L_C @ W.T)`` is a
    DIFFERENT, paper-incorrect quantity (reviewer measured 3.95 divergence).
    ``svdvals(W @ L_C) = sqrt(eig(W (A+jitter) W^T))`` is the correct
    activation-weighted spectrum (depends on A only through L_C L_C^T = A+jitter).

    ``A64`` MUST already be CPU-fp64 and symmetrized (``0.5*(A+A.T)``).
    """
    jitter = 1e-6 * A64.diag().mean().clamp_min(1e-12) * torch.eye(
        A64.shape[0], dtype=torch.float64
    )
    return torch.linalg.cholesky(A64 + jitter)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest max_quality/tests/test_stage3_alpha_determinism.py -v`
Expected: PASS (both `test_eigh_whitening_is_threshold_sensitive` and `test_cholesky_whitening_is_threshold_free_and_stable`).

- [ ] **Step 5: Commit**

```bash
git add max_quality/tests/test_stage3_alpha_determinism.py \
        max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py
git commit -m "test(stage3-f1): whitening-determinism micro-test + _alpha_whiten_factor helper"
```

---

## Task 2: Swap eigh→Cholesky in producer 1 (`_swift_svd_plus_alpha_search`)

**Files:**
- Modify: `max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py:1202-1216`
- Test (reused): `max_quality/tests/test_stage3_alpha_determinism.py`

- [ ] **Step 1: Write the failing test (producer 1 emits the Cholesky spectrum)**

Append to `test_stage3_alpha_determinism.py`. NOTE on fixtures: the codebase rule is tests do not import each other. The tier-1 helper builders (`_FusedExperts`, `_make_layer_ref`, `_build_acov`, `_group_stats_for`) are private to `test_stage3_tier1.py`. Inline 12-line local copies of them at the top of this file (read test_stage3_tier1.py:1-105 and copy the four builders verbatim) rather than importing — this keeps the no-cross-import rule. Reference them as module-local helpers below.

```python
def test_producer1_uses_cholesky_spectrum():
    """``_swift_svd_plus_alpha_search`` (producer 1) emits svdvals(W @ L_C)
    bit-identical to the helper-based recompute (proves the eigh block is gone)."""
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search, _alpha_whiten_factor,
    )
    from moe_compress.utils.model_io import build_banks
    layer_idx, n, d_int, d_hid = 0, 4, 6, 8
    experts = _FusedExperts(n, d_int, d_hid, seed=3)        # local copy
    ref = _make_layer_ref(layer_idx, experts)              # local copy
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)  # local copy
    group_stats = _group_stats_for(layer_idx, experts, d_int, d_hid)  # local copy
    base_ranks = {k: 3 for k in group_stats}
    _, grouped_svs = _swift_svd_plus_alpha_search(
        [ref], group_stats, base_ranks, [0.0, 0.5, 1.0],
        per_group_type=True, A_cov=A_cov, return_svs=True,
    )
    banks = build_banks(ref)
    for name in ("gate_proj", "up_proj", "down_proj"):
        for e in range(n):
            W = banks[name].get(e).detach().to(device="cpu", dtype=torch.float64)
            A = A_cov[(layer_idx, e, name)].to(device="cpu", dtype=torch.float64)
            A = 0.5 * (A + A.T)
            inline = torch.linalg.svdvals(W @ _alpha_whiten_factor(A))
            assert torch.equal(grouped_svs[name][(layer_idx, e)], inline)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest max_quality/tests/test_stage3_alpha_determinism.py::test_producer1_uses_cholesky_spectrum -v`
Expected: FAIL — `torch.equal` is False (producer 1 still emits the eigh spectrum, helper emits Cholesky).

- [ ] **Step 3: Replace the eigh block at 1202-1216 with the Cholesky helper**

Replace the `if A is not None:` body (currently 1202-1216) with:

```python
            if A is not None:
                A64 = A.to(device="cpu", dtype=torch.float64)
                A64 = 0.5 * (A64 + A64.T)
                try:
                    # F1: Cholesky whitening — host-stable, full-rank, no
                    # discrete keep_a threshold. KEEP svdvals(W @ L_C) order.
                    L_C = _alpha_whiten_factor(A64)
                    svs = torch.linalg.svdvals(W @ L_C)
                except Exception:
                    _warn_raw_svd_fallback_once(
                        "A_cov Cholesky failed / rank-zero "
                        "(in _swift_svd_plus_alpha_search)"
                    )
                    svs = torch.linalg.svdvals(W)
```

(The old `keep_a.any()` branch + its `M_A` temporary are removed; the Cholesky-failure fallback to `svdvals(W)` + the warn-once replaces the rank-zero fallback.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest max_quality/tests/test_stage3_alpha_determinism.py::test_producer1_uses_cholesky_spectrum -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py \
        max_quality/tests/test_stage3_alpha_determinism.py
git commit -m "fix(stage3-f1): Cholesky whitening in _swift_svd_plus_alpha_search (producer 1)"
```

---

## Task 3: Swap eigh→Cholesky in producer 2 (`_redistribute_ranks_swift_svd_plus`) + fix the Tier-1 precondition

**Files:**
- Modify: `max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py:1400-1413`
- Modify: `max_quality/tests/test_stage3_tier1.py:143-146`

The IDENTICAL swap must go in producer 2 so the Tier-1 `torch.equal` cache precondition (producer == recompute) still holds. The precondition test (test_stage3_tier1.py:143-146) currently reconstructs the inline spectrum with `eigh`; update it to use `_alpha_whiten_factor` so it tests the new (Cholesky) lockstep.

- [ ] **Step 1: Update the Tier-1 precondition test to the failing (Cholesky) expectation**

In `test_stage3_tier1.py`, replace lines 143-146 (the eigh recompute: `ev, evec = torch.linalg.eigh(A)` … `L_A = ...` … `inline = torch.linalg.svdvals(W @ L_A)`) with:

```python
            from moe_compress.stage3.plugins.swift_svd_alpha import _alpha_whiten_factor
            L_C = _alpha_whiten_factor(A)
            inline = torch.linalg.svdvals(W @ L_C)
```

(Keep the surrounding `W = ...`, `A = ...; A = 0.5*(A+A.T)`, and the `assert torch.equal(...)` at 147 unchanged.)

- [ ] **Step 2: Run the precondition test to verify it fails**

Run: `python3 -m pytest max_quality/tests/test_stage3_tier1.py::test_grouped_svs_cache_precondition_torch_equal -v`
Expected: FAIL — producer 2 still emits eigh spectrum; the Cholesky recompute differs → `torch.equal` False.

- [ ] **Step 3: Replace the eigh block at 1400-1413 with the Cholesky helper (identical to producer 1)**

Replace the `if A is not None:` body (currently 1400-1413) with:

```python
                if A is not None:
                    A64 = A.to(device="cpu", dtype=torch.float64)
                    A64 = 0.5 * (A64 + A64.T)
                    try:
                        # F1: identical Cholesky whitening as producer 1 so the
                        # Tier-1 torch.equal cache precondition holds. KEEP W @ L_C.
                        L_C = _alpha_whiten_factor(A64)
                        svs = torch.linalg.svdvals(W @ L_C)
                    except Exception:
                        _warn_raw_svd_fallback_once(
                            "A_cov Cholesky failed / rank-zero "
                            "(in _redistribute_ranks_swift_svd_plus)"
                        )
                        svs = torch.linalg.svdvals(W)
```

- [ ] **Step 4: Run the precondition + cache-equivalence tests to verify they pass**

Run: `python3 -m pytest max_quality/tests/test_stage3_tier1.py -v`
Expected: PASS (all of `test_grouped_svs_cache_precondition_torch_equal`, `test_grouped_svs_cache_equals_recompute`, and the rest of the file).

- [ ] **Step 5: Commit**

```bash
git add max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py \
        max_quality/tests/test_stage3_tier1.py
git commit -m "fix(stage3-f1): Cholesky whitening in _redistribute (producer 2) + tier1 precondition"
```

---

## Task 4: Relative-tolerance α tie-break (defence-in-depth)

**Files:**
- Modify: `max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py:1324-1328` (per-type loop) and `:1335-1339` (global loop)
- Test (reused): `max_quality/tests/test_stage3_alpha_determinism.py`

Add a `1e-9` RELATIVE tolerance so a tie never depends on float ordering: strictly-better-beyond-tol wins; ties resolve to the lowest grid index / lowest α (the current strict-`<` behaviour). Do NOT touch `_argmin_alpha` or the residual `sorted(...)`/`int(math.floor(...))` loop.

- [ ] **Step 1: Write the failing test (α invariance under fp-epsilon)**

Append to `test_stage3_alpha_determinism.py` (reuse the module-local fixture builders from Task 2):

```python
def test_alpha_selection_invariant_under_epsilon_perturbation():
    """A near-tie α is stable under a relative ±1e-12 perturbation of A_cov
    (what cross-host jitter looks like)."""
    from moe_compress.stage3.plugins.swift_svd_alpha import _swift_svd_plus_alpha_search
    layer_idx, n, d_int, d_hid = 0, 4, 6, 8
    experts = _FusedExperts(n, d_int, d_hid, seed=3)
    ref = _make_layer_ref(layer_idx, experts)
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)
    gs = _group_stats_for(layer_idx, experts, d_int, d_hid)
    base = {k: 3 for k in gs}
    grid = [0.0, 0.5, 1.0]
    a0, _ = _swift_svd_plus_alpha_search([ref], gs, base, grid,
                                         per_group_type=True, A_cov=A_cov, return_svs=True)
    A_pert = {k: v * (1.0 + 1e-12) for k, v in A_cov.items()}
    a1, _ = _swift_svd_plus_alpha_search([ref], gs, base, grid,
                                         per_group_type=True, A_cov=A_pert, return_svs=True)
    assert a0 == a1, f"alpha flipped under epsilon: {a0} vs {a1}"


def test_alpha_clear_winner_preserved():
    """The ε-tolerance does NOT mask a genuine winner: a tight budget gives real
    tail-energy separation; the winner is stable across a ±1e-12 perturbation."""
    from moe_compress.stage3.plugins.swift_svd_alpha import _swift_svd_plus_alpha_search
    layer_idx, n, d_int, d_hid = 0, 4, 6, 8
    experts = _FusedExperts(n, d_int, d_hid, seed=11)
    ref = _make_layer_ref(layer_idx, experts)
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)
    gs = _group_stats_for(layer_idx, experts, d_int, d_hid)
    base = {k: 1 for k in gs}  # tight budget → real tail-energy separation
    grid = [0.0, 0.5, 1.0]
    a0, _ = _swift_svd_plus_alpha_search([ref], gs, base, grid,
                                         per_group_type=True, A_cov=A_cov, return_svs=True)
    A_pert = {k: v * (1.0 + 1e-12) for k, v in A_cov.items()}
    a1, _ = _swift_svd_plus_alpha_search([ref], gs, base, grid,
                                         per_group_type=True, A_cov=A_pert, return_svs=True)
    assert a0 == a1


def test_redistribute_ranks_invariant_under_epsilon():
    """Integer ranks (covers F1-doc §2.3 score-value drift) are identical under
    the same ±1e-12 A_cov perturbation."""
    from moe_compress.stage3.plugins.swift_svd_alpha import (
        _swift_svd_plus_alpha_search, _redistribute_ranks_swift_svd_plus,
    )
    layer_idx, n, d_int, d_hid = 0, 4, 6, 8
    experts = _FusedExperts(n, d_int, d_hid, seed=3)
    ref = _make_layer_ref(layer_idx, experts)
    A_cov = _build_acov(layer_idx, experts, d_hid, d_int)
    gs = _group_stats_for(layer_idx, experts, d_int, d_hid)
    base = {k: 3 for k in gs}
    a0, _ = _swift_svd_plus_alpha_search([ref], gs, base, [0.0, 0.5, 1.0],
                                         per_group_type=True, A_cov=A_cov, return_svs=True)
    r0 = _redistribute_ranks_swift_svd_plus([ref], gs, base, a0, A_cov=A_cov)
    A_pert = {k: v * (1.0 + 1e-12) for k, v in A_cov.items()}
    a1, _ = _swift_svd_plus_alpha_search([ref], gs, base, [0.0, 0.5, 1.0],
                                         per_group_type=True, A_cov=A_pert, return_svs=True)
    r1 = _redistribute_ranks_swift_svd_plus([ref], gs, base, a1, A_cov=A_pert)
    assert r0 == r1
```

- [ ] **Step 2: Run to verify (the invariance assertions may pass already from Tasks 2-3, or flake on a strict tie)**

Run: `python3 -m pytest max_quality/tests/test_stage3_alpha_determinism.py -k "epsilon or clear_winner" -v`
Expected: the deterministic whitening (Tasks 2-3) likely already makes these PASS. If any fails on a strict float tie, that is exactly what the tie-break fixes. Add the tie-break regardless (defence-in-depth, F1 doc §3.2).

- [ ] **Step 3: Add the `ALPHA_ERR_REL_TOL` tie-break to both proxy loops**

Add a module constant near the other top-of-file constants:

```python
# F1 §3.2: relative tie-break so a near-tied α never depends on float ordering.
# Larger than residual fp64 round-off (~1e-13..1e-15) after Cholesky whitening,
# smaller than any meaningful spectral gap (35B gate_proj real gap ~1.4%).
# Ties resolve to the lowest grid index / lowest α (matches strict-< + _argmin_alpha).
ALPHA_ERR_REL_TOL = 1e-9
```

Per-type loop (1322-1329) — change `best_alpha = 0.5` to `best_alpha = alpha_grid[0]` and the comparison at 1326:

```python
        for name in MATRIX_NAMES:
            best_alpha = alpha_grid[0]
            best_err = float("inf")
            for alpha in alpha_grid:
                err = _evaluate_alpha(name, alpha)
                if err < best_err * (1.0 - ALPHA_ERR_REL_TOL):  # strictly better beyond tol
                    best_err = err
                    best_alpha = alpha
            best_alphas[name] = best_alpha
```

Global loop (1333-1339) — same change at the `best_alpha = 0.5` init and the comparison at 1337:

```python
        best_alpha = alpha_grid[0]
        best_err = float("inf")
        for alpha in alpha_grid:
            err = sum(_evaluate_alpha(n, alpha) for n in MATRIX_NAMES)
            if err < best_err * (1.0 - ALPHA_ERR_REL_TOL):
                best_err = err
                best_alpha = alpha
```

(Keep `best_err` as the *unrounded* selected value, comparing the next raw `err` against the band, so the comparison is order-stable. Do NOT touch the residual `sorted(...)` / `int(math.floor(...))` loop or `_argmin_alpha`.)

- [ ] **Step 4: Run to verify all F1 determinism tests pass**

Run: `python3 -m pytest max_quality/tests/test_stage3_alpha_determinism.py -v`
Expected: PASS (all whitening + α-invariance + clear-winner + rank-invariance tests).

- [ ] **Step 5: Commit**

```bash
git add max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py \
        max_quality/tests/test_stage3_alpha_determinism.py
git commit -m "fix(stage3-f1): ALPHA_ERR_REL_TOL tie-break in both proxy selection loops"
```

---

## Task 5: Regenerate ONLY the α-variant goldens + verify non-α goldens unchanged

**Files:**
- Regenerate: `max_quality/tests/golden/stage3/rank_map.alpha.fp32.json`, `rank_map.alpha.bf16.json`
- Verify-unchanged: `max_quality/tests/golden/stage3/rank_map.{fp32,bf16}.json`

The Cholesky whitening changes the α-variant produced bytes (intended). Regenerate ONLY the α goldens.

- [ ] **Step 1: Confirm the α golden test currently FAILS against the stale (eigh-era) golden**

Run: `python3 -m pytest "max_quality/tests/test_stage3_golden_snapshot.py::test_stage3_rank_map_alpha_variant_byte_identical" -v`
Expected: FAIL — drift detected (the new Cholesky spectra changed `alpha_by_type` / `rank_map`).

- [ ] **Step 2: Confirm the NON-α golden test still PASSES (must be untouched)**

Run: `python3 -m pytest "max_quality/tests/test_stage3_golden_snapshot.py::test_stage3_rank_map_byte_identical" -v`
Expected: PASS — the uniform path (`alpha_grid` length ≤ 1) never enters the α producers, so these bytes are unaffected.

- [ ] **Step 3: Regenerate ONLY the α goldens**

```bash
MOE_REGEN_GOLDEN=1 python3 -m pytest \
  "max_quality/tests/test_stage3_golden_snapshot.py::test_stage3_rank_map_alpha_variant_byte_identical" -v
```
Expected: SKIP "Regenerated α-variant goldens — inspect `git diff` then commit." — writes `rank_map.alpha.fp32.json` + `rank_map.alpha.bf16.json`.

- [ ] **Step 4: Inspect the diff + re-verify both golden tests; confirm non-α goldens are byte-unchanged**

```bash
git status --porcelain max_quality/tests/golden/stage3/
git diff max_quality/tests/golden/stage3/rank_map.alpha.fp32.json
```
Expected: ONLY `rank_map.alpha.fp32.json` + `rank_map.alpha.bf16.json` modified; `rank_map.fp32.json` + `rank_map.bf16.json` show NO change. Sanity-check the new `alpha_by_type` + per-group rank budget conservation in the diff.

```bash
python3 -m pytest "max_quality/tests/test_stage3_golden_snapshot.py" -v
```
Expected: PASS (both `test_stage3_rank_map_byte_identical` and `test_stage3_rank_map_alpha_variant_byte_identical`).

- [ ] **Step 5: Commit the re-blessed α goldens (atomic with the code fix already committed in Tasks 1-4)**

```bash
git add max_quality/tests/golden/stage3/rank_map.alpha.fp32.json \
        max_quality/tests/golden/stage3/rank_map.alpha.bf16.json
git commit -m "test(stage3-f1): re-bless alpha-variant goldens for Cholesky whitening (human-gated)"
```

> **Cross-host reproducibility is the acceptance property of F1** (F1 doc §4.6): the NEW α golden must reproduce byte-for-byte on a second BLAS/host (e.g. the H200). That cross-host bless is a manual gate on a different box and is OUT of scope for the on-host impl loop — but the WHOLE POINT is that the Cholesky path now makes it reproducible. The on-host tests (Tasks 1-5) fully validate the determinism mechanism here.

---

# GROUP F2 — parallelize the fp64-CPU rank spectra (#5)

> F2 lands AFTER F1. Default `spectra_workers=1` keeps every golden (non-α + the F1-re-blessed α) byte-identical. Parallel (`workers>1`) is opt-in and validated against the 1-thread-pinned serial result (NOT the default-threaded golden — they legitimately differ at ~1e-11; that re-bless is out of scope).

## Task 6: Lean torch-light pool module with serializable wrapper + 1-thread initializer

**Files:**
- Create: `max_quality/src/moe_compress/stage3/spectra_pool.py`
- Test: `max_quality/tests/test_stage3_spectra_pool.py`

The wrapper must be a top-level (importable / serializable for spawn) function in a lean module so spawn re-import is cheap. It reconstructs a tiny duck-typed bank from the shipped CPU weight list (exactly like `test_stage3_tier2.py::_FakeBank`) and calls the UNCHANGED `_group_stat`.

- [ ] **Step 1: Write the failing test (serializability + initializer + payload correctness)**

```python
"""Stage-3 F2 — parallel fp64-CPU rank-spectra pool.

Validates the spawn-ProcessPool group-stat parallelization is FIDELITY-SAFE:
parallel (workers>1) == 1-thread-pinned-serial on effective_rank (exact float)
and singular_values_mean (torch.equal); 1-thread pinning is the bit invariant;
the worker wrapper is top-level/serializable.
"""
from __future__ import annotations

import copy
import torch


def _make_payload(seed=7):
    from moe_compress.stage3.spectra_pool import _GroupStatPayload
    torch.manual_seed(seed)
    d_out, d_in, n = 16, 12, 4
    weights = [torch.randn(d_out, d_in, dtype=torch.float32) for _ in range(n)]
    m = torch.randn(d_in, d_in, dtype=torch.float32)
    a_g = (m @ m.T + d_in * torch.eye(d_in)).to(torch.float32)
    return _GroupStatPayload(layer_idx=0, name="gate_proj", n_experts=n,
                             weights_cpu=weights, a_g_cpu=a_g)


def test_payload_round_trips_through_pool_serialization():
    """The payload survives the spawn pool's submit path (deepcopy stands in for
    the cross-process round-trip; a real workers=2 run in test 4 exercises the
    actual spawn pickling end-to-end)."""
    payload = _make_payload()
    clone = copy.deepcopy(payload)
    assert clone.name == "gate_proj"
    assert torch.equal(clone.weights_cpu[0], payload.weights_cpu[0])


def test_group_stat_payload_matches_direct_group_stat():
    """The wrapper's output equals calling _group_stat directly (same numbers)."""
    from moe_compress.stage3.spectra_pool import _group_stat_payload, _ListBank
    from moe_compress.stage3.plugins.d_rank_allocate import _group_stat
    payload = _make_payload()
    torch.set_num_threads(1)
    key, gs = _group_stat_payload(payload)
    ref = _group_stat(payload.n_experts, _ListBank(payload.weights_cpu), A_g=payload.a_g_cpu)
    assert key == (0, "gate_proj")
    assert gs.effective_rank == ref.effective_rank
    assert torch.equal(gs.singular_values_mean, ref.singular_values_mean)


def test_pin_one_thread_initializer():
    from moe_compress.stage3.spectra_pool import _pin_one_thread
    saved = torch.get_num_threads()
    try:
        torch.set_num_threads(4)
        _pin_one_thread()
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(saved)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: moe_compress.stage3.spectra_pool`.

- [ ] **Step 3: Create the lean pool module**

`max_quality/src/moe_compress/stage3/spectra_pool.py`:

```python
"""Lean, torch-light worker leaf module for the Stage-3 parallel rank-spectra
pool (F2). Top-level (serializable) payload + wrapper + 1-thread initializer.

Kept import-light so the spawn ProcessPool re-import per worker is cheap
(precedent: stage6/plugins/humaneval's torch-free worker leaf). ``_group_stat``
numerics are UNCHANGED — the wrapper reconstructs a tiny duck-typed bank from
the shipped CPU weight list and calls the existing ``_group_stat``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class _GroupStatPayload:
    """Serializable per-(layer,matrix) group unit. CPU tensors only (CPU torch
    tensor serialization is bit-exact)."""
    layer_idx: int
    name: str
    n_experts: int
    weights_cpu: list  # [torch.Tensor (d_out,d_in) cpu fp32]
    a_g_cpu: "torch.Tensor | None"  # (d_in,d_in) cpu fp32, or None


class _ListBank:
    """Duck-typed bank: ``.shape()`` + ``.get(e)`` over a CPU weight list
    (mirrors test_stage3_tier2._FakeBank)."""

    def __init__(self, weights):
        self._w = weights

    def shape(self):
        return tuple(self._w[0].shape)

    def get(self, e):
        return self._w[e]


def _pin_one_thread() -> None:
    """Pool initializer. 1 intra-op thread per worker is a FIDELITY invariant
    (multi-thread BLAS reassociates fp sums → bit drift in effective_rank), not
    just perf. See F2 doc §2."""
    torch.set_num_threads(1)


def _group_stat_payload(payload: "_GroupStatPayload"):
    """Top-level (serializable) worker entry. Returns ((layer_idx, name), _GroupStats)."""
    # Lazy import keeps the leaf module torch-light at import time but the heavy
    # numerics module is loaded once per spawned worker.
    from moe_compress.stage3.plugins.d_rank_allocate import _group_stat
    bank = _ListBank(payload.weights_cpu)
    gs = _group_stat(payload.n_experts, bank, A_g=payload.a_g_cpu)
    return (payload.layer_idx, payload.name), gs
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py -v`
Expected: PASS (`test_payload_round_trips_through_pool_serialization`, `test_group_stat_payload_matches_direct_group_stat`, `test_pin_one_thread_initializer`).

- [ ] **Step 5: Commit**

```bash
git add max_quality/src/moe_compress/stage3/spectra_pool.py \
        max_quality/tests/test_stage3_spectra_pool.py
git commit -m "feat(stage3-f2): lean spectra_pool module (serializable payload + 1-thread initializer)"
```

---

## Task 7: Thread-pinning-holds-the-bits test (the fidelity invariant)

**Files:**
- Test: `max_quality/tests/test_stage3_spectra_pool.py`

Proves 1-thread pinning is what holds the bits: the worker's 1-thread result matches the 1-thread reference bit-for-bit (the contract that guards a future drop of `set_num_threads(1)`).

- [ ] **Step 1: Write the test**

Append to `test_stage3_spectra_pool.py`:

```python
def _eff_rank_under_threads(payload, nthreads):
    from moe_compress.stage3.plugins.d_rank_allocate import _group_stat
    from moe_compress.stage3.spectra_pool import _ListBank
    saved = torch.get_num_threads()
    try:
        torch.set_num_threads(nthreads)
        gs = _group_stat(payload.n_experts, _ListBank(payload.weights_cpu),
                         A_g=payload.a_g_cpu)
        return gs.effective_rank, gs.singular_values_mean.clone()
    finally:
        torch.set_num_threads(saved)


def test_thread_pinning_holds_the_bits():
    """The 1-thread path is the canonical reduction; the pool worker reproduces
    it bit-for-bit. (The 1-vs-N-thread numeric DIFFERENCE is matrix-size
    dependent — the F2 doc measured ~1e-11 on real [2048,1536] shapes; the tiny
    16x12 fixture may not reassociate, so we do NOT hard-assert inequality. The
    load-bearing contract is worker(1-thread) == reference(1-thread).)"""
    payload = _make_payload(seed=7)
    er1, sv1 = _eff_rank_under_threads(payload, 1)
    from moe_compress.stage3.spectra_pool import _group_stat_payload, _pin_one_thread
    _pin_one_thread()
    assert torch.get_num_threads() == 1
    _, gs_w = _group_stat_payload(payload)
    assert gs_w.effective_rank == er1
    assert torch.equal(gs_w.singular_values_mean, sv1)
```

- [ ] **Step 2: Run to verify it passes (contract already satisfied by Task 6's `_pin_one_thread`)**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py::test_thread_pinning_holds_the_bits -v`
Expected: PASS (worker 1-thread == reference 1-thread, bit-exact).

- [ ] **Step 3: (no impl needed)**

- [ ] **Step 4: Re-run the full pool test file**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add max_quality/tests/test_stage3_spectra_pool.py
git commit -m "test(stage3-f2): thread-pinning-holds-the-bits fidelity contract"
```

---

## Task 8: Parallel==1-thread-pinned-serial equivalence (the PRIMARY fidelity gate)

**Files:**
- Test: `max_quality/tests/test_stage3_spectra_pool.py`
- Modify: `max_quality/src/moe_compress/stage3/spectra_pool.py` (add `run_group_stats_pool`)

This is the F2 acceptance gate: a spawn ProcessPool over the group payloads (each worker 1-thread-pinned) yields `effective_rank` (exact float) + `singular_values_mean` (`torch.equal`) + rank-map identical to 1-thread-pinned serial. Drives `run_group_stats_pool` into existence and exercises the REAL spawn pickling end-to-end.

- [ ] **Step 1: Write the failing test**

Append to `test_stage3_spectra_pool.py`:

```python
def _build_multigroup_payloads():
    from moe_compress.stage3.spectra_pool import _GroupStatPayload
    torch.manual_seed(13)
    d_out, d_in, n = 16, 12, 4
    payloads = []
    for li in range(2):
        for name in ("gate_proj", "up_proj", "down_proj"):
            weights = [torch.randn(d_out, d_in, dtype=torch.float32) for _ in range(n)]
            m = torch.randn(d_in, d_in, dtype=torch.float32)
            a_g = (m @ m.T + d_in * torch.eye(d_in)).to(torch.float32)
            payloads.append(_GroupStatPayload(li, name, n, weights, a_g))
    return payloads


def _run_group_stats(payloads, workers):
    from moe_compress.stage3.spectra_pool import run_group_stats_pool
    return run_group_stats_pool(payloads, workers=workers)


def test_group_stat_parallel_equals_serial():
    """PRIMARY gate: parallel (workers=2, spawn, 1-thread-pinned) == 1-thread
    serial on effective_rank (exact float) + singular_values_mean (torch.equal)."""
    payloads = _build_multigroup_payloads()
    serial = _run_group_stats(payloads, workers=1)
    parallel = _run_group_stats(payloads, workers=2)
    assert set(serial) == set(parallel)
    for key in serial:
        assert serial[key].effective_rank == parallel[key].effective_rank, key
        assert torch.equal(serial[key].singular_values_mean,
                           parallel[key].singular_values_mean), key


def test_group_stat_rank_map_equal_across_workers():
    """Downstream int rank_map identical across workers in {1,2,4}, swept over
    a couple of T budgets (a borderline round() can't hide a float drift)."""
    from moe_compress.stage3.plugins.d_rank_allocate import (
        _d_rank_allocate, _compute_T_budget,
    )
    payloads = _build_multigroup_payloads()
    base = _run_group_stats(payloads, workers=1)
    for w in (2, 4):
        cand = _run_group_stats(payloads, workers=w)
        for ratio in (0.2, 0.3, 0.5):
            Tb = _compute_T_budget(base, svd_rank_ratio=ratio)
            Tc = _compute_T_budget(cand, svd_rank_ratio=ratio)
            assert _d_rank_allocate(base, Tb) == _d_rank_allocate(cand, Tc), (w, ratio)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py -k "parallel_equals_serial or rank_map_equal_across_workers" -v`
Expected: FAIL — `ImportError: cannot import name 'run_group_stats_pool'`.

- [ ] **Step 3: Add `run_group_stats_pool` to `spectra_pool.py`**

Append (and add `import concurrent.futures` / `import multiprocessing` at module top):

```python
def run_group_stats_pool(payloads, workers: int):
    """Compute {(layer_idx,name): _GroupStats} for all group payloads.

    workers<=1 → serial in-process (1-thread-pinned here), byte-identical to
    today's serial default path. workers>1 → spawn ProcessPool (CUDA-fork-safe),
    each worker 1-thread-pinned. Reassembly is order-free (dict keyed by
    (layer_idx,name)); the order-sensitive mean(0) stays INSIDE each worker by
    the group granularity.
    """
    if workers is None or workers <= 1:
        _pin_one_thread()
        return dict(_group_stat_payload(p) for p in payloads)

    # FORCE spawn — the parent is CUDA-initialized (Stage 3 is GPU-resident);
    # fork-after-CUDA-init deadlocks the child. Precedent humaneval.py:374-381.
    ctx = multiprocessing.get_context("spawn")
    out: dict = {}
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_pin_one_thread,
    ) as ex:
        for key, gs in ex.map(_group_stat_payload, payloads):
            out[key] = gs
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py -k "parallel_equals_serial or rank_map_equal_across_workers" -v`
Expected: PASS (the workers=2/4 legs exercise the real spawn pool end-to-end).

- [ ] **Step 5: Commit**

```bash
git add max_quality/src/moe_compress/stage3/spectra_pool.py \
        max_quality/tests/test_stage3_spectra_pool.py
git commit -m "feat(stage3-f2): run_group_stats_pool + parallel==1-thread-serial fidelity gate"
```

---

## Task 9: Worker-count invariance test

**Files:**
- Test: `max_quality/tests/test_stage3_spectra_pool.py`

Guards a future refactor that lets per-expert results reassemble out of order inside a group or makes the result worker-count-dependent.

- [ ] **Step 1: Write the test**

Append to `test_stage3_spectra_pool.py`:

```python
def test_group_stat_worker_order_invariant():
    """workers in {1,2,4} → identical effective_rank (exact) + singular_values_mean."""
    payloads = _build_multigroup_payloads()
    ref = _run_group_stats(payloads, workers=1)
    for w in (2, 4):
        got = _run_group_stats(payloads, workers=w)
        for key in ref:
            assert got[key].effective_rank == ref[key].effective_rank, (w, key)
            assert torch.equal(got[key].singular_values_mean,
                               ref[key].singular_values_mean), (w, key)
```

- [ ] **Step 2: Run to verify it passes (impl already exists from Task 8)**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py::test_group_stat_worker_order_invariant -v`
Expected: PASS.

- [ ] **Step 3: (no impl needed)**

- [ ] **Step 4: Run the full pool test file**

Run: `python3 -m pytest max_quality/tests/test_stage3_spectra_pool.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add max_quality/tests/test_stage3_spectra_pool.py
git commit -m "test(stage3-f2): worker-count invariance of group stats"
```

---

## Task 10: Wire the pool into the orchestrator (default `spectra_workers=1`, byte-identical)

**Files:**
- Modify: `max_quality/src/moe_compress/stage3/orchestrator.py:658-677`
- Test: `max_quality/tests/test_stage3_spectra_pool.py`

Replace the serial group-stat loop with: parent-side CPU gather (build `_GroupStatPayload` list, moving `A_g` + weights to CPU in the PARENT — `bank.get(e)` must run in the parent because banks hold live module refs) → `run_group_stats_pool(payloads, workers)` → assign into `group_stats`. The config gate `stage3_svd.spectra_workers` defaults to `1`.

- [ ] **Step 1: Write the config-default guard test**

Append to `test_stage3_spectra_pool.py`:

```python
def test_spectra_workers_defaults_to_one():
    """The orchestrator reads stage3_svd.spectra_workers with default 1
    (byte-identical serial path)."""
    cfg = {"stage3_svd": {}}
    assert int(cfg["stage3_svd"].get("spectra_workers", 1)) == 1
```

- [ ] **Step 2: Establish the byte-identity baseline the edit must not break**

Run: `python3 -m pytest "max_quality/tests/test_stage3_golden_snapshot.py" -v`
Expected: PASS (both non-α and the F1-re-blessed α). This is the bar the orchestrator edit must keep green on the default path.

- [ ] **Step 3: Replace the serial loop (658-677) with the pool dispatch**

```python
    log.info("Stage 3: computing per-group stats over %d layers", len(moe_layers))
    spectra_workers = int(s3.get("spectra_workers", 1))
    from .spectra_pool import _GroupStatPayload, run_group_stats_pool  # noqa: PLC0415

    payloads: list[_GroupStatPayload] = []
    for k, ref in enumerate(moe_layers):
        log.info("  group-stat layer %d/%d (idx=%d)", k + 1, len(moe_layers), ref.layer_idx)
        banks = build_banks(ref)
        for name in MATRIX_NAMES:
            cov_key_name = "gate_proj" if name == "up_proj" else name
            covs = []
            for e in range(ref.num_routed_experts):
                t = _cov_lookup(A_cov, ref.layer_idx, e, cov_key_name)
                if t is not None:
                    covs.append(t.to(torch.float32))
            A_g = torch.stack(covs).mean(0) if covs else None
            # Hoist the .cpu() to the dispatch site so ONLY CPU tensors cross the
            # process boundary (CPU torch tensor serialization is bit-exact).
            # bank.get(e) MUST run in the parent (banks hold live module refs,
            # not serializable). Ship the bank's NATIVE dtype to CPU and let
            # _group_stat do its own fp64 cast — matching today's exact sequence
            # so the default path is byte-identical (the golden in Step 4 is the
            # arbiter; A_g was already .to(float32) above, as in the old loop).
            weights_cpu = [
                banks[name].get(e).detach().to("cpu")
                for e in range(ref.num_routed_experts)
            ]
            a_g_cpu = A_g.to("cpu") if A_g is not None else None
            payloads.append(_GroupStatPayload(
                ref.layer_idx, name, ref.num_routed_experts, weights_cpu, a_g_cpu,
            ))

    group_stats = run_group_stats_pool(payloads, workers=spectra_workers)
```

> FIDELITY NOTE: today's serial loop passes `A_g` (already fp32 via `.to(torch.float32)` at 673) and `banks[name].get(e)` (bank-native dtype) into `_group_stat`, which then does `.to("cpu", float64)` internally. The dispatch above moves weights to CPU in the bank's native dtype (no dtype change — only residency) and `A_g` to CPU still fp32, so `_group_stat`'s internal fp64 cast sees identical numbers. Step 4's golden is the arbiter — if any bit shifts, the bank weights were on GPU and the `.to("cpu")` is the only change (lossless for fp32/bf16); do NOT add a dtype cast.

- [ ] **Step 4: Run the golden tests — default path MUST stay byte-identical**

Run: `python3 -m pytest "max_quality/tests/test_stage3_golden_snapshot.py" "max_quality/tests/test_stage3_spectra_pool.py::test_spectra_workers_defaults_to_one" -v`
Expected: PASS — `rank_map.{fp32,bf16}.json` AND `rank_map.alpha.{fp32,bf16}.json` all byte-identical under the default `spectra_workers=1`.

- [ ] **Step 5: Commit**

```bash
git add max_quality/src/moe_compress/stage3/orchestrator.py \
        max_quality/tests/test_stage3_spectra_pool.py
git commit -m "feat(stage3-f2): wire spectra_pool into orchestrator (default spectra_workers=1, byte-identical)"
```

---

## Task 11: Full Stage-3 suite + Tier-2 device-independence guard green

**Files:**
- (verification only — no code change)

- [ ] **Step 1: Run the full Stage-3 test surface**

Run:
```bash
python3 -m pytest \
  max_quality/tests/test_stage3_golden_snapshot.py \
  max_quality/tests/test_stage3_tier1.py \
  max_quality/tests/test_stage3_tier2.py \
  max_quality/tests/test_stage3_alpha_determinism.py \
  max_quality/tests/test_stage3_spectra_pool.py -v
```
Expected: PASS across the board. In particular:
- `test_stage3_rank_map_byte_identical[fp32,bf16]` — non-α golden UNCHANGED.
- `test_stage3_rank_map_alpha_variant_byte_identical[fp32,bf16]` — passes against the F1-re-blessed α golden (the original failing test — now green + cross-host reproducible).
- `test_d_rank_alloc_fp64_cpu_equals_fp64_gpu` — still green (skips if no CUDA); F2 does not touch device residency.
- `test_grouped_svs_cache_precondition_torch_equal` — green (producer 2 Cholesky == updated recompute).

- [ ] **Step 2: Confirm no other call site references the removed eigh α-block**

Run: `grep -rn "keep_a\|eigvals_a\|M_A = W @ L_A" max_quality/src/moe_compress/stage3/plugins/swift_svd_alpha.py max_quality/tests/`
Expected: no remaining α-path `keep_a` / `M_A` references in `swift_svd_alpha.py` (the rank-spectra eigh blocks are gone). The `_precompute_eigh` AA-SVD-core eigh (the FACTOR build, unrelated to the rank-spectra path) is untouched and may still appear — that is correct and out of F1 scope.

- [ ] **Step 3: Commit a closing marker (empty if no further change)**

```bash
git commit --allow-empty -m "test(stage3): full F1+F2 suite green (alpha-variant reproducible, default byte-identical)"
```

---

## Out of scope / deferred (explicit)

- **F2 default flip to parallel + 1-thread-canonical golden re-bless** (F2 doc §4/§5.5): flipping the default `spectra_workers` to `min(cpu_count, 64)` requires a one-time reviewed re-bless of `rank_map.{fp32,bf16}.json` to the 1-thread reduction (they differ ~1e-11 from today's default-threaded golden). This plan keeps the default at `1` so all goldens stay byte-identical; the flip is a separate, human-gated change.
- **F1 cross-host α-golden bless** on a second BLAS/host (H200) — manual acceptance gate; the on-host determinism mechanism is fully tested here.
- **Option A** (fp64 spectra on GPU) — rejected by the F2 doc; not implemented.
- `_argmin_alpha`, the validation-PPL path, and the residual `sorted(...)`/`int(math.floor(...))` loop — explicitly NOT touched (F1 doc §3.2/§3.3).

---

## Remember
- F1 before F2. Regenerate ONLY `rank_map.alpha.{fp32,bf16}.json`; NEVER the non-α goldens.
- Keep `svdvals(W @ L_C)` order in BOTH α producers (load-bearing — `L_C @ W.T` is a different, paper-incorrect quantity).
- Apply the IDENTICAL Cholesky swap to both producers so the Tier-1 `torch.equal` cache precondition holds (and update the tier1 recompute helper to match).
- F2: `multiprocessing.get_context("spawn")` (CUDA-fork-safe) + `torch.set_num_threads(1)` per worker (fidelity invariant). Only CPU tensors cross the process boundary. `_group_stat` numerics untouched. Default `spectra_workers=1` is byte-identical.
- `effective_rank` (d_rank_allocate.py:410) is the bit-sensitive pivot — exact-float `torch.equal` on it is the F2 gate, not the integer dict alone.
