# PLAN — Fix MergeMoE CUDA `lstsq` crash (`gelsd` → CUDA-safe driver)

**Workstream B / Stage 2.** Plan only. No implementation in this branch.

- **File touched (implementation, later):** `max_quality/src/moe_compress/stage2/mergemoe.py` **only.**
- **Tests touched (later):** `max_quality/tests/test_stage2_plugin_mergemoe_step.py` (driver alignment — see §6).
- **Branch:** `plan/stage2-mergemoe-gels`.
- **Urgency:** LOW. This is an **opt-in** path (`merge_step="mergemoe"`); the
  production default is `merge_step="freq_weighted"` (the 30pct / SC recipe),
  which is byte-identical to legacy and does **not** touch this code. This is a
  correctness/robustness fix for a path that currently crashes on GPU.

> Path note: the bug report calls it `stage2/mergemoe.py:318`. The real
> repo path is `max_quality/src/moe_compress/stage2/mergemoe.py`; the line
> numbers (224, 287, 318) match `origin/main` exactly.

---

## 1. The bug (verified against `origin/main`, reproduced on CUDA)

`mergemoe.py:318`:

```python
sol = torch.linalg.lstsq(P, Q_stack_T, driver="gelsd")
```

`device` is resolved at `mergemoe.py:224` as `member_downs[0].device`. The caller
(`merging.py`, `_merge_experts_inplace`, the `effective_merge_step == "mergemoe"`
block ~L325–339) passes the model's **own** weight tensors — which on any GPU run
live on the GPU banks. So `P`, `Q_stack_T`, and the solve all sit on CUDA.

`torch.linalg.lstsq` on CUDA supports **only** `driver="gels"`. Reproduced
verbatim on torch 2.11.0+cu130:

```
RuntimeError: torch.linalg.lstsq: `driver` other than `gels` is not supported on CUDA
```

**Conclusion:** `merge_step="mergemoe"` raises on every GPU run. It has only ever
worked on CPU (where the unit tests run with `device=None`). The CPU-only test
suite never exercised the CUDA path, so the crash was latent.

---

## 2. What `P` is, and the conditioning regime (the hard question)

From the math (module docstring, Eq. 5) and the code:

- `P = σ(W_G_merged·X̂) ⊙ (W_U_merged·X̂)`, shape **`(T, d_int)`**, where
  `W_*_merged = Σ_j b_j W_*^j` is the freq-weighted merged gate/up.
- `T = token_cap` (default **1024**); `d_int = 512` for the Qwen3.6-35B-A3B
  target. So `P` is **tall** (`T ≥ d_int`, 2× margin) → the least-squares system
  is over-determined and **generically full column rank**.
- The solve is already **condition-gated** at `mergemoe.py:287`:
  `if not (cond_P < 1e8): return freq_weighted_down(...)`. The `1e8` threshold
  is pinned by `SC_STAGE12_COMPREHENSIVE_PLAN.md §581` (risk-mitigation R1) and
  guarded by `test_mergemoe_cond_threshold_constant_matches_plan`.

**Empirical conditioning of real `P`.** A `(1024, 512)` SwiGLU intermediate built
from realistic small-magnitude expert weights on random calibration tokens has
`cond(P) ≈ 6` (measured). SwiGLU intermediates on real tokens are well-conditioned
in the common case; the `1e8` gate exists to catch the pathological
rank-deficient cluster (e.g. dead experts, degenerate calibration), which is
sent to the freq-weighted fallback **before** the solve.

---

## 3. `gels` (QR) vs `gelsd` (SVD): numerical justification

`gelsd` = SVD-based, computes the **minimum-norm** least-squares solution and is
robust to rank-deficiency (it truncates tiny singular values). `gels` = QR-based,
faster, but assumes `P` has **full rank**; on a rank-deficient `P` the QR solve
is ill-defined / unstable.

Measured `gels`(CUDA) vs `gelsd`(CPU) on `(1024, 512)` × `N=4`:

| regime | `cond(P)` | solution-vector rel diff | merged-down rel diff |
|---|---|---|---|
| realistic well-conditioned | ~6 | — | **2.1e-06** |
| synthetic well-cond | ~1e2 | 1.2e-05 | — |
| synthetic ill-cond (sub-gate) | ~1e6–3e7 | **~1e2–3e3** (diverges) | — (not tested for merged-down) |

> The ill-conditioned-band solution rel diff is **construction-dependent**: a
> fresh reproduction on real `(1024, 512)` shapes (torch 2.11+cu130) gives
> `1e2–3e3` for `cond(P) ∈ [1e6, 3e7)` — orders of magnitude larger than the
> earlier `~0.8–1.3` estimate. The conclusion is unchanged and in fact
> strengthened: in this band `gels` and `gelsd` disagree massively on the
> solution. Critically (see §4), in this band the `gels` output is **finite**
> and its **residual is no worse than (often lower than) `gelsd`'s** — the only
> signal that flags the divergence is the **solution norm** (`‖x_qr‖ ~1e8–1e9`
> vs `~1e6`).

**Key findings:**

1. **In the well-conditioned regime** (the realistic case, and the only regime
   reachable past the gate for healthy clusters), `gels` reproduces `gelsd` to
   ~**2e-6 relative** on the actual output of interest — the **merged
   `down_proj` weight** `W_D^merged = Σ_j b_j W_D^j @ T1_block_j`. That is the
   quantity golden snapshots pin, not `T1` itself.
2. **In the ill-conditioned-but-sub-gate regime** (`1e6 ≤ cond < 1e8`), `gels`
   and `gelsd` genuinely **disagree on the solution** (rel diff ~1.0). This is
   not a bug — SVD picks the minimum-norm solution, QR does not, and both can
   "fit" `P→Q` with low residual. So **the `1e8` gate alone does NOT guarantee
   gels-safety**: a cluster with `cond(P)` in `[1e6, 1e8)` survives the gate but
   would get a materially different `T1` from `gels` vs `gelsd`.

This is the decisive numerical point: the existing `1e8` gate was tuned for
**SVD** (`gelsd`), which is well-defined up to its threshold. A bare driver swap
to `gels` widens the regime in which the two drivers diverge and is **not** a
free swap on principle, even though it is empirically harmless on healthy
clusters.

---

## 4. Decision — recommended fix

Three candidates were considered:

- **(A) Plain `gels` on-device.** GPU-fast. Correct on well-conditioned `P`
  (the common case). Risk: on sub-gate-but-ill-conditioned `P` (`cond ∈
  [~1e6, 1e8)`) the QR solution diverges from the SVD solution the snapshots
  and the paper math assume. Changes the meaning of the `1e8` gate.
- **(B) Keep `gelsd` on CPU** (`P.cpu()` → solve → `.to(device)`). Bit-for-bit
  consistent with the golden snapshots and the paper's SVD posture. But the
  CPU `gelsd` solve is the ~426× slower path the report flags; per-layer cost
  on the real model is unacceptable for a GPU run.
- **(C) `gels` on-device, with a tightened **pre-solve** cond-gate so the
  ill-conditioned `[1e6, 1e8)` band is routed to the existing freq-weighted
  fallback BEFORE the solve.** Correct AND GPU-fast.

### Recommendation: **Option C — `gels` on-device + a tightened CUDA-path cond-gate (`1e6`) so `gels` only runs on provably full-rank `P`.**

Concretely (to be implemented in the follow-up, `mergemoe.py` only):

1. **Driver selection by device.** When `P.is_cuda`, use `driver="gels"`;
   when `P` is on CPU, keep `driver="gelsd"` (preserves the existing CPU
   golden snapshots **byte-identical** — see §5). This is a 2-line
   `driver = "gels" if P.is_cuda else "gelsd"` selection at L318.

2. **Make the gate guarantee gels-safety — lower the CUDA-path cond-gate
   *before* the solve (Option C2, PRIMARY).** The current `1e8` gate is an
   SVD-era threshold. For `gels` (QR) to be numerically trustworthy we need
   `P` comfortably full-rank.

   **C2 (PRIMARY recommendation):** on the CUDA path, lower the cond-gate to
   `_COND_THRESHOLD_CUDA = 1e6` (a new constant), keeping `_COND_THRESHOLD =
   1e8` on the CPU `gelsd` path. Any cluster with `cond(P) ∈ [1e6, 1e8)` then
   routes to the existing `_freq_weighted_down` fallback **before** the solve;
   `gels` only ever runs on provably full-rank `P` where §3 verified
   `gels ≈ gelsd` to ~`2e-6`. This catches the divergence band at the one
   place a cheap test can see it — the conditioning number, computed
   pre-solve. Costs the second constant + updates
   `test_mergemoe_cond_threshold_constant_matches_plan` (which must now pin
   both constants).

   **Why NOT a post-solve residual/finiteness guard (the earlier "C1"):** a
   fresh reproduction on real `(1024, 512)` shapes (torch 2.11+cu130) shows
   that in the `cond(P) ∈ [1e6, 1e8)` band the `gels` (QR) output is
   **finite**, and its **relative residual `‖P·T1ᵀ−Q‖/‖Q‖` is *lower* than
   `gelsd`'s** (QR minimizes residual on the data it sees; SVD trades residual
   for minimum norm). So a finiteness+residual guard **never fires** in the
   exact band it was meant to catch, while it is blind to the actual failure
   mode — **solution-norm blow-up** (`‖x_qr‖ ~1e8–1e9` vs the well-conditioned
   `~1e6`). A residual guard is therefore **inert** and is rejected.

   **If any post-solve guard is kept at all (optional belt-and-suspenders),**
   it must key on **solution-norm blow-up relative to the freq-weighted scale**
   — e.g. fall back when `‖T1‖` exceeds the freq-weighted-down norm by some
   large factor — **NOT** on residual or finiteness. This is secondary to the
   pre-solve cond-gate (C2), which already excludes the divergence band; it
   exists only to defend against a `cond(P)` estimate that under-reports.

   **CPU goldens unchanged:** the CPU path keeps `_COND_THRESHOLD = 1e8` and
   `driver="gelsd"`, so every existing CPU golden snapshot stays byte-identical
   (see §5).

**Why not B:** the 426× CPU penalty defeats the purpose; the user's perf posture
(`feedback_speedup_questions_target_real_run`) makes a slow-but-correct path the
wrong default for a GPU run.

**Why not plain A:** leaves the `[1e6, 1e8)` divergence band silently mis-solved.
C is A plus a cheap **pre-solve** cond-gate (the `1e6` CUDA threshold), so it is
correct AND GPU-fast. Note a post-solve residual/finiteness guard would *not*
fix A — see §4.2: in that band `gels` is finite with a residual no worse than
`gelsd`, so the divergence must be excluded *before* the solve, not detected
after it.

---

## 5. Golden-snapshot impact

- **CPU snapshots: ZERO change.** The existing tests run on CPU (`device=None`)
  and the test reference solve at `test_*_mergemoe_step.py:L72 (call) + L108
  (comment)` uses `driver="gelsd"`. By selecting the driver on `P.is_cuda`, the
  **CPU path keeps `gelsd`** (and `_COND_THRESHOLD = 1e8`), so every existing CPU
  golden snapshot (`atol=1e-5, rtol=1e-4` at L110/L128; byte-identical at L155,
  `_snapshot_banks`) stays valid **unchanged**. The CUDA-only `1e6` gate (C2)
  never touches the CPU branch.
- **`_freq_weighted_down` reuse — verified sound.** The C2 fallback target is
  the same `_freq_weighted_down(W_D, b)` already used by the cond-gate at
  `mergemoe.py:285,294` (def at L332). It takes the fp32 member-`down` list plus
  scalar freq weights `b` and reduces to the legacy freq-weighted merge — the
  exact byte-identical-to-legacy path the default `merge_step="freq_weighted"`
  produces. Routing the `[1e6, 1e8)` CUDA band into it is therefore a no-new-code
  reuse of an already-pinned merge.
- **CUDA path: no existing snapshot.** There is currently no CUDA golden
  snapshot for `mergemoe` (the path crashes, so none could exist). The fix
  *creates* the first working CUDA path. Per §3, on well-conditioned `P` the
  CUDA `gels` merged-down matches CPU `gelsd` to ~2e-6 rel — within the
  existing `atol=1e-5, rtol=1e-4` band. A new **CUDA-guarded** parity test
  (skipif no CUDA) asserting CUDA-`gels` ≈ CPU-`gelsd` within that tolerance is
  recommended as part of the implementation.

**Net:** the recommended fix is snapshot-preserving (CPU unchanged) and adds a
new, tolerance-bounded CUDA path. No existing golden needs re-baselining.

---

## 6. Paper-fidelity sign-off

**Needed: LIGHT sign-off, not a re-derivation.**

- The paper (Miao et al., *MergeMoE*, arXiv:2510.14436, Eq. 6) specifies
  `T₁ = Q·P†` — the **pseudoinverse / least-squares** solution. It does **not**
  mandate a LAPACK driver. `gelsd` and `gels` both target the same
  least-squares problem; they differ only in the rank-deficient corner.
- Therefore the driver swap is **within paper fidelity** *as long as* the
  rank-deficient corner is handled — which is exactly what the tightened
  CUDA-path cond-gate (C2, `cond < 1e6`) does: it routes any near-rank-deficient
  `P` to the freq-weighted fallback **before** `gels` runs. The paper's `P†` is
  the minimum-norm (SVD) solution; on full-rank `P` (the gated regime) `gels`
  and `gelsd` give the same answer, so fidelity holds.
- **Sign-off ask:** confirm that "device-dependent LAPACK driver, with a
  full-rank guard ensuring gels≈the paper's `P†`" is an acceptable
  implementation deviation. This should be recorded as a new deviation tag in
  the module docstring, e.g. **`D-mergemoe-cuda-driver`**, alongside the
  existing `D-mergemoe-*` tags. No change to the merge **algorithm**.

The existing docstring comment at L315–316 ("driver=\"gelsd\" — SVD-based,
robust to rank-deficient P …") must be updated to describe the device-dependent
selection and the rationale.

---

## 7. Implementation checklist (follow-up branch, NOT here)

1. `mergemoe.py` L318: `driver = "gels" if P.is_cuda else "gelsd"`; pass to
   `lstsq`.
2. `mergemoe.py`: add a new CUDA-path cond constant `_COND_THRESHOLD_CUDA = 1e6`
   (keep `_COND_THRESHOLD = 1e8` for CPU). At the gate (L287), select the
   threshold on `P.is_cuda` so the `[1e6, 1e8)` band routes to
   `_freq_weighted_down` **before** the solve (Option C2). Do **NOT** add a
   post-solve residual/finiteness guard — it is inert in this band (§4.2). An
   optional solution-norm-blow-up guard (relative to the freq-weighted-down
   scale) may be added as belt-and-suspenders, but never a residual one.
3. `mergemoe.py` L315–316 + module docstring: replace the `gelsd`-only comment;
   add deviation tag `D-mergemoe-cuda-driver`.
4. `test_stage2_plugin_mergemoe_step.py`: (a) update
   `test_mergemoe_cond_threshold_constant_matches_plan` to pin **both**
   `_COND_THRESHOLD = 1e8` (CPU) and `_COND_THRESHOLD_CUDA = 1e6`; (b) add a
   CUDA-guarded (`@pytest.mark.skipif(not torch.cuda.is_available())`) parity
   test asserting CUDA-`gels` merged-down ≈ CPU-`gelsd` within
   `atol=1e-5, rtol=1e-4`. Leave all existing CPU tests untouched (they keep
   `gelsd` and the `1e8` gate).
5. Run the full `test_stage2_plugin_mergemoe_step.py` on a CUDA box to confirm
   the crash is gone and the new parity test passes.

**Out of scope:** the caller in `merging.py` (no change needed — it already
passes correct device tensors), the `1e8` constant on the **CPU** path (kept),
and any non-`mergemoe` merge step.

---

## 8. Open questions for sign-off

1. **RESOLVED — C2 (per-device gate constants), not C1.** The earlier "C1 vs C2,
   decide at review" question is closed: a fresh reproduction (torch 2.11+cu130,
   real `(1024, 512)` shapes) shows the C1 post-solve residual/finiteness guard
   is **inert** — in the `[1e6, 1e8)` divergence band `gels` is finite and its
   residual is *lower* than `gelsd`'s, so the guard never fires. The only
   discriminating pre-solve signal is the conditioning number. The plan therefore
   adopts **C2**: a CUDA-path `_COND_THRESHOLD_CUDA = 1e6` gate (CPU keeps `1e8`),
   routing `[1e6, 1e8)` to `_freq_weighted_down` before the solve. (No "escalate
   to C2 only if the residual guard is noisy" path — that escalation would never
   trigger, since the residual guard is silent by construction.)
2. Acceptable to record `D-mergemoe-cuda-driver` as a deviation, or does the
   user want the CPU-`gelsd` posture preserved on GPU at the 426× cost
   (Option B)? Plan recommends the deviation.
3. Tolerance for the new CUDA parity test — reuse the existing
   `atol=1e-5, rtol=1e-4`? Plan recommends yes (matches measured ~2e-6 rel).
