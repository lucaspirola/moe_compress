# PLAN_OPT_C_vectorized_reservoir.md

## Implementation Plan: Optimization C — Vectorized Reservoir Sampling

**Spec sources**: `tasks/SC_FAST_PLAN_V3.md` §4 "Optimization C — Vectorized reservoir" (lines 277–296); Vitter, J.S. (1985). "Random sampling with a reservoir." *ACM Transactions on Mathematical Software*, 11(1):37–57.
**Target file**: `max_quality/src/moe_compress/stage2/profiling.py`
**Branch**: `main` (HEAD `3b9db2c`)
**Date**: 2026-05-27

---

### 1. Goal and Spec Citation

Replace the Python `for i in range(n)` loop in `_LayerInputAccumulator.add` (lines 58–80 of `max_quality/src/moe_compress/stage2/profiling.py`) with a fully vectorized implementation of Algorithm R (Vitter 1985). The change is anchored at two levels: (1) the spec `tasks/SC_FAST_PLAN_V3.md` §4 / Optimization C (lines 277–296), which cites a measured 45× speedup prototype at 1.89 ms/batch vs 85.9 ms/batch, saving ~6.8 min per SC row; and (2) the Vitter 1985 paper, which proves that Algorithm R produces a uniform reservoir sample over all tokens seen so far. The statistical contract is unchanged; only the per-batch RNG consumption pattern changes, so sample identities will differ from the current `.item()`-loop implementation. No other function in `profiling.py` is touched.

---

### 2. Exact Algorithm (Pseudocode)

The `add` method is split into three phases determined by `self.buffer` state:

```
def add(self, hidden: torch.Tensor) -> None:
    # Step 0: flatten and move to CPU — preserves existing contract.
    flat = hidden.reshape(-1, hidden.shape[-1]).detach().to("cpu")  # (n, H), contiguous
    n = flat.shape[0]
    if n == 0:
        return

    # ---------------------------------------------------------------
    # Phase A — First-ever call: deterministic prefix take.
    # Identical to current implementation; does NOT consume generator.
    # ---------------------------------------------------------------
    if self.buffer is None:
        take = min(n, self.max_samples)
        self.buffer = flat[:take].contiguous().clone()
        self.seen = n
        return

    # ---------------------------------------------------------------
    # Phase B — Buffer not yet full: fill remaining capacity first.
    # Also does NOT consume generator (every arriving token below cap
    # is kept unconditionally, probability = 1.0).
    # ---------------------------------------------------------------
    current_size = self.buffer.shape[0]
    if current_size < self.max_samples:
        remaining = self.max_samples - current_size
        fill_count = min(n, remaining)
        self.buffer = torch.cat(
            [self.buffer, flat[:fill_count]], dim=0
        ).contiguous()          # stays contiguous after cat
        self.seen += fill_count
        # If all n tokens fit below the cap, done.
        if fill_count == n:
            return
        # Otherwise trim flat to the unprocessed tail and fall through.
        flat = flat[fill_count:].contiguous()
        n = n - fill_count

    # ---------------------------------------------------------------
    # Phase C — Buffer is full (shape[0] == max_samples).
    # Vectorized Algorithm R for the remaining n tokens.
    # ---------------------------------------------------------------
    # pos[i] = 1-indexed global position of flat[i] in the entire stream
    # seen before this call  → (self.seen + 1) .. (self.seen + n)
    pos = torch.arange(
        self.seen + 1, self.seen + n + 1, dtype=torch.float64
    )                                              # (n,) float64 for precision

    # keep_prob[i] = min(max_samples / pos[i], 1.0)
    keep_probs = torch.clamp_max(
        self.max_samples / pos, 1.0
    ).to(torch.float32)                            # (n,) float32 for rand comparison

    # One uniform draw per token using the seeded generator.
    coin = torch.rand(n, generator=self._generator)   # (n,)  — CPU generator

    # Boolean mask of tokens that win their coin flip.
    keep_mask = coin < keep_probs                  # (n,) bool

    n_kept = int(keep_mask.sum())
    if n_kept > 0:
        # Indices into flat[] of kept tokens.
        kept_local = keep_mask.nonzero(as_tuple=False).squeeze(1)  # (n_kept,)

        # Uniform random target slot in [0, max_samples) per kept token.
        target_slots = torch.randint(
            0, self.max_samples, (n_kept,),
            generator=self._generator,
        )                                          # (n_kept,) int64

        # Vectorised in-place scatter: buffer[target_slots[k]] = flat[kept_local[k]]
        self.buffer.index_copy_(0, target_slots, flat[kept_local])

    self.seen += n
```

**Generator call order per batch** (hot path, Phase C only):
1. `torch.rand(n, generator=self._generator)` — one call for `n` coin flips.
2. `torch.randint(0, max_samples, (n_kept,), generator=self._generator)` — one call for target slots. This call is skipped entirely when `n_kept == 0`.

Both calls are to the CPU generator. No `.item()` calls. No GPU operations.

---

### 3. Statistical Contract

**Claim**: the new vectorized method produces the same marginal distribution as Algorithm R: after observing `N` tokens total, every token `t` has equal probability `min(max_samples / N, 1.0)` of being in the final reservoir.

**Proof (informal)**:

In classic Algorithm R for token at 1-indexed position `p` (where `p > max_samples`), the token is placed in a uniform random slot `j ∈ [0, max_samples)` with probability `max_samples / p`. Tokens at positions `p ≤ max_samples` are placed deterministically (unconditional keep in Phase A/B).

The vectorized implementation executes the same decision per token: token `i` in the current batch at global 1-indexed position `pos_i = self.seen + i + 1` is accepted with probability `min(max_samples / pos_i, 1.0)`, and if accepted, placed at a uniform random slot in `[0, max_samples)`. This is identical to the scalar loop decision for each individual token. The order of generator calls within a batch (coin flips for all `n` tokens first, then target slots for the `n_kept` accepted tokens) differs from the scalar loop (which interleaves coin flip and slot selection per token), but the marginal probability for any single token is unchanged.

**Caveat on the RNG stream**: the new implementation is not byte-compatible with the old loop. The sequence of random values emitted by `self._generator` for any given seed will differ between the two implementations, producing different token selections (different `buffer` contents). This is expected and documented at spec line 287. The final buffer for both implementations is a valid uniform reservoir sample; they are simply two different draws from the same distribution.

**Caveat on duplicate target slots** (deviation D1 — to be documented in the plugin docstring per [[paper-fidelity-review-loop]]):
When two kept tokens in the same batch select the same target slot, the textbook scalar Algorithm R would process them sequentially: token i places into slot s, then token j (j > i, in batch order) overwrites slot s. The vectorized `index_copy_(0, target_slots, source)` on CPU processes target indices in the order they appear in `target_slots`, so the same "last write wins" semantics applies — and because `kept_local` is in batch order (from `nonzero()`), the vectorized version produces the SAME outcome as the scalar loop *for any given accepted set + slot selection*. The two implementations differ only because they consume different generator outputs, not because of `index_copy_` semantics. (CPU `index_copy_` is deterministic; CUDA `index_copy_` is non-deterministic with duplicates, but this code path is CPU-only.)

---

### 4. Files to Touch

#### 4a. Modify

**`/home/lucas/ai/moe_compress/max_quality/src/moe_compress/stage2/profiling.py`**

- Lines 34–50 (class docstring): Add citation to SC_FAST_PLAN_V3.md §4-C and Vitter 1985. Note that the vectorized implementation changes the RNG consumption pattern (sample identities differ from the scalar-loop baseline, but the marginal distribution is identical Algorithm R). Document the deviation D1 (duplicate target slot resolution) explicitly.
- Lines 58–80 (`add` method): Replace entirely with the three-phase vectorized implementation from Section 2 above.
- No other lines in this file are touched. `_profile_layer` is out of scope.

#### 4b. Create

**`/home/lucas/ai/moe_compress/max_quality/tests/test_layer_input_accumulator_uniform.py`**

New statistical test file. Full specification in Section 6 below.

#### 4c. Goldens to Re-baseline

After exhaustive search of `max_quality/tests/test_stage2_*.py`, **no golden-snapshot test files for Stage 2 exist** in the current test suite. The filenames mentioned at spec line 296 (`test_stage2_output_space_cost_*.py`) were forward-looking. The files that exercise `_LayerInputAccumulator` at runtime are:
- `test_stage2_profiling.py` — shape/structural tests only
- `test_stage2_expert_distill.py` — uses synthetic `torch.randn` as `layer_inputs`, not accumulator output
- `test_stage2_plugin_expert_distill.py` — uses `_StubAcc`, does not instantiate the real class
- `test_stage2_plugin_layer_merge.py:152` — `isinstance` check only
- `test_stage2_pipeline_run_layer.py:344,370` — `is None` or `isinstance` checks only
- `test_stage2_plugin_output_space_cost.py:102` — sets `layer_input_acc = None`

**Conclusion**: no test file needs re-baselining. The implementer must run `grep -r "layer_input_acc\|_LayerInputAccumulator" max_quality/tests/` and confirm no test file asserts specific buffer byte content. If any new caller was added between this plan and implementation, regenerate it.

---

### 5. Public API Preservation

**Public API preserved (no changes to signatures or attribute layout):**

| Element | Before | After |
|---|---|---|
| `__init__(max_samples, *, seed)` | unchanged | unchanged |
| `add(hidden)` | replaced body | same signature |
| `get()` | unchanged | unchanged |
| `self.max_samples` | int | int |
| `self.buffer` | `Tensor \| None`, CPU, contiguous | CPU, contiguous — identical |
| `self.seen` | int | int |
| `self._generator` | `torch.Generator(device="cpu")` | `torch.Generator(device="cpu")` — identical |

**CPU constraint**: `.detach().to("cpu")` is applied to `hidden` at the very first line of `add`. All subsequent torch operations are on CPU. No GPU call is ever made inside `add`. No `.item()` is ever called.

---

### 6. Test Cases

#### 6a. Existing tests (must pass without modification)

From `test_stage2_profiling.py`:
- `test_layer_input_acc_caps_at_max_samples`
- `test_layer_input_acc_get_before_any_add_is_none`
- `test_layer_input_acc_is_deterministic_under_seed`
- `test_layer_input_acc_different_seeds_diverge`

From `test_stage2_expert_distill.py`:
- `test_layer_input_acc_initial_capture`
- `test_layer_input_acc_capped_at_max_samples`
- `test_layer_input_acc_reservoir_extends_then_replaces`  ← exercises Phase B→C split
- `test_layer_input_acc_get_before_any_add_returns_none`

#### 6b. New file: `test_layer_input_accumulator_uniform.py`

```python
"""Statistical uniformity tests for vectorized _LayerInputAccumulator (Opt C)."""
import math
import torch

from moe_compress.stage2.profiling import _LayerInputAccumulator


def test_mean_position_uniformity():
    """Per SC_FAST_PLAN_V3 §4-C lines 293-294: with N_seeds seeds, the grand
    mean of per-seed sampled-position means should be within 3σ of n_tokens/2.

    IMPORTANT: tokens are fed in *multiple* `add()` calls so that Phase C
    (reservoir replacement) is actually exercised. A single 100k-token
    add() would go through Phase A (deterministic prefix take) and never
    test the reservoir — that's why the chunked feed matters.
    """
    n_tokens = 100_000
    max_samples = 1024
    hidden_size = 4
    n_seeds = 1000
    chunk_size = 1000   # 100 chunks; first chunk → Phase A, rest → Phase B/C

    # Token t encoded as a row where flat[t, :] = float(t), so we can recover
    # the token index from any buffer row.
    flat = torch.stack([
        torch.full((hidden_size,), float(t)) for t in range(n_tokens)
    ])
    expected_mean_pos = (n_tokens - 1) / 2.0   # 49999.5

    mean_positions = []
    for seed in range(n_seeds):
        acc = _LayerInputAccumulator(max_samples=max_samples, seed=seed)
        for chunk in flat.split(chunk_size):
            acc.add(chunk)
        buf = acc.get()
        assert buf.shape == (max_samples, hidden_size)
        positions = buf[:, 0]
        mean_positions.append(positions.mean().item())

    grand_mean = float(torch.tensor(mean_positions).mean())
    # std(uniform[0, n_tokens)) ≈ n_tokens / sqrt(12); std-of-per-seed-mean
    # of max_samples draws ≈ that / sqrt(max_samples); std of grand_mean
    # over n_seeds ≈ that / sqrt(n_seeds).
    sigma_grand_mean = (
        (n_tokens / math.sqrt(12)) / math.sqrt(max_samples) / math.sqrt(n_seeds)
    )
    assert abs(grand_mean - expected_mean_pos) < 3.0 * sigma_grand_mean, (
        f"grand_mean={grand_mean:.1f} expected={expected_mean_pos:.1f} "
        f"3σ_bound={3*sigma_grand_mean:.1f}"
    )


def test_determinism_same_seed():
    """Same-seed same-output contract preserved."""
    torch.manual_seed(42)
    flat = torch.randn(200, 8)

    a = _LayerInputAccumulator(max_samples=50, seed=7)
    b = _LayerInputAccumulator(max_samples=50, seed=7)
    a.add(flat)
    b.add(flat)
    assert torch.equal(a.get(), b.get())


def test_no_item_call_on_gpu_tensor():
    """Guard against future regression that re-introduces .item() calls."""
    if not torch.cuda.is_available():
        return
    acc = _LayerInputAccumulator(max_samples=16, seed=0)
    x = torch.randn(4, 4, 8, device="cuda")
    acc.add(x)
    buf = acc.get()
    assert buf is not None
    assert buf.device.type == "cpu"
```

---

### 7. Risk Register

**R1 — Goldens change** (spec line 287)
Probability: medium. Buffer contents differ from the current implementation due to different RNG stream consumption. After exhaustive test-file search, no test byte-pins buffer contents. **Mitigation**: implementer reruns the grep audit at implementation time.

**R2 — First-batch initialization must be preserved exactly**
Phase A (`self.buffer is None`) is byte-for-byte the current code's behavior: deterministic prefix take, no generator consumption.

**R3 — Phase B→C transition (split batch)**
`test_layer_input_acc_reservoir_extends_then_replaces` exercises this. The implementer must trim `flat` to the post-fill suffix BEFORE entering Phase C, and update `self.seen += fill_count` first so `pos` is offset correctly.

**R4 — `.item()` on CUDA tensor**
`.detach().to("cpu")` first line stays. No `.item()` anywhere.

**R5 — Generator contention with `torch.manual_seed`**
`self._generator` is a separate `torch.Generator` object, unaffected by `torch.manual_seed`. Unchanged from current implementation.

**R6 — `index_copy_` with repeated target slots**
On CPU, `index_copy_` is deterministic and processes in order. Because `kept_local` is in batch order and the scalar loop also processes in batch order, the two implementations would produce identical buffers *for any given accepted set + slot selection*. The only difference is which tokens are accepted (different RNG stream). **Document deviation D1 in docstring** despite the equivalence claim — keep the audit-trail explicit.

---

### 8. Acceptance Gates (in order)

- [ ] **G1** — `pytest max_quality/tests/test_stage2_profiling.py` green
- [ ] **G2** — `pytest max_quality/tests/test_layer_input_accumulator_uniform.py` green
- [ ] **G3** — `pytest max_quality/tests/test_stage2_expert_distill.py` green
- [ ] **G4** — `pytest max_quality/tests/test_stage2_plugin_layer_merge.py max_quality/tests/test_stage2_plugin_expert_distill.py max_quality/tests/test_stage2_pipeline_run_layer.py` green
- [ ] **G5** — Golden audit grep: `grep -r "layer_input_acc\|_LayerInputAccumulator" max_quality/tests/` shows no byte-pinning assertions on accumulator buffer content
- [ ] **G6** — `pytest max_quality/tests/` full suite green
- [ ] **G7** — Commit on `main`:
  `perf(stage2): vectorize _LayerInputAccumulator.add (Algorithm R, 45x speedup)`
  Body: cites SC_FAST_PLAN_V3.md §4-C + Vitter 1985; notes RNG stream differs; confirms no goldens needed re-baseline.

---

### 9. Out of Scope

- `_profile_layer` and every other function in `profiling.py` — no changes.
- The public API of `_LayerInputAccumulator` — unchanged.
- Moving the accumulator to GPU — `.detach().to("cpu")` constraint preserved.
- Any other optimization in `SC_FAST_PLAN_V3.md` (Opts A, B, D) — separate work items.
- Changing calibration data, batch sizes, or any Stage 2 configuration knob.

---

**Order of work for the implementer:**
1. Modify `profiling.py` `add` method + docstring (with deviation D1 documented)
2. Run G1
3. Create `test_layer_input_accumulator_uniform.py`
4. Run G2, G3, G4
5. Audit goldens (G5)
6. Run G6 (full suite)
7. Commit (G7)
