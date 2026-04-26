# Stage memory profiles

Empirical CPU-RAM and CUDA-VRAM curves observed on a100-large
(1204 GB host RAM, 86 GB VRAM, NVIDIA A100-SXM4-80GB) for the
Qwen3.6-35B-A3B pipeline at `target.total_reduction_ratio = 0.30`.

These are the curves we'd expect any future run to roughly reproduce.
Big deviations point at a leak or a config change.

## Stage 3 — SVD

### Phase 1: skeleton build (`from_config` + `_resize_moe_stack_to_metadata`)

| Metric | Behavior |
|---|---|
| Wall-clock | ~7–8 min |
| VRAM | stays at 0.8 GB (CUDA driver overhead only — model still on CPU/meta) |
| RAM | climbs ~4 GB/min from baseline ~67 GB → ~120 GB |
| CPU% | ~10% (single-threaded init) |

### Phase 2: streaming load (`load_compressed_model` per-tensor swap)

| Metric | Behavior |
|---|---|
| Wall-clock | ~30–60 s |
| VRAM | jumps from 0.8 GB → ~62 GB as shards land |
| RAM | drops sharply (skeleton tensors freed on swap) — observed 124 → 85 GB |
| CPU% | spikes briefly to 12–14% during shard mmap reads |

If RAM CLIMBS instead of dropping during this phase, the streaming load
regressed (e.g. someone reintroduced a CPU-side `state_dict = ...` dict).
See `tests/test_load_compressed_model.py::test_streaming_swap_keeps_memory_bounded`
for the regression check.

### Phase 3: B-covariance collection (`_collect_pruned_input_covariance`)

| Metric | Behavior |
|---|---|
| Wall-clock | **~8.5 min/layer × 40 layers ≈ 5h40m** |
| VRAM | steady at ~62 GB (model resident, no further allocation) |
| GPU util | 65–75% during forward passes (calibration loop) |
| RAM | **+4.75 GB per completed layer** — linear, no spikes |
| CPU% | ~14% steady |

Per-layer Trackio metrics:
- `stage3/bcov_layer` — 1-indexed counter (1 → 40)
- `stage3/bcov_layer_idx` — model layer index (0 → 39 for Qwen3.6)
- `stage3/bcov_ram_used_gb` — host CPU RAM at end of that layer

**Why 4.75 GB/layer:** the `InputCovarianceAccumulator` stores per-(layer, expert,
matrix) input covariance at fp32 (default — Stage 2 explicitly opts into
bf16 via `covariance_storage_dtype: bfloat16`, Stage 3 does not). Per layer
at ~256 routed experts × ~17 MB/expert (hidden² + intermediate²) ≈ 4.4 GB
in tensors + Python overhead.

**Final RAM at end of Phase 3:** ~131 GB (after layer 1) + 39 × 4.75 ≈ **~316 GB**.

**Sizing rule:** Stage 3 requires a host with **at least ~350 GB CPU RAM**
(or set `B_acc.set_storage_dtype(torch.bfloat16)` to halve this — saves
~85 GB peak at the cost of ~ε accuracy in the SVD A-weighted norm).
a100-large at 1204 GB has ~880 GB headroom — comfortable. Do NOT run
Stage 3 on a host with <350 GB without the bf16 opt-in.

### Phase 4: AA-SVD per-expert factor + reconstruction-error metrics

| Metric | Behavior |
|---|---|
| Wall-clock | ~30–60 s/layer (closed-form SVDs, fast) |
| VRAM | brief spikes during per-(layer, matrix) SVD on cuda |
| RAM | mostly flat (originals dict snapshot ~70 GB held on CPU) |

Per-(layer, matrix) Trackio metrics:
- `stage3/recon_rel_err/{gate_proj,up_proj,down_proj}` — Frobenius
  ‖W − UₖVₖ‖ / ‖W‖. Healthy: < 0.1; rank-too-small if > 0.2.

### Phase 5 (optional): LBFGS block refine

Only fires if `s3.block_refine.enabled: true` in the config (currently
**off**). When on:

- Per-(layer, matrix) Trackio: `stage3/refine_loss_init/_final/_rel_drop`
- Healthy `rel_drop`: 0.10–0.50 (10–50% loss reduction). Negative or near-zero
  rel_drop = LBFGS not converging or already-optimal.

### Phase 6: Hub upload of Stage 3 checkpoint

| Metric | Behavior |
|---|---|
| Wall-clock | model dir (~50 GB) + `_stage3_original_weights.pt` (~70 GB at bf16) ≈ 5–10 min on Hub Xet |
| Network | hf_xet rate ~250 MB/s observed |

After upload completes, sidecar lives at `<base>-stage3/artifacts/_stage3_original_weights.pt`.
Stage 4 reads it from the bucket (same job) or re-downloads from Hub on resume.

## Stage 4 — EoRA

(To be filled in after the first successful run produces telemetry.)

## Stage 5 — Router KD

(To be filled in after the first successful run produces telemetry.)

## Stage 6 — Validation

(To be filled in after the first successful run produces telemetry.)

---

## How to read the live curves on Trackio

Open `https://huggingface.co/spaces/pirola/trackio`, pick project
`moe-compress-strategy-a`, then the run name. Curves to watch:

| Trackio key | Phase | Healthy reading |
|---|---|---|
| `sys/vram_used_gb` | All | 60–75 GB during Stages 3–5; should stay < 80 |
| `sys/ram_used_gb` | Stage 3 | climbs ~4.75 GB/layer linearly |
| `sys/gpu_util_pct` | Stages 3–5 forward passes | 60–90% during calibration / KD |
| `sys/cpu_pct` | All | 10–20% (low — pipeline is GPU-bound) |
| `stage3/bcov_layer` | B-cov phase | 1 → 40 over ~5h40m |
| `stage3/recon_rel_err/*` | AA-SVD phase | < 0.1 per matrix |
| `stage5/loss` | Router KD | monotonically decreasing |
| `stage5/grad_norm` | Router KD | bounded, not collapsing to 0 |
