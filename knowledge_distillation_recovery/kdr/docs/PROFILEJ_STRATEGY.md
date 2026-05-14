# Profile-J KD Recovery — Strategy and Recipe

**Audience:** reviewer / future contributor unfamiliar with this run.
**Scope:** what we are doing, why, and what the current recipe encodes.
**Status:** post run-1 collapse and revised recipe (commit `999dc1a`).

---

## 1. Goal

Produce a single GGUF-compatible mixed-precision quantization of
[`Zyphra/ZAYA1-8B`](https://huggingface.co/Zyphra/ZAYA1-8B) (an 8.84 B-parameter MoE model)
that:

1. fits in **≈3.25 GB** on disk,
2. runs on a **GTX 1050 Mobile** (sm_61, ~4 GB VRAM) under
   [`llama.cpp`](https://github.com/ggml-org/llama.cpp) at usable context length, and
3. recovers as much of the BF16 baseline's downstream quality as a
   short knowledge-distillation pass over 50 M tokens can deliver.

The compression target is encoded in the **Profile-J spec_map**
(see `kdr/configs/zaya1_8b_da_qad_profileJ_gguf.yaml`):

| Module pattern              | Format  | bpw     | Modules touched |
|-----------------------------|---------|---------|-----------------|
| `linear_fc1` (MoE gate+up)  | IQ2_XS  | 2.3125  | 640             |
| `linear_fc2` (MoE down)     | Q3_K    | 3.4375  | 640             |
| `linear_q`, `linear_k`      | IQ4_XS  | 4.25    | 120 + 120       |
| `o_proj` (attention out)    | IQ4_XS  | 4.25    | 80              |
| `embed_tokens` / `lm_head`  | Q5_K    | 5.5     | 1 (tied)        |
| `val_proj{1,2}` (CCA value) | F16     | 16      | carved out      |
| `router`, `norm`            | F16     | 16      | carved out      |
| KV cache (runtime)          | INT4    | 4 / 4   | per-channel K + per-token V |

Profile-J is the bottom of the recipe space we support — the IQ2_XS on
expert weights (the bulk of an 8 B MoE) is the dominant noise source.

## 2. KD framing

### 2.1 Loss

We use **forward Kullback–Leibler divergence** (a.k.a. forward-KL,
"FKLD") at the logits, averaged per token:

$$
\mathcal{L}_{\text{KD}}(s, t) = T^2 \cdot
\frac{1}{B T} \sum_{i=1}^{B T}
\text{KL}\bigl(\,\text{softmax}(t_i / T)\,\|\,\text{log\_softmax}(s_i / T)\,\bigr)
$$

where $s, t \in \mathbb{R}^{B \times T \times V}$ are student / teacher
logits and $T$ is the temperature. The $T^2$ factor undoes the gradient
shrink induced by temperature-softmax, following Hinton et al. (2015).

Reduction is **`batchmean`** (divide by token count, not by token count
× vocab size — modelopt's default `"mean"` would silently collapse the
gradient signal by a factor of $\sim 150\text{k}$).

Implementation lives in `kdr/src/kdr/kd_loss.py::forward_kld_loss`,
delegating to either modelopt's `LogitsDistillationLoss` (when
installed) or our `_NativeKLDLoss` (a 4-line PyTorch parity
implementation, bit-equal to modelopt at any $T$ — see
`tests/test_kld_parity.py`).

The native path uses `F.log_softmax(..., dtype=torch.float32)` so the
reduction accumulates in fp32 *internally* without materialising a full
`(B*T, V)` fp32 logits tensor (saves ~1.2 GB per micro at $V \approx 150\text{k}$).

### 2.2 Self-distillation

Teacher and student are **the same checkpoint** (`Zyphra/ZAYA1-8B`
BF16). The student's weights are wrapped by the Native backend's STE
parametrisations (Section 3.2) so its *forward* runs in fake-quant
space; the teacher stays in BF16 under `torch.no_grad()`. KD therefore
asks the student-quantised model to mimic the un-quantised model's
output distribution on a fixed calibration corpus.

This is "Chapter 3 DA-QAD" in our internal taxonomy (deployment-aware
quantization-aware distillation): we know exactly what runtime
quantization a downstream consumer will apply (Profile-J + INT4 KV),
and we train the student so that the post-quant forward already
approximates the BF16 teacher.

### 2.3 Router replay (LLR-0025)

Naive self-distillation through an MoE has a hidden failure mode:
expert routing on the post-quant student may pick different experts
than the BF16 teacher, especially early in training when KD loss is
high. The student then chases a *different* expert subset's outputs,
amplifying drift.

We sidestep this by **pinning the student's MoE routing to the
teacher's per-token expert selection** for the duration of each
micro-batch. The teacher's router outputs are captured via a forward
hook, the student's router output is swapped for the teacher's at the
same layer index, and gating proceeds from the same `(logits,
top_k_indices)` tuple. See
`kdr/src/kdr/adapters/router_replay.py` for the wiring.

## 3. Quantization mechanics

### 3.1 Recipe split

`partition_and_dispatch` (`kdr/src/kdr/quant/factory.py`) routes each
quantizer in the YAML's `quant` block to one of two backends:

- **ModelOpt** — handles the KV-cache quant (INT4 K per-channel + INT4
  V per-token via a `mtq.quantize` activation hook).
- **Native** — our hand-rolled STE simulator for the four GGUF
  super-block formats (IQ2_XS, Q3_K, IQ4_XS, Q5_K) on linear weights.

The split exists because GGUF super-block formats are not in
modelopt's matrix; we need them to produce a `.gguf` artifact that
llama.cpp's reference decoder can load.

### 3.2 STE simulator (Native backend)

For each Linear matched by the spec_map, we register an
`nn.utils.parametrize` parametrization on `.weight` that runs the
appropriate `*_quant_snap` function on every forward and wraps it in
the STE formula:

```python
def forward(self, w):
    snapped = <format>_quant_snap(w, axis=-1)  # block-wise quant→dequant
    return w + (snapped - w).detach()           # STE: fwd=snap, bwd=identity
```

Forward returns the fake-quanted tensor (so the matmul sees the
post-quant values); backward passes gradient identity through `w`
because the delta term is `.detach()`'d. The optimizer therefore sees
the *gradient at the pre-quant weight*, while the forward operates
*post-quant* — a classical straight-through estimator.

Each `_*_snap_block` function snaps a 256-element super-block to the
format's codebook (e.g. IQ2_XS does a 65,536-way argmin over
(magnitude × sign) codeword pairs per 8-element chunk). Implementations
in `kdr/src/kdr/quant/native_backend/ste_simulators.py`, codebook
constants transcribed verbatim from llama.cpp's `ggml-common.h` at a
pinned commit (`gguf_codebooks.py`).

### 3.3 Per-step snap cache (the keystone optimisation)

Within one optimizer step (grad_accum = 32 micros), the underlying
weight does not change — yet `nn.utils.parametrize` re-invokes the
parametrization on every `.weight` access (32 times per weight per
window). For Profile-J that's $1{,}401 \text{ Linears} \times 32 \text{ micros}
\approx 45\text{k}$ redundant snaps per step, each potentially several
TFLOPS (IQ2_XS argmin is ≈7.7 TFLOPS on a 4096×14336 expert weight).

The `_CachingSTEParametrization` base class (in
`kdr/src/kdr/quant/native_backend/backend.py`) caches the snapped
tensor on cache miss, returns the same snap reconstructed via the STE
formula on subsequent hits, and is invalidated by
`NativeBackend.invalidate_ste_cache()` immediately after each
`optim.step()`. The trainer hooks invalidation through a
`post_optim_step_callbacks` list (`kdr/training/loop.py`).

Empirically: **478× speed-up at steady state** (526 s/micro → 1.1
s/micro on B200), with one $\sim$250 s "miss" on micro 1 of each
window. Net per-step wall clock ≈ 284 s (vs. ~16,000 s pre-cache).
Autograd correctness is preserved: gradient still flows through the
live `w`; the cached snap only substitutes the forward value.

### 3.4 Auxiliary perf items

| Item | Where | Win |
|---|---|---|
| Hoist IQ2_XS joint codebook + transpose + `code_sq` to per-(device, dtype) cache | `gguf_codebooks.get_iq2xs_joint` | Eliminates ~5 k redundant codebook materialisations per step |
| `torch.index_select` for argmin gather (vs fancy indexing) | `_iq2xs_snap_block` | Contiguous gather kernel, modest |
| `F.log_softmax(..., dtype=fp32)` instead of full `.float()` upcast | `kd_loss._NativeKLDLoss.forward` | Avoids 1.2 GB transient fp32 logits tensor per micro |
| Drop `.clone()` on captured router outputs | `router_replay._capture_hook` | Eliminates 60+ memcpys per micro |
| Defer per-micro `.item()` sync of `last_real_loss` | `loop._step_one_micro` | One fewer `cudaStreamSynchronize` per micro |
| Env-gated bf16 IQ2_XS argmin | `_iq2xs_argmin_use_bf16` (off by default) | Future ~2× on the codebook search; gated by numerics test |

## 4. Distillation recipe (post run-1 revision)

The current YAML (commit `999dc1a` of `Profile-J GGUF`):

```yaml
distillation:
  loss: forward_kld
  temperature: 1.0           # endpoint of the linear ramp
  temperature_start: 4.0     # ramps τ 4.0 → 1.0 over the run (soft-to-hard
                             # curriculum; borrowed from max_quality stage-5)
  optimizer: adamw_bnb_8bit  # 8-bit AdamW from bitsandbytes >= 0.49.2
  learning_rate: 5.0e-5      # was 2.0e-4; see Section 5
  min_learning_rate: 5.0e-6  # was 4.5e-7; keeps ~10× cosine ratio
  weight_decay: 0.01
  betas: [0.9, 0.95]         # was [0.9, 0.999]; faster β₂ EMA for QAT
  grad_clip_norm: 0.5        # was 1.0
  warmup_steps: 60           # was 20 (now ~16% of run, was 5%)
  total_tokens: 50_000_000
  per_device_batch_size: 1
  gradient_accumulation: 32
  sequence_length: 4096
  log_every_n_steps: 5
  eval_every_n_steps: 50
  save_every_n_steps: 25     # was 50; tighter rollback granularity
  trainable_scope: full      # all 8.84B params trainable
  use_gradient_checkpointing: true
```

### 4.1 Temperature curriculum and τ-invariant tracking (borrowed from max_quality stage-5 KD)

`temperature_start: 4.0` activates a per-step linear ramp; the
`forward_kld_loss` call at each micro receives the schedule-aware τ.
High τ early smooths the teacher's softmax distribution so the STE
updates aren't chasing sharp logit peaks while IQ2_XS codebook
assignments are still settling; the schedule sharpens to τ=1.0 by the
final step once the student is in-basin. Defaults to constant
`temperature` when `temperature_start` is unset — fully backward
compatible.

Because raw KD loss is τ-dependent (the formula divides by T inside
the softmax and multiplies by T² outside), we track
**`raw_kl = loss / T²`** in addition to loss. `raw_kl` is comparable
across any τ schedule — it is the metric we use for the save-best
pointer (Section 6.3).

### 4.2 EMA-best step pointer

Every committed window updates an EMA of `raw_kl` with α=0.2 (mq
stage-5's choice). When the EMA improves on the best-so-far, the
current step's metadata (`step`, `raw_kl`, `raw_kl_ema`, `temperature`,
`loss`) is recorded.

**Burn-in.** Best-pointer updates are gated to `step > warmup_steps`.
Two reasons: (a) during warmup the lr is ramping and AdamW's
second-moment EMA is settling, so any "best" inside that window is
not representative of the true loss landscape; (b) the curriculum has
T at its highest in the early steps, which divides raw_kl by a large
T² and artificially deflates the metric — without the gate the
pointer would latch a high-T minimum that no later (genuinely better
but temperature-sharpened) basin could ever beat.

At every `save_every_n_steps` boundary the trainer writes a small
JSON pointer file
`<artifacts_dir>/kdr_<mode>_best_partial_pointer.json` recording the
best step and the corresponding partial-dir name. A post-mortem
(or the final-save selector) resolves the pointer to find the partial
representing the lowest-EMA basin — robust against the run-1 pattern
where a precursor spike at step 15 preceded a step-25 collapse.

**bf16-mode behaviour.** The curriculum, `raw_kl` tracking, and EMA
best-pointer all run unconditionally (no quant-mode gate). When the
trainer is invoked in `bf16` mode no Native backend is installed and
the `post_optim_step_callbacks` list is empty — the cache-invalidation
hook simply has nothing to invalidate. The temperature ramp is still
useful in BF16-only KD (it's the original max_quality stage-5 use
case) and the best-pointer is still useful for rescuing any KD run.

LR schedule is **linear warmup → cosine decay** to `min_learning_rate`.
With `total_tokens=50M`, `per_device_batch_size=1`, `seq_length=4096`,
`gradient_accumulation=32` the run is 381 optimizer steps long.

Calibration data is **`nvidia/Nemotron-Cascade-2-SFT-Data`** sampled at
13,000 sequences × 4096 tokens with the subset weights below (intent:
match the mixture our deployment runtime will see — heavy on chat,
some math/science, light on terminal/SWE):

```yaml
subset_weights: { math: 0.21, science: 0.11, chat: 0.56,
                  instruction_following: 0.033, conversational_agent: 0.0331,
                  swe: 0.02, terminal_agent: 0.0331 }
ptq_subset_size: 256       # first 256 sequences feed modelopt's amax calibration
```

## 5. Why these specific numbers — the run-1 post-mortem

Run-1 used the inherited Minitron / BF16 preset (`lr=2e-4`, `warmup=20`,
$\beta_2=0.999$). Loss curve was:

| step | lr        | loss   |
|------|-----------|--------|
| 1    | 1.0e-5    | 3.84   |
| 5    | 5.0e-5    | 1.40   |
| 10   | 1.0e-4    | 0.64   |
| 15   | 1.5e-4    | 2.99   |
| 20   | **2.0e-4 peak** | 1.22   |
| 25   | 1.999e-4  | **47.20** ⚠ |
| 30   | 1.997e-4  | 49.21  |
| 35   | 1.993e-4  | 57.04  |
| 50   | (eval)    | wikitext2_ppl = 4.5 × 10²⁴ |

**Root cause.** At 2.3 bpw, IQ2_XS represents every weight as one of
$\approx$5 codebook values per sub-block. The gap between adjacent
codebook entries is *smaller* than the per-step weight update at
$\eta = 2 \times 10^{-4}$. The STE returns identity gradient, so the
optimizer never sees that its updates are flipping codebook assignments
on the forward pass; once peak LR was reached, hundreds of expert
weights re-quantised to a different codebook entry every step. The
model never found a basin — each step landed in a fresh
fake-quant configuration.

Diagnosis confirmed by the loss trajectory: smooth descent during
sub-peak warmup (step 1 → 10), inflection at step 15 (lr crossing
$\sim$1.5e-4 where the flip rate becomes non-trivial), full divergence
the step *after* peak (step 25), and plateau at high loss because the
next step keeps re-flipping a different subset.

**Aggravators.**

1. **Self-distillation** — the teacher is the *un-quantised* student,
   so the KD target sits inside the post-PTQ student's basin. Early
   loss is therefore healthy (3.84 → 0.64 by step 10); peak LR
   dominates the trajectory because the loss surface near the PTQ init
   is shallow.
2. **$\beta_2 = 0.999$** — AdamW's second-moment EMA has a $\sim$1000-
   step time constant; under a 20-step warmup it hasn't equilibrated
   when peak LR hits, *inflating* per-parameter step sizes.
3. **`last_real_loss` = last-micro-of-window loss** — the step-15
   logged value of 2.99 was a precursor spike, not noise (heartbeats
   captured the per-micro detail post-instrumentation).

**Revisions** (all distillation-block, quant spec untouched):

| field              | old      | new      | why |
|---|---|---|---|
| `learning_rate`    | 2.0e-4   | **5.0e-5**   | 4× drop into the standard Q-LoRA / QAT range for ≤3-bit quant |
| `min_learning_rate`| 4.5e-7   | 5.0e-6   | ~10× cosine ratio (was ~110×) preserves tail learning signal |
| `betas[1]`         | 0.999    | **0.95** | $\sim$20-step time constant matches warmup; MoE QAT precedent |
| `grad_clip_norm`   | 1.0      | 0.5      | precursor spike at step 15; cheap insurance |
| `warmup_steps`     | 20       | **60**   | 5% → 16% of run; AdamW second moment has time to settle |
| `temperature`      | 1.0 (constant) | 1.0 (endpoint) + `temperature_start: 4.0` (ramp) | soft-to-hard curriculum; high T early smooths the teacher distribution during the codebook-flip-sensitive phase |
| `save_every_n_steps` | 50     | 25       | tighter rollback granularity given uncertainty |

Trade-off: the 50 M-token budget will under-fit relative to BF16
expectations at this lower LR. Run-2 is the conservative checkpoint;
if wikitext2 PPL is sane but elevated, future iterations can raise LR
selectively or extend `total_tokens`.

## 6. Operational tooling

### 6.1 Observability

- **Per-micro heartbeats** (`KDR_MICRO_HEARTBEAT=N`, default 4): every
  N micros the trainer logs `step / micro_in_window / loss / dt`. Used
  to detect stalled forwards inside a single window.
- **JIT compile monitor** (`cli/train.py::_install_jit_monitor`): wraps
  `torch.utils.cpp_extension.load{,_inline}` to print
  `jit load START / DONE name=… in N.Ns` for every CUDA-extension
  compile. Surfaces the ~20 s first-time build of
  `modelopt_cuda_ext`.
- **Persistent JIT cache** (`TORCH_EXTENSIONS_DIR=$CACHE_MOUNT/torch_ext`):
  modelopt's CUDA kernel survives container kills within an instance,
  saving the 20 s rebuild on relaunch.
- **NaN circuit breaker** (`loop._all_finite`): five consecutive
  windows with a non-finite loss raise; sub-threshold NaN micros are
  zero'd and skipped per the original recipe.

### 6.2 Spot-resilient saves

Every `save_every_n_steps` (now 25), the trainer writes a partial to
`pirola/kdr-partials-<run_id>` on the Hub. `run_id` is a deterministic
hash of `(config, student SHA, mode)` (LLR-0031), so a preempt → fresh
launch with the same YAML resumes from the latest partial via
`bootstrap.sh`'s partials-query path (LLR-0033). The final artifact
lands at `pirola/kdr-recovered-<run_id>`.

The safetensors save path applies a **dedupe-shared-storage** pass
(`io/save.py::_dedupe_shared_storage`) before
`unwrapped.save_pretrained`. This is needed because
`nn.utils.parametrize` relocates `lm_head.weight` to
`lm_head.parametrizations.weight.original`, breaking transformers'
standard tied-weights detection between `lm_head` ↔ `embed_tokens`.
Dedupe walks the state_dict by storage identity `(data_ptr, offset,
numel, dtype)` and keeps the canonical (non-parametrize) name on
collision. On resume the parametrize machinery is re-installed by
`NativeBackend.apply_quant`, which re-materialises the dropped key
from the kept canonical tensor.

## 7. References

| Topic | File |
|---|---|
| HLR-0016 — Profile-J `da_qad` GGUF recovery YAML intent | `requirements/hlr/HLR-0016.md` |
| LLR-0001 / 0002 / 0003 — FKLD loss | `requirements/llr/LLR-000{1,2,3}.md` |
| LLR-0015 — STE simulator design | `requirements/llr/LLR-0015.md` |
| LLR-0025 — Router replay | `requirements/llr/LLR-0025.md` |
| LLR-0031 / 0033 — Run-ID and partials resume | `requirements/llr/LLR-00{31,33}.md` |
| LLR-0053 — GGUF codebook constants | `requirements/llr/LLR-0053.md` |
| LLR-0061 / 0063 — IQ2_XS argmin tile + codebook unpack | `requirements/llr/LLR-006{1,3}.md` |
| Profile-J YAML | `configs/zaya1_8b_da_qad_profileJ_gguf.yaml` |
| STE simulators | `src/kdr/quant/native_backend/ste_simulators.py` |
| Caching parametrisation | `src/kdr/quant/native_backend/backend.py::_CachingSTEParametrization` |
| FKLD loss | `src/kdr/kd_loss.py` |
| Trainer loop | `src/kdr/training/loop.py` |
| Save path + dedupe | `src/kdr/io/save.py` |

External: Hinton et al. (2015) for the $T^2$-scaled KD loss;
ggml-org/llama.cpp for GGUF codebook semantics; the modelopt
`LogitsDistillationLoss` we mirror for the canonical reduction.
