# `vllm_calibration_hooks.patch` + `vllm_calibration_stage2_profile.patch` — manifest

Single source of truth for the patches' identity. Other places that reference
either patch (the HF Jobs build script, the README uploaded to
`pirola/vllm-patched-calib`) should match this file.

The wheel produced by the build script applies **both** patches in this
table. They are sibling new-file patches (independent of each other on
the apply graph; the hooks patch creates 10 modules, the stage2_profile
patch creates 1), but the second patch's `_layer_in_handler` is the
required receiver for the first patch's `layer_in` dispatch site — so
either one alone produces a partially-wired wheel.

## Current

| Field | Value |
|---|---|
| Immutable tag | `calib-v2-layer-input-reservoir` |
| Branch (active) | `main` |
| vLLM upstream SHA | `ad7125a43e176d4161099480a66f0169609a690` (v0.21.0) |
| Patch 1 line count | **10920** |
| Patch 1 MD5 | **`af1c38a2686c74012fc0f86b5449f23c`** (vllm_calibration_hooks.patch) |
| Patch 2 line count | **802** |
| Patch 2 MD5 | **`2a2012457ef4a45ca36757b33a3c4e15`** (vllm_calibration_stage2_profile.patch) |
| HF model repo | `pirola/vllm-patched-calib` |
| Wheel filename pattern | `vllm-0.21.1.dev0+gad7125a43.d<YYYYMMDD>-cp312-cp312-linux_x86_64.whl` |
| Torch / CUDA pinned in build | `torch==2.11.0+cu130` |
| `TORCH_CUDA_ARCH_LIST` | `8.0;9.0a;10.0;12.0` (A100 / H100·H200 / B200 / RTX 6000 Pro) |

## Verifying locally

```bash
md5sum max_quality/patches/vllm_calibration_hooks.patch
# expect: af1c38a2686c74012fc0f86b5449f23c
wc -l max_quality/patches/vllm_calibration_hooks.patch
# expect: 10920

md5sum max_quality/patches/vllm_calibration_stage2_profile.patch
# expect: 2a2012457ef4a45ca36757b33a3c4e15
wc -l max_quality/patches/vllm_calibration_stage2_profile.patch
# expect: 802

# Re-apply both against a fresh v0.21.0 checkout (idempotency check):
git clone --depth 1 --branch v0.21.0 https://github.com/vllm-project/vllm /tmp/vllm-fresh
cd /tmp/vllm-fresh
git apply --check /path/to/vllm_calibration_hooks.patch && echo OK
git apply /path/to/vllm_calibration_hooks.patch
git apply --check /path/to/vllm_calibration_stage2_profile.patch && echo OK
```

## Change log

### `calib-v2-wanda-scalar-row` (previous)

W-1 (audit `tasks/AUDIT_CALIBRATION_COMPLETENESS_V2.md` §W-1, plan
`tasks/PLAN_W1_WANDA_SCALAR_ROW_CAPTURE.md`). Promotes the routing-
weighted Wanda intra-expert ``scalar_row`` signal to a cross-run
calibration sidecar so the A0..A11 ablation grid skips the per-row
~2× Stage-3 calibration sweep that the in-process plugin previously
required. Expected saving: ~60 min per 12-row sweep, per
(teacher × calibration mix) combo.

- `vllm/calibration_wanda_scalar_row.py`: new module that subscribes
  the ``router`` and ``expert_in`` calibration_hooks callbacks; stashes
  per-layer ``topk_weights`` on the router fire (overwrite-on-fire,
  NO explicit clear — mirrors `vllm.calibration_reap_scores` at
  patch:8134-8473) and scatter-accumulates per-(layer, expert,
  "gate_proj") sum-of-squares of ``(x_t * g_{e,t})`` per input
  channel from the ``expert_in`` hook. ``dump_wanda_scalar_row``
  divides by token counts to produce the running mean
  ``E_t[(x_t * g_{e,t})^2]`` and writes the sidecar payload via
  `moe_compress.utils.cached_calibration_signals.save_wanda_scalar_row`
  (schema v1, Pattern B + Pattern O — atomic write + manifest-last).
  ``dump_wanda_scalar_row_checkpoint`` / ``load_wanda_scalar_row_checkpoint``
  store the RAW sum-of-squares (NOT the normalised mean) so a
  kill-mid-calibration + resume can additively combine new chunks
  with the saved sums (additivity invariant tested in T6 of the
  plan).
- ``down_proj`` is NOT captured by this writer (the ``expert_in``
  hook fires on the pre-routing hidden state; ``down_proj`` input is
  the post-act intermediate). The plugin's cache-MISS fallback
  retains the in-process calibration sweep that captures both
  matrices; cache HIT for ``gate_proj`` alone is sufficient for
  ``gate_proj`` + ``up_proj`` scoring (D-gate-up-share). On cache
  HIT, ``down_proj`` scoring is skipped (matches the consumer's
  existing "no scalar_row → skip" path at
  ``stage3/plugins/wanda_intra_expert_score.py:_compute_scores``).
- Env-var gating: ``VLLM_CALIB_CAPTURE_WANDA_SCALAR_ROW=1`` is sampled
  at import. Requires ``VLLM_CALIB_CAPTURE_ROUTER=1`` +
  ``VLLM_CALIB_CAPTURE_EXPERT=1`` (so the ``expert_in`` callback
  dispatches; NO FlashInfer requirement -- ``expert_in`` fires on
  every MoE backend).

Driver-side companion (``build_self_traces_calib_vllm.py``): new
``--capture-wanda-scalar-row`` + ``--wanda-scalar-row-checkpoint-every-chunks``
flags + 3 insertion points (argparse, env-gate block, setup +
checkpoint + final-dump block).

Consumer-side companion: ``Stage3WandaScalarRowCacheProvider``
(``max_quality/src/moe_compress/stage3/plugins/wanda_scalar_row_cache.py``)
registered in the Stage 3 ``PluginRegistry`` alongside
``Stage3InputCovCacheProvider``. On cache HIT, populates
``ctx["stage3.wanda_scalar_row"]`` so
``WandaIntraExpertScorePlugin.collect_wanda_scores`` short-circuits
its per-layer calibration sweep entirely via
``_WandaScalarRowAccumulator.from_payload``. On cache MISS or when
the operator opts out, the existing in-process calibration sweep
runs unchanged (cache MISS = unchanged behaviour).

Schema bump: ``SCHEMA_VERSIONS["wanda_scalar_row"]`` introduced at 1
(new green-field signal). Manifest-LAST is REQUIRED at load time —
missing manifest raises ``ManifestMismatchError`` (no legacy
artifacts exist on disk, so the bare-``torch.load`` fallback would
be dead code; L-1 plan-reviewer-v1 fold).

TODO (separate branch, NOT W-1 scope): the inline comment at
`vllm_calibration_hooks.patch:8218` in
``vllm/calibration_reap_scores.py`` claims
``_ROUTER_WEIGHTS_STASH: Cleared after each expert_out_unweighted
use``, but the code does NOT clear; it just overwrites. Replace
with ``Overwritten on next router fire; not explicitly cleared
(per-layer dispatch order guarantees correctness)``.

### `calib-v2-stage2-profile-writer` (in-flight)

Adds the Plugin #12 REDO -- Optimization A profile-pass sidecar writer
(SC_FAST_PLAN_V3 section 4). Lives in its own patch file
``vllm_calibration_stage2_profile.patch`` (separate from the main
``vllm_calibration_hooks.patch`` until the wheel rebuild folds it in).

- `vllm/calibration_stage2_profile.py`: new module that wraps the
  canonical writer logic from
  ``moe_compress.calibration.stage2_profile_writer``. Subscribes the
  ``router`` and ``expert_out_unweighted`` calibration_hooks callbacks
  to populate a live :class:`ReamCostAccumulator` (so the Eq. 8
  numerator is computed via the SAME ``finalize_batch`` code path the
  live profiling uses -- Bug #1 fix: per-token jointly-active pair
  cosines, NOT cos(mean_i, mean_j)) and an
  :class:`InputCovarianceAccumulator` with ``storage_dtype`` pinned
  IMMEDIATELY by ``setup(llm, cov_storage_dtype=...)`` (Bug avoidance:
  no reliance on the default fp32 at ``activation_hooks.py:961``).
  ``dump_stage2_profile`` finalizes pending cov entries layer-by-layer,
  asserts the live ``storage_dtype`` still matches the configured
  value, translates layer_idx-keyed dicts to layer_rank-keyed payload
  dicts, and writes the v3 sidecar via
  ``moe_compress.utils.cached_calibration_signals.save_stage2_profile_v3``.
  ``dump_stage2_profile_checkpoint`` / ``load_stage2_profile_checkpoint``
  mirror the imatrix / REAP / input-cov resumability cadence (atomic
  tmp+rename, schema-versioned, CPU-resident).
- Env-var gating: ``VLLM_CALIB_CAPTURE_STAGE2_PROFILE=1`` is sampled
  at import. Requires ``VLLM_CALIB_CAPTURE_ROUTER=1`` +
  ``VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1`` +
  ``VLLM_USE_FLASHINFER_MOE_FP16=0`` (FlashInfer monolithic path lacks
  ``expert_out_unweighted``).

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-stage2-profile``, ``--stage2-profile-cov-storage-dtype``
(choices: float16 / bfloat16 / float32, default float16),
``--stage2-profile-checkpoint-every-chunks`` (default 1). Five
insertion points: argparse (before --capture-per-expert-max), env-gate
block (after routing-stats env block), setup block (after input-cov
setup), periodic checkpoint (in the chunk loop after input-cov
checkpoint), final dump (after input-cov dump).

Stage 2 reader-side companion: ``Stage2ProfileCacheProvider``
(``max_quality/src/moe_compress/stage2/plugins/stage2_profile_cache.py``)
registered in the Stage 2 ``PluginRegistry`` AFTER ``LayerMergePlugin``
(OQ-1 Option A -- in-place hydration of the fresh ream_acc /
layer_input_acc that ``LayerMergePlugin.on_layer_setup`` constructs).
Gated on ``stage2_reap_ream.profile_sidecar.enabled`` (default false).
On full hit: hydrates ream_acc + cov_acc + layer_input_acc so
``LayerMergePlugin.on_profile`` early-returns (Pattern A skip). On
partial / miss: no-op; live forward runs unchanged.

Cross-validation at load time: schema_version, cov_storage_dtype
(against the run's ``s2.covariance_storage_dtype``), n_layers,
n_experts, top_k, model_hash -- any mismatch raises ``ValueError``
with the standard "Delete the sidecar to regenerate" message.

Schema bump: ``SCHEMA_VERSIONS["stage2_profile"]`` bumped from 1 to 3
(skips 2 to signal a clean break from the deleted prior Plugin #12
v1 writer). The v1 dataclass was never persisted by a production
writer; the v1 -> v3 bump is intentionally NOT forward-compatible.
Pattern K applies forward (v3 -> v4 SHOULD preserve readers when
adding optional fields).

### `calib-v2-layer-input-reservoir` (current)

CRITICAL-1 follow-up to ``calib-v2-max-layer-early-exit``. Adds the
``layer_in`` hook + per-layer reservoir capture wiring so the canonical
``stage2_profile_writer`` can populate ``layer_input_reservoir`` during
vLLM-backed calibration runs (previously only the in-process profile
pass populated it; vLLM calibrations were silently writing empty
reservoirs).

- `vllm/calibration_hooks.py`: registers the new ``layer_in`` callback
  name in the ``_CALLBACK_NAMES`` set and dispatches it from the
  ``Qwen3MoeSparseMoeBlock.forward`` pre-MoE entrypoint. The hook
  fires once per (layer_idx, hidden) pair, before routing.
- `vllm/model_executor/models/qwen3_moe.py`: hooks the new dispatch
  into the MoE block forward with the existing capture gate
  (no new env-var; the hook is dormant unless a writer subscribes).
- Driver wiring lives in
  ``max_quality/src/moe_compress/calibration/stage2_profile_writer.py``
  (canonical writer's ``_on_layer_in_callback``) + the patch copy in
  ``max_quality/patches/vllm_calibration_stage2_profile.patch``. The
  callback constructs a per-layer
  ``moe_compress.stage2.profiling._LayerInputAccumulator`` lazily on
  first sighting (seed = ``layer_idx`` for per-layer determinism, same
  contract as the in-process profile pass at ``profiling.py:86``) and
  feeds each batch into the vectorised Vitter Algorithm R reservoir.
- Driver flag: ``--capture-layer-input-reservoir`` (default ON; turn
  OFF only for the legacy empty-reservoir behaviour).
- MEDIUM-fix follow-up (this tag): the layer-input-reservoir
  checkpoint payload now serialises the per-layer
  ``torch.Generator.get_state()`` tensor (uint8[5056] on CPU) so the
  resumed RNG stream is byte-identical to the no-resume reference
  EVEN AFTER Phase C entry. The pre-fix 3-tuple payload
  ``(buffer, seen, max_samples)`` re-seeded the generator on load, so
  Phase-C-active checkpoints diverged from the reference (the
  reservoir's RNG draws on tokens 101..200 would be the same draws the
  pre-checkpoint half had already consumed). The loader emits a WARN
  + seed-re-init fallback for pre-MEDIUM-fix checkpoints (byte-
  identical resume preserved only if Phase C was never entered pre-
  checkpoint). New test
  ``test_checkpoint_resume_byte_identical_after_phase_c`` forces Phase
  C entry pre-checkpoint via cap=64 + multi-chunk feed and asserts
  buffer + generator state byte-equality against the no-resume
  reference.

### `calib-v2-max-layer-early-exit` (previous)
Adds the L2 ``max_layer`` early-exit gate to ``Qwen3MoeModel.forward``
(Item L2 of the calibration-v2 writers campaign — foundation for L1's
sequential REAP+REAM per-layer profiling and a standalone optimisation
for any writer whose payload comes from layer ``L`` or earlier).
- `vllm/calibration_hooks.py`: new module-level ``_CALIB_MAX_LAYER:
  int | None`` (sampled at import from ``VLLM_CALIB_MAX_LAYER`` via
  ``_parse_max_layer_env``; ``""`` and ``"-1"`` map to ``None``;
  malformed values map to ``None`` rather than raising), new public
  ``set_calibration_max_layer(layer: int | None)`` /
  ``get_calibration_max_layer()`` runtime accessors (the setter
  normalises ``None`` and negative ints to the disabled sentinel and
  rejects non-int with ``TypeError``), and a new docstring section
  describing how L2 composes with the per-hook capture gates. Both
  new functions are exported in ``__all__``.
- `vllm/model_executor/models/qwen3_moe.py`: ``Qwen3MoeModel.forward``
  reads ``_ch._CALIB_MAX_LAYER`` ONCE at forward entry and uses the
  value to derive ``effective_end = min(self.end_layer, max(self.
  start_layer, _max_layer + 1))`` for the ``islice`` over decoder
  layers (so the boundary is INCLUSIVE: ``_CALIB_MAX_LAYER=N`` runs
  layers ``start_layer..N`` and skips ``N+1..end_layer-1``; out-of-
  range values clamp to the model boundary; values below the shard's
  ``start_layer`` collapse to zero layers on that shard, which is the
  correct pipeline-parallel semantics). When ``_CALIB_MAX_LAYER is
  None`` (default), ``effective_end`` == ``self.end_layer`` => the
  loop is byte-identical to the un-patched code path so the
  production hot path pays nothing. The single attribute load at
  forward entry keeps ``@support_torch_compile`` specialisation
  clean: a value change forces a recompile, by design.
- `vllm/envs.py`: ``VLLM_CALIB_MAX_LAYER: int = -1`` added to the
  ``TYPE_CHECKING`` block and the ``environment_variables`` dispatch
  dict (``lambda: int(os.getenv("VLLM_CALIB_MAX_LAYER", "-1"))``).
  Default ``-1`` matches the ``_parse_max_layer_env`` "disabled"
  sentinel.
- `tests/test_calibration_max_layer.py`: +8 tests covering the
  default-None contract, env-var sampling at import for ``0/5/23``,
  the disabled-sentinel mapping for ``""/-1/-7``, malformed-env
  fallback (``"foo"/"1.5"/"0x5"/" "`` -> ``None``), runtime setter
  round-trip via the public accessors, setter-clears with ``None`` /
  negative-int, setter-rejects with ``TypeError`` for
  ``str/float/list/tuple/object``, and the ``effective_end`` math
  spec (single-rank ``start=0, end=24`` + pipeline-parallel shard
  ``start=12, end=24``, including the below-shard-start collapse).

NO new driver-side flag in this patch: the calibration driver
(``build_self_traces_calib_vllm.py``) can opt in by either exporting
``VLLM_CALIB_MAX_LAYER=<N>`` BEFORE the ``from vllm import LLM``
import or by calling ``vllm.calibration_hooks.set_calibration_max_
layer(N)`` AFTER the import but BEFORE ``LLM.generate`` -- L1 will
wire this in the per-layer loop.

NO new Stage 3 / Stage 2.5 reader companion: L2 is an OPTIMISATION
of the existing forward; the writers that benefit (block-outputs,
imatrix, input-cov, REAP-scores, per-expert-max, output-reservoir,
routing-stats, router-logits-stats) already write the same on-disk
sidecars regardless of how many layers were actually run. Operators
must understand that setting ``VLLM_CALIB_MAX_LAYER=N`` together with
a layer-spanning writer (e.g. block-outputs over all layers) will
produce sidecars containing data ONLY for layers ``0..N`` -- the
downstream consumers handle "missing layer" the same way they handle
any other absent sidecar entry.

### `calib-v2-block-outputs-writer`
Adds the per-MoE-block hidden-states writer on a fixed N-prompt subset
(Item 7 of the calibration-v2 writers campaign).
- `vllm/calibration_block_outputs.py`: new module subscribing the
  existing ``block_out`` hook (dispatched from
  ``Qwen3MoeSparseMoeBlock.forward`` under
  ``VLLM_CALIB_CAPTURE_BLOCK=1``) to clone the post-MoE-block hidden
  states (pre-residual-add) into per-rank CPU bf16 accumulators. The
  distinguishing trait of this writer is the **subset-close gate**: the
  driver owns the cumulative prompt counter and calls
  ``close_subset()`` once the count reaches
  ``VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE`` (default 128); subsequent
  ``block_out`` dispatches early-return. ``dump_block_outputs`` writes
  ONE per-layer sidecar at
  ``<jsonl_parent>/sidecars/block_hidden/layer_{idx:04d}.pt`` (schema v1
  ``BlockHiddenPayload`` -- already declared by Item 0, no schema bump
  needed) by concatenating each rank's per-batch tensors along dim 0
  into a single ``[n_tokens, hidden]`` bf16 slab.
  ``dump_block_outputs_checkpoint`` /
  ``load_block_outputs_checkpoint`` mirror the
  imatrix / REAP / input-cov / per-expert-max / routing-stats /
  router-logits-stats / output-reservoir resumability cadence (atomic
  tmp+rename, schema-versioned, CPU-resident accumulators ->
  CUDA-graph-safe). Subset size is sampled at module import from
  ``VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE`` so the subset-close contract
  stays consistent across a resume; a mismatch on load raises
  ``ValueError``. The subset-closed flag itself is also serialized in
  the checkpoint so a resumed run that already closed the subset stays
  a no-op for the remaining JSONL. NO FlashInfer restriction: the
  ``block_out`` hook is dispatched from the model-level forward, not
  from a kernel path, so this writer works on any MoE backend (parallel
  with routing-stats / router-logits-stats).
- `vllm/envs.py`: add ``VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS`` +
  ``VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE`` to the ``TYPE_CHECKING``
  block and the ``environment_variables`` dispatch dict.
- `tests/test_calibration_block_outputs_smoke.py`: +6 tests covering
  the env-off short-circuit, the subset-close gate (idempotent +
  no-ops subsequent dispatches), multi-layer rank ordering with
  layer-ids dispatched out of order, payload shape + per-batch
  concatenation along dim 0 with the bf16 dtype contract,
  checkpoint round-trip including the subset-closed flag + hidden_dim
  + cumulative prompt counter, and the per-layer sidecar dump path
  (one ``save_block_hidden`` call per populated rank).

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-block-outputs`` + ``--block-outputs-subset-size`` (default
128) + ``--block-outputs-checkpoint-every-chunks`` (default 1),
parallel env-gating block (``VLLM_CALIB_CAPTURE_BLOCK_OUTPUTS=1``,
``VLLM_CALIB_CAPTURE_BLOCK=1``,
``VLLM_CALIB_BLOCK_OUTPUTS_SUBSET_SIZE=<value>``), post-
``_load_teacher_vllm`` setup + resume hydration, periodic in-loop
checkpoint, in-loop subset-close at the JSONL flush boundary as soon
as ``already_done + n_new >= args.block_outputs_subset_size``, and
final dump (belt-and-braces ``close_subset()`` call right before the
dump to lock the ``n_prompts_in_subset`` field on the payload).
``bo_ckpt_path`` is hoisted out of the per-feature ``if`` block (same
N1 pattern as Item 1 / Item 6).

Stage 3 reader-side companion:
``Stage3BlockHiddenCacheProvider``
(``max_quality/src/moe_compress/stage3/plugins/block_hidden_cache.py``)
registered in the Stage 3 orchestrator's plugin list and consulted via
``PluginRegistry.dispatch_first("on_load", ...)`` right BEFORE the
``walk_phases(("refine_blocks",), ...)`` call (gated on
``stage3_svd.block_refine.enabled``). On hit the provider walks
``sidecars/block_hidden/``, loads every ``layer_{idx:04d}.pt``,
reshapes each ``[n_tokens, hidden]`` slab into
``[n_prompts, seq_len, hidden]`` (inferring seq_len from ``ctx.calib``),
and chunks into per-batch ``[batch_size, seq_len, hidden]`` bf16 CPU
tensors (matching the live ``teacher_targets`` shape contract in
``_phase_c5_block_refine``). The cache is populated on
``ctx.teacher_targets_cache: dict[int, list[Tensor]]`` and threaded
through ``BlockRefinePlugin.refine_blocks`` into the
``_phase_c5_block_refine`` kwarg. On a layer-keyed hit with a
shape-match the live teacher block forward is SKIPPED; on miss
(missing sidecars dir, token-count alignment failure, batch-count
mismatch, or per-batch shape mismatch) block_refine falls through to
the unchanged live teacher forward with an actionable warning.

Stage 2.5 `merge_repair` cache reader: **SCOPE-CUT**. The router_kd
trainer drives `_LayerOutputCapture` via its own per-batch dataloader
(not the calibration JSONL), and the token-map alignment between the
flat Item-7 sidecar and the trainer's `input_ids` would require all
three of: (a) the trainer to switch dataset to the calibration JSONL,
(b) byte-identical chat-template rendering, (c) deterministic batch
shuffle order. The Stage 3 reader alone captures the high-cost win
(skipping the live teacher block forward inside Phase C.5). See
`max_quality/docs/calibration_v2_data_capture_plan.md` "Scope-cut
consumer notes" for the reversibility plan.

Schema bump: NONE -- ``BlockHiddenPayload`` was already declared with
``SCHEMA_VERSIONS["block_hidden"] = 1`` by Item 0.

### `calib-v2-output-reservoir-writer` (previous)
Adds the per-(layer, expert) expert-output reservoir writer
(Item 6 of the calibration-v2 writers campaign).
- `vllm/calibration_output_reservoir.py`: new module subscribing the
  existing ``expert_out_unweighted`` hook (chained alongside REAP-
  scores and per-expert-max via the multi-callback registry) to
  reservoir-sample per-(layer, expert) unweighted expert outputs into
  CPU fp32 reservoirs (lazy-allocated per cell, lazily because typical
  sparse routing only touches a fraction of (layer, expert) pairs per
  dispatch). The two-phase fill+sample math replicates
  :meth:`ExpertOutputAccumulator.update` byte-for-byte: Phase 1 fills
  empty slots sequentially while ``seen < max_tokens``; Phase 2
  accepts each further token with probability ``max_tokens / (seen +
  n_fill + j)`` and on accept writes to a uniformly-random slot
  (last-wins on collision). The persistent-buffer clone guard mirrors
  per-expert-max. ``dump_output_reservoir`` stacks all per-(rank, e)
  reservoirs into a dense ``[n_layers, n_experts, max_tokens, hidden]``
  bf16 tensor (zero-padded for unfilled cells) and writes the
  ``OutputReservoirPayload`` sidecar (schema v1, includes per-(layer,
  expert) ``valid_count`` and ``total_seen`` int64 + the ``max_tokens``
  scalar) via
  ``moe_compress.utils.cached_calibration_signals.save_output_reservoir``.
  ``dump_output_reservoir_checkpoint`` /
  ``load_output_reservoir_checkpoint`` mirror the imatrix / REAP /
  input-cov / per-expert-max / routing-stats / router-logits-stats
  resumability cadence (atomic tmp+rename, schema-versioned,
  CPU-resident reservoirs -> CUDA-graph-safe). Cap is sampled at
  module import from ``VLLM_CALIB_OUTPUT_RESERVOIR_CAP`` (default 256)
  so the writer's two-phase math is consistent across a resume; a
  mismatch on load raises ``ValueError``. FlashInfer monolithic path
  is disabled (no ``expert_out_unweighted`` available there).
- `vllm/envs.py`: add ``VLLM_CALIB_CAPTURE_OUTPUT_RESERVOIR`` +
  ``VLLM_CALIB_OUTPUT_RESERVOIR_CAP`` to the ``TYPE_CHECKING`` block
  and the ``environment_variables`` dispatch dict for discoverability
  through ``vllm.envs``.
- `tests/test_calibration_output_reservoir_smoke.py`: +6 tests covering
  the fill-only-regime byte-equality math (token rows preserved in
  routing order), env-off short-circuit (setup + callback both no-op),
  checkpoint round-trip (reservoirs + bookkeeping + cap survive a
  reimport), two-segment additivity in the fill-only regime (exact
  equality across a checkpoint+reload split), persistent-buffer clone
  safety (post-callback ``unweighted.fill_(...)`` does NOT corrupt
  stored reservoir rows), and the final-dump payload shape +
  ``valid_count`` correctness (rank-1 untouched cells stay zero-filled
  with valid_count=0; rank-0 populated cells preserve their slabs).

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-output-reservoir`` + ``--output-reservoir-cap`` (default
256) + ``--output-reservoir-checkpoint-every-chunks`` (default 1),
parallel env-gating block (``VLLM_CALIB_CAPTURE_OUTPUT_RESERVOIR=1``,
``VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1``,
``VLLM_USE_FLASHINFER_MOE_FP16=0``, ``VLLM_CALIB_OUTPUT_RESERVOIR_CAP=<value>``),
post-``_load_teacher_vllm`` setup + resume hydration, periodic in-loop
checkpoint, and final dump that removes the now-stale ckpt.
``or_ckpt_path`` is hoisted out of the per-feature ``if`` block (same
N1 pattern as Item 1).

Stage 1 reader-side companion: ``Stage1OutputReservoirCacheProvider``
(``max_quality/src/moe_compress/stage1/plugins/output_reservoir_cache.py``)
consulted by the Stage 1 orchestrator's STEP 4.8 immediately AFTER
STEP 4.7 (Item 4's router_logits_stats attempt) with the same
try/except guard pattern. On hit the provider hydrates a pre-
finalized ``ExpertOutputAccumulator`` directly into ``ctx.output_acc``
(slicing each cell to ``valid_count`` per the writer's bookkeeping;
zero-valid-count cells are excluded to match the live absent-key
convention) AND the orchestrator drops ``"output_reservoir"`` from
``needed`` so the live Phase-B per-expert reservoir-sample HookSpec
is NOT registered. STEP 5's ``ctx.set("output_acc", ...)`` and STEP
7's ``built["output_reservoir"].finalize()`` are both guarded against
the cache-hit case (``if not _or_cache_hit`` and
``if "output_reservoir" in built`` respectively). Topology consistency
check raises ``ValueError`` if the sidecar's ``n_layers`` disagrees
with the live model.

Schema bump: ``SCHEMA_VERSIONS["output_reservoir"] = 1`` (new entry).

### `calib-v2-router-logits-stats-writer` (previous)
Adds the per-(layer, expert) sink-vs-normal router-score aggregate
writer (Item 4 of the calibration-v2 writers campaign).
- `vllm/calibration_router_logits_stats.py`: new module subscribing the
  existing ``router`` hook (chained alongside any other ``router``
  subscriber via the multi-callback registry) to softmax each per-token
  router-logits row inline, scatter-add per-(layer, expert) score sums
  partitioned by a sink/normal mask, and scatter-add the top-k ids of
  sink tokens into a per-expert ``fire_on_sink`` counter. Per-layer
  ``n_sink_tokens`` / ``n_normal_tokens`` scalars are tracked so the
  Stage 1 consumer can invert sums -> means. Sink-mask resolution:
  ``input_id == bos_token_id`` when the dispatch supplies both, else
  position-0-only fallback (today's vLLM router dispatch does NOT
  surface ``input_ids`` to the callback, so the fallback is the de
  facto active path; the writer is forward-compatible if the dispatch
  grows the kwarg). ``setup(llm, bos_token_id=None)`` extends the
  per-writer setup() signature to capture the BOS id. NO
  ``EXPERT_UNWEIGHTED`` / FlashInfer dependency.
  ``dump_router_logits_stats_checkpoint`` /
  ``load_router_logits_stats_checkpoint`` mirror the imatrix / REAP /
  input-cov / per-expert-max / routing-stats resumability cadence
  (atomic tmp+rename, schema-versioned, CPU-resident accumulators ->
  CUDA-graph-safe). Final dump writes the ``RouterLogitsStatsPayload``
  sidecar (schema v1, per-(layer, expert) ``score_sink_sum`` /
  ``score_normal_sum`` float32 + ``fire_on_sink`` int64 + per-layer
  ``n_sink_tokens`` / ``n_normal_tokens`` int64 + ``bos_token_id``) via
  ``moe_compress.utils.cached_calibration_signals.save_router_logits_stats``.
- `vllm/envs.py`: add ``VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS`` to the
  ``TYPE_CHECKING`` block + the ``environment_variables`` dispatch
  dict for discoverability through ``vllm.envs``.
- `tests/test_calibration_router_logits_stats_smoke.py`: +6 tests
  covering the env-off short-circuit, the BOS-id sink-mask branch
  (input_ids supplied), the position-0 fallback branch (input_ids
  omitted -- the de facto active path), additive accumulation across
  multiple callback invocations, checkpoint round-trip preserving
  bos_token_id (incl. the None branch), and the final-dump payload
  shape + per-(layer, expert) means derivation.

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-router-logits-stats`` +
``--router-logits-stats-checkpoint-every-chunks``, parallel env-gating
block (``VLLM_CALIB_CAPTURE_ROUTER_LOGITS_STATS=1`` +
``VLLM_CALIB_CAPTURE_ROUTER=1`` only -- NO FlashInfer or
EXPERT_UNWEIGHTED requirement), post-``_load_teacher_vllm`` setup
that passes ``bos_token_id=tokenizer.bos_token_id``, resume hydration,
periodic in-loop checkpoint, and final dump that removes the now-stale
ckpt. ``router_logits_ckpt_path`` is hoisted out of the per-feature
``if`` block (same N1 pattern as prior items).

Stage 1 reader-side companion: ``Stage1RouterLogitsStatsCacheProvider``
(``max_quality/src/moe_compress/stage1/plugins/router_logits_stats_cache.py``)
consulted by the Stage 1 orchestrator's STEP 4.7 immediately AFTER
STEP 4.6 (Item 3's routing_stats attempt) with the same try/except
guard pattern. On hit the provider hydrates a pre-finalized
``SinkTokenRoutingAccumulator`` directly into ``ctx.sink_acc``
(overwriting the live setup-built accumulator) AND the orchestrator
drops ``"sink_routing"`` from ``needed`` so the live router-logits +
softmax + top-k HookSpec is NOT registered. The R3 guard
(``sink_token_enabled=False``) is honored: the provider returns ``None``
without consulting the sidecar so the user's explicit disable is
preserved. Topology consistency check raises ``ValueError`` if the
sidecar's ``n_layers`` disagrees with the live model. STEP 7's
``finalize()`` guard (``if "sink_routing" in built:``) was already in
place from prior work and continues to short-circuit cleanly on cache
hit.

Schema bump: ``SCHEMA_VERSIONS["router_logits_stats"] = 1`` (new entry).

### `calib-v2-routing-stats-writer` (previous)
Adds the per-(layer, expert) routing-frequency + mean-routing-weight
writer (Item 3 of the calibration-v2 writers campaign).
- `vllm/calibration_routing_stats.py`: new module subscribing the
  existing ``router`` hook (chained alongside any other ``router``
  subscriber via the multi-callback registry) to scatter-add per-
  (layer, expert) token counts (``int64``) and routing-weight sums
  (``float32``). ``dump_routing_stats`` derives
  ``mean_weight = weight_sum / freq.clamp(min=1).float()`` so zero-
  traffic cells surface as 0.0 (no NaN) and writes the dense
  ``RoutingStatsPayload`` sidecar (schema v1, shape
  ``[n_layers, n_experts]`` int64 freq + float32 mean_weight) via
  ``moe_compress.utils.cached_calibration_signals.save_routing_stats``.
  ``dump_routing_stats_checkpoint`` / ``load_routing_stats_checkpoint``
  mirror the imatrix / REAP / input-cov / per-expert-max resumability
  cadence (atomic tmp+rename, schema-versioned, CPU-resident
  accumulators -> CUDA-graph-safe). NO ``EXPERT_UNWEIGHTED`` /
  FlashInfer dependency -- the ``router`` hook fires on every MoE
  backend, so the writer works alongside any other writer combination
  (or alone).
- `vllm/envs.py`: add ``VLLM_CALIB_CAPTURE_ROUTING_STATS`` to the
  ``TYPE_CHECKING`` block + the ``environment_variables`` dispatch
  dict for discoverability through ``vllm.envs``.
- `tests/test_calibration_routing_stats_smoke.py`: +6 tests covering
  the env-off short-circuit (setup + callback both no-op), freq
  accumulation correctness (per-token scatter-add), mean-weight
  normalization correctness via the dump path (weights 0.3+0.5 to
  expert 0 -> mean=0.4), checkpoint round-trip, two-segment additivity
  for the freq + weight-sum operations, and the gate-off setup-is-a-
  no-op contract (no callback registered against ``router``).

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-routing-stats`` + ``--routing-stats-checkpoint-every-chunks``,
parallel env-gating block (``VLLM_CALIB_CAPTURE_ROUTING_STATS=1`` +
``VLLM_CALIB_CAPTURE_ROUTER=1`` only -- NO FlashInfer or
EXPERT_UNWEIGHTED requirement), post-``_load_teacher_vllm`` setup +
resume hydration, periodic in-loop checkpoint, and final dump that
removes the now-stale ckpt. ``rts_ckpt_path`` is hoisted out of the
per-feature ``if`` block (same N1 pattern as Item 1).

Stage 1 reader-side companion: ``Stage1RoutingStatsCacheProvider``
(``max_quality/src/moe_compress/stage1/plugins/routing_stats_cache.py``)
consulted by the Stage 1 orchestrator's STEP 4.6 immediately AFTER
STEP 4.5 (Item 2's per_expert_max attempt) with the same try/except
guard pattern. On hit the cached payload is deposited on
``ctx.routing_stats_payload``; on miss the ctx is untouched. NO
``needed`` filter change -- there is no live downstream consumer to
skip (Item 3 lays infrastructure for future plugins). Topology
consistency check raises ``ValueError`` if the sidecar's ``n_layers``
disagrees with the live model.

Stage 2 reader-side companion: ``Stage2RoutingStatsCacheProvider``
(``max_quality/src/moe_compress/stage2/plugins/routing_stats_cache.py``)
registered in the Stage 2 ``PluginRegistry`` immediately AFTER
``Stage2ReapScoresCacheProvider``. Divergence from the spec text:
``PluginRegistry.dispatch_first`` is first-winner-takes-all, so if
REAP-cache hits the routing-stats provider's ``on_load`` is never
reached through that chain. The orchestrator therefore makes an
EXPLICIT follow-up call (``isinstance(_plug,
Stage2RoutingStatsCacheProvider)`` lookup -> ``on_load(...)``) so
routing-stats always gets a chance to populate ctx, regardless of
REAP-cache outcome.

Schema bump: ``SCHEMA_VERSIONS["routing_stats"] = 1`` (new entry).

### `calib-v2-per-expert-max-writer` (previous)
Adds the per-(layer, expert) ``down_proj`` output max-L_inf writer
(Item 2 of the calibration-v2 writers campaign).
- `vllm/calibration_per_expert_max.py`: new module subscribing the
  existing ``expert_out_unweighted`` hook (chained alongside REAP-
  scores' own subscriber via the multi-callback registry) to scatter-
  amax per-(layer, expert) ``|f_j(x)|_inf`` into a CPU fp32 accumulator
  initialized to ``-inf``. Token counts are scatter-added in parallel.
  ``dump_per_expert_max`` zero-fills the ``-inf`` sentinel for zero-
  traffic cells and writes the dense ``Stage1PerExpertMaxPayload``
  sidecar (schema v1, shape ``[n_layers, n_experts] float32`` +
  matching ``int64`` counts) via
  ``moe_compress.utils.cached_calibration_signals.save_per_expert_max``.
  ``dump_per_expert_max_checkpoint`` / ``load_per_expert_max_checkpoint``
  mirror the imatrix / REAP / input-cov resumability cadence (atomic
  tmp+rename, schema-versioned, CPU-resident accumulators -> CUDA-
  graph-safe).
- `vllm/envs.py`: add ``VLLM_CALIB_CAPTURE_PER_EXPERT_MAX`` to the
  ``TYPE_CHECKING`` block + the ``environment_variables`` dispatch dict
  for discoverability through ``vllm.envs``.
- `tests/test_calibration_per_expert_max_smoke.py`: +6 tests covering
  accumulator scatter-amax math, env-off short-circuit (setup + callback
  both no-op), checkpoint round-trip (with ``-inf`` cells preserved),
  two-segment additivity for the max operation, persistent-buffer clone
  safety (post-callback mutation of ``unweighted`` does NOT corrupt the
  accumulator), and the dump-payload shape + zero-fill contract (partial
  accumulator with -inf cells produces a clean ``[n_layers, n_experts]``
  payload with all -inf zero-filled).

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-per-expert-max`` + ``--per-expert-max-checkpoint-every-chunks``,
parallel env-gating block (``VLLM_CALIB_CAPTURE_PER_EXPERT_MAX=1``,
``VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1``, ``VLLM_USE_FLASHINFER_MOE_FP16=0``),
post-``_load_teacher_vllm`` setup + resume hydration, periodic in-loop
checkpoint, and final dump that removes the now-stale ckpt. ``pem_ckpt_path``
is hoisted out of the per-feature ``if`` block (same N1 pattern as Item 1).

Stage 1 reader-side companion: ``Stage1PerExpertMaxCacheProvider``
(``max_quality/src/moe_compress/stage1/plugins/per_expert_max_cache.py``)
consulted by the Stage 1 orchestrator's STEP 4.5 BEFORE accumulator
construction (STEP 5). On hit the cached payload's dense tensor is
unpacked into a ``DownProjMaxAccumulator.per_expert_max`` dict keyed
by ``(layer_idx, expert_id)`` (mapping rank -> layer_idx via the live
``MoELayerRef`` list) and the accumulator is set on ``ctx.max_acc``;
the orchestrator's ``needed`` filter drops ``downproj_max`` from the
live registrations so the Phase B forward skips max-magnitude
collection; the STEP 7 ``finalize`` is guarded by
``if "downproj_max" in built:`` so a cache hit doesn't re-enter the
live accumulator that was never constructed. Zero-traffic experts
(cached value exactly 0.0) are omitted from the dict to match the
live accumulator's absent-key convention.

Schema bump: ``SCHEMA_VERSIONS["per_expert_max"] = 1`` (new entry).

### `calib-v2-input-cov-writer-chained-callbacks` (previous)
Review-fix follow-up to `calib-v2-input-cov-writer`. The single-slot
callback registry in `vllm/calibration_hooks.py` silently overwrote any
previous subscriber when a second writer registered for the same hook
name -- imatrix's `_on_expert_in` and input-cov's `_on_expert_in` could
not coexist (whichever ran second won; the other's accumulators stayed
empty without warning).

- `vllm/calibration_hooks.py`: `_callbacks: dict[str, list[Callable]]`
  (was `dict[str, Callable | None]`). `register_callback(name, fn)` now
  APPENDS `fn` to the list (identity-based de-dup so the same callable
  registered twice fires once), or CLEARS the list when `fn is None`.
  `dispatch(name, ...)` iterates the list in registration order and
  invokes every callback on the same payload.
- `tests/test_calibration_hooks_smoke.py`: +3 tests --
  `test_chained_callbacks_coexist_on_expert_in` pins both writers
  receiving every dispatch; `test_register_callback_dedup_same_callable`
  guards against double-fire on duplicate `register_callback` calls
  (resume / reinit paths); `test_register_callback_none_clears_list`
  pins the clear-on-None contract.
- `vllm/calibration_input_cov.py`: extended docstring on `_on_expert_in`
  explaining the CPU-residency choice (CUDA-graph capture safety), and
  on `dump_input_cov`'s log message clarifying that the sidecar stores
  the SHARED gate+up input covariance under `'gate_proj'` (consumers
  alias up_proj -> gate_proj via `_cov_lookup`).

### `calib-v2-input-cov-writer` (previous)
Adds the per-(layer, expert, "gate_proj") teacher input-covariance Σ_in
writer (Item 1 of the calibration-v2 writers campaign).
- `vllm/calibration_input_cov.py`: new module hooking ``expert_in`` to
  scatter-reduce per-(layer, expert) ``x^T x`` into a CPU fp32
  accumulator dict keyed by ``(layer_idx, expert_idx, "gate_proj")`` --
  the EXACT shape used by the Stage 2 writer's
  ``_stage2_input_covariance.pt``. Final ``dump_input_cov`` writes the
  payload via
  ``moe_compress.utils.cached_calibration_signals.save_covariance``
  (schema bumped to v2 to reflect the dict-valued payload).
  ``dump_input_cov_checkpoint`` / ``load_input_cov_checkpoint`` mirror
  the reap-scores / imatrix resumability cadence (atomic tmp+rename,
  schema-versioned, CPU-resident accumulators -> CUDA-graph-safe).
- `vllm/envs.py`: add ``VLLM_CALIB_CAPTURE_INPUT_COV`` to the
  ``TYPE_CHECKING`` block + the ``environment_variables`` dispatch dict
  for discoverability through ``vllm.envs``.
- `tests/test_calibration_input_cov_smoke.py`: +6 tests covering
  accumulator math, env-off short-circuit, checkpoint round-trip,
  two-segment additivity, dump-payload shape, and lazy-allocation of
  layers not pre-discovered by setup().

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
``--capture-input-covariance`` + ``--input-cov-checkpoint-every-chunks``,
parallel env-gating block (``VLLM_CALIB_CAPTURE_INPUT_COV=1``,
``VLLM_CALIB_CAPTURE_EXPERT=1``), post-``_load_teacher_vllm`` setup +
resume hydration, periodic in-loop checkpoint, and final dump that
removes the now-stale ckpt.

Stage 3 reader-side companion: ``Stage3InputCovCacheProvider``
(``max_quality/src/moe_compress/stage3/plugins/input_cov_cache.py``)
consulted by the Stage 3 orchestrator's run-glue BEFORE the legacy
``_load_stage2_covariance`` call. On hit the cached payload's
``sigma_in`` dict drops into ``A_cov`` directly; on miss the legacy
path runs. The provider is also registered FIRST in the Stage 3
``PluginRegistry`` for introspection parity with the Stage 2 / Stage 4
cache providers.

Stage 4 reader-side companion: ``Stage4InputCovCacheProvider``
(``max_quality/src/moe_compress/stage4/plugins/input_cov_cache.py``)
registered as the FIRST plugin in ``PluginRegistry``, BEFORE
``EoraInputsPlugin``. Its ``on_load`` populates ``ctx.A_cov`` +
``ctx.a_storage_dtype`` from the sidecar; the orchestrator dispatches
``on_load`` before ``walk_phases("load_eora_inputs", ...)``.
``EoraInputsPlugin.load_eora_inputs`` now starts with a
``ctx.has("A_cov")`` short-circuit guard so the on-disk
``_stage2_input_covariance.pt`` load is skipped on a cache hit.

Schema bump: ``SCHEMA_VERSIONS["covariance"]`` bumped from 1 to 2.
``CovariancePayload`` switched from a single 4-D tensor field (v1) to
the dict-valued layout (v2) -- mandatory because the live consumers in
Stage 3/4 (``_cov_lookup``, ``_compute_eora_factors``) need per-(layer,
expert, matrix) keying that a 4-D tensor cannot express without a
separate index mapping. v1 was never written by any production writer;
this is a forward-only bump.

### `calib-v2-reap-scores-writer` (previous)
Adds the REAP-scores writer (V1+V2 of the calibration-v2 writers campaign).
- `vllm/calibration_reap_scores.py`: new module pairing the `router` +
  `expert_out_unweighted` source-patched hooks to accumulate per-(layer,
  expert) `g_j · ‖f_j‖₂` contributions and per-expert token counts. Final
  `dump_reap_scores` normalizes to `S_j = (1/|X_j|)·Σ g_j·‖f_j‖₂` (REAP
  Eq. 9, arXiv:2510.13999) and writes the sidecar via
  `moe_compress.utils.cached_calibration_signals.save_reap_scores`. Periodic
  `dump_reap_scores_checkpoint` mirrors the imatrix resumability cadence;
  `load_reap_scores_checkpoint` hydrates on `--resume`. Atomic-write
  contract: tmp + `os.replace`, schema-versioned, CPU-resident
  accumulators (CUDA-graph-safe).
- `tests/test_calibration_reap_scores_smoke.py`: +6 tests covering
  accumulator math, env-off short-circuit, checkpoint round-trip,
  two-segment additivity, router-stash-miss guard, and dump
  normalization.

Driver-side companion (`build_self_traces_calib_vllm.py`): new flags
`--capture-reap-scores` + `--reap-scores-checkpoint-every-chunks`,
parallel env-gating block (`VLLM_CALIB_CAPTURE_REAP_SCORES=1`,
`VLLM_CALIB_CAPTURE_ROUTER=1`, `VLLM_CALIB_CAPTURE_EXPERT_UNWEIGHTED=1`,
`VLLM_USE_FLASHINFER_MOE_FP16=0`), post-`_load_teacher_vllm` setup +
resume hydration, periodic in-loop checkpoint, and final dump that
removes the now-stale ckpt.

Stage 2 reader-side companion: `Stage2ReapScoresCacheProvider`
(`max_quality/src/moe_compress/stage2/plugins/reap_scores_cache.py`)
registered as the FIRST plugin in `PluginRegistry`, BEFORE
`ReapScoringPlugin`. Its `on_load` hydrates `ctx.reap_scores_payload`
from the sidecar; its `on_score` populates `ctx.scores` + `ctx.freq`
from the cached row. `ReapScoringPlugin.on_score` now starts with a
`ctx.has("scores")` early-return guard so the live REAP forward is
skipped on a cache hit.

### `calib-v2-imatrix-resumable` (previous)
Adds spot-preemption resumability for the imatrix capture path.
- `vllm/calibration_imatrix.py`: new public API
  - `dump_imatrix_checkpoint(path)` — atomic `.imatrix.ckpt` via tmp+rename.
  - `load_imatrix_checkpoint(path) -> int` — hydrates accumulators in-place,
    preserving the CUDA-graph-pinned buffers; returns the loaded
    cumulative prompt count.
  - `get_n_prompts_accumulated() -> int` / `set_n_prompts_accumulated(n)` —
    driver-owned cumulative counter; source of truth for `m_last_chunk`.
- `vllm/calibration_imatrix.py`: `_write_dat` now uses tmp+rename, so the
  final `.imatrix.dat` is also atomic.
- `tests/test_calibration_imatrix_smoke.py`: +5 tests covering checkpoint
  round-trip, two-segment additivity, atomic-write crash safety, cumulative
  `chunk_count`, and the new final-dump atomicity contract.

### `calib-v2-patch-locked` (previous, frozen)
| Patch line count | 3087 |
| Patch MD5 | `9effe235a95940d806f626ee1dc841c8` |

Tag kept immutable so older wheels referencing it remain reproducible.

## Driver-side companion changes (NOT in this patch — they live in the repo)

Both behaviors above require the driver to participate; see commit history
on `feat/calibration-v2`:
- `max_quality/scripts/build_self_traces_calib_vllm.py`
  - New flag `--imatrix-checkpoint-every-chunks` (default 1).
  - Calls `load_imatrix_checkpoint` after `setup(llm)` when `--resume` and the
    checkpoint file exists.
  - Calls `dump_imatrix_checkpoint` inside the chunk loop after each JSONL
    flush boundary.
  - Final `dump_imatrix` uses `get_n_prompts_accumulated()` (cumulative
    across instance lifetimes).
  - `.npz` logit sidecars now written via tmp+rename for atomicity.
  - JSONL resume validates each line as JSON and truncates at the first
    parse failure (drops trailing partial lines from preempted writes).

## Schema bumps

Sidecar schema versions for the cached-calibration-signals provider-pair
infrastructure. Bump these when changing the dataclass layout in
`max_quality/src/moe_compress/utils/cached_calibration_signals.py`.

| Signal | schema_version | Notes |
|---|---|---|
| `phase_b` | 1 | initial |
| `stage2_profile` | 3 | v3 -- Plugin #12 REDO (Optimization A). Replaces deleted v1 (delta_gate / delta_expert / a_gate_up / a_down / token_counts). Now: `format_version: 3`, `model_hash`, `top_k`, `cov_storage_dtype` (cross-validated), `total_tokens_per_layer [n_layers] int64`, `gate_logit_profiles dict[int -> list[(int, Tensor[T_b, E] fp32)]]`, `sim_tensor [n_layers, E, E] fp64`, `neuron_act_sum/_count dict[(layer_rank, expert)]`, `cov_acc/_token_count dict[(layer_rank, expert, matrix_name)]` (finalized covariance, FP/BF/F16), `layer_input_reservoir list[Tensor[N, hidden] bf16]` (len == n_layers, always populated when --capture-stage2-profile is on). All dict keys use `layer_rank` (portable across models). |
| `reap_scores` | 1 | initial — V1+V2 writers campaign, REAP Eq. 9 (arXiv:2510.13999); `[n_layers, n_experts] float32` + matching `int64` counts |
| `per_expert_max` | 1 | initial — Item 2 writers campaign, Stage 1 cheap-pruning candidate-ranking signal; `[n_layers, n_experts] float32` (max of `\|f_j(x)\|_inf` over tokens routed to expert j in layer rank l, zero-filled for zero-traffic cells) + matching `int64` token counts |
| `routing_stats` | 1 | initial — Item 3 writers campaign, per-(layer, expert) routing frequency + mean routing weight; `[n_layers, n_experts]` int64 freq + float32 mean_weight (zero where freq==0, no NaN). NO immediate downstream consumer; payload deposited on `ctx.routing_stats_payload` for future plugins (routing-aware ablation gating, mean-weight-weighted REAP variants). |
| `router_logits_stats` | 1 | initial — Item 4 writers campaign, per-(layer, expert) sink-vs-normal router-score aggregates; `[n_layers, n_experts]` float32 `score_sink_sum` + float32 `score_normal_sum` + int64 `fire_on_sink` + per-layer int64 `n_sink_tokens` / `n_normal_tokens` + `bos_token_id`. Consumed by Stage 1's `Stage1RouterLogitsStatsCacheProvider` -- on hit hydrates a pre-finalized `SinkTokenRoutingAccumulator` into `ctx.sink_acc` and the orchestrator drops `"sink_routing"` from the live calibration HookSpec set (sink-token detection runs from the cached aggregates). |
| `covariance` | 2 | Item 1 writers campaign — dict-valued payload, keys `(layer_idx, expert_idx, matrix_name)` -> fp16 `Tensor[d_in, d_in]`, byte-shape-compatible with the Stage 2 writer's `_stage2_input_covariance.pt`. v1 was never persisted by a production writer; forward-only bump. |
| `router_kd_logits` | 1 | initial — matches the existing .npz writer format in `build_self_traces_calib_vllm.py` |
| `block_hidden` | 1 | initial |
| `teacher_eval` | 1 | initial |

**JSONL self-traces row schema bumps** (CALIBRATION_MIX_V2_PLAN.md Step 5/6):

  * `build_self_traces_calib.py` `_trace_cache_key` schema_version: 6 → 7.
  * `build_self_traces_calib_vllm.py` `_trace_cache_key_vllm` schema_version: 8 → 9.

Both bumps add the per-row `completion_source` field
(`"teacher_generated"` for rows produced by model.generate;
`"canonical"` for v2 TEACHER_FORCED rows synthesized from the source
dataset's canonical assistant turn). Existing v6/v8 JSONLs on disk are
NOT cache-hit by v7/v9 runs — the bumps intentionally segregate so a
v1-era cache doesn't silently mix into a v2 calibration tensor. See
`max_quality/docs/calibration_mix_v2.md` for the full v2 mix definition.

## Performance disclosures

Driver-side / plugin-side perf characteristics worth surfacing at the
manifest level so CI flake reviews don't re-derive them from scratch.

### `expert_distill` v2 cold-cache CI cost

`expert_distill.py` v2 target (D-expert-distill-paper-lift, Lift 2,
paper Eqs. 1-3) adds a per-group `_router_routing_weights` recompute
(full-softmax `σ_orig(x)` over the unpruned router, then a TopK mask
and renormalization) inside `_distill_merged_group`. Empirical wall-
clock on the `tests/test_stage2_assignment_v2.py` suite:

- Main branch baseline: 52 passed, 6 skipped in ~3.7s
- This branch, cold cache: ~120s total with occasional `--timeout=60`
  failures
- This branch, warm cache: 1.5-13s (no regression)

The cold slowdown is the cost of the per-group router-routing-weight
recompute, paid once per group on the first distillation pass through
a layer. Warm runs amortize via Python's import cache + torch's kernel
cache.

If this becomes a CI flake (cold-cache runs intermittently exceed the
60s wall), the cheapest fix is to **cache the routing weights at first
compute and reuse them across the per-group loop** — `σ_orig` is
shared across all groups within a layer (the router is the same
tensor), so the current per-group recompute is O(num_groups) wasteful.
This is a follow-up optimization tracked as audit finding E-2 (MEDIUM)
in `tasks/AUDIT_CALIBRATION_COMPLETENESS_V2.md`.

Authoritative source: `expert_distill.py:262-285`.

## When to bump the immutable tag

Whenever the patch's functional contents change (line count or MD5 differ),
create a new `calib-v2-<short-mnemonic>` tag pointing at the new patch
commit, update this file's "Current" section, update the HF Jobs build
script's expected MD5 comment, and re-run the HF Jobs build to publish a
fresh wheel. Older wheels keep working with their original tag.
