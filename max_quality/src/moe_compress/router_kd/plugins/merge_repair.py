"""Stage-2.5 merge-repair concern (RK-6 of the Router-KD plugin refactor).

Paper
-----
Hyeon & Do, "Is Retraining-Free Enough? The Necessity of Router
Calibration for Efficient MoE Compression" — arXiv:2603.02217 (§F.3,
Eq. 3, Table 1). audit/spec_compliance/01_papers/2603.02217/source.md.

Equation 3: the per-batch vocab-KL distillation objective
    L_KD = KL(softmax(s_t / τ) || softmax(s_s / τ)) · τ²
where ``s_t``, ``s_s`` are the teacher and student vocabulary logits
and ``τ`` is the distillation temperature.

§F.3 fixes the calibration data and hyperparameters; Table 1 reports
the resulting recovery on Mixtral/Qwen-MoE post-pruning/post-merging.

Official code
-------------
**None published.** Verified 2026-05: the paper's source.md contains
no code link; first author Sieun Hyeon (Seoul National University) has
no public router-KD repo.

Calibration deviation D11 (SHARED with Stage 2 / Stage 2.5)
-----------------------------------------------------------
Paper §F.3 Table 1 uses ``c4``. The project uses multi-domain
Nemotron-Cascade-2-SFT-Data with weighted subsets — task-aware
calibration better matches target deployment distribution. The D11
row's canonical owner is :mod:`stage2.plugins.reap_scoring`.

Deviation D-merge-repair-grad-flow (CANONICAL OWNER — declared here)
--------------------------------------------------------------------
Paper §5 (audit/spec_compliance/01_papers/2603.02217/source.md L824–834)
fixes the Eq. 3 frozen-experts contract:

    "although the distillation loss is defined on the output token
    distribution, gradients are backpropagated and applied exclusively
    to the student router parameters θ_R, while all expert and backbone
    parameters remain frozen … parameter updates are restricted to the
    router."

Direction E (this plugin, opt-in via ``stage5_router_kd.merge_repair.enabled``)
deviates from that contract along TWO axes:

  (a) **Centroid-row unfreeze.** :func:`_unfreeze_merged_experts` sets
      ``requires_grad=True`` on the stacked ``gate_up_proj`` / ``down_proj``
      expert parameters of every layer that has merged centroids, and
      registers a per-row gradient mask so the optimizer updates *only*
      the centroid rows. From the paper's strict reading "expert
      parameters remain frozen" this is the deviation: those rows
      receive gradient. The router param-group remains the primary
      trainable surface; the unfrozen expert rows form a second AdamW
      group with ``weight_decay=0.0`` (the downstream realization lives
      in :class:`router_kd.plugins.kd_optimizer.KdOptimizerPlugin` — see
      its module docstring's "Deviation D-merge-repair-grad-flow"
      cross-ref row, which points HERE as the canonical owner).

  (b) **Per-layer MoE-output MSE term.** Eq. 3 is a single vocab-KL
      objective. This plugin's :func:`_merge_repair_mse` adds a
      ``mse_weight * mean_layers F.mse_loss(student_block_out,
      teacher_block_out)`` term to the loss (combined in
      :meth:`VocabKdPlugin.compute_kd_loss`). The student block-output
      capture keeps autograd history; the teacher capture is detached
      (fixed MSE target). This term is project-original — Eq. 3 does
      not include any per-layer hidden-state regression.

**Default-off invariant.** When
``stage5_router_kd.merge_repair.enabled`` is false (the default),
:meth:`MergeRepairPlugin.is_enabled` returns ``False`` for every
``stage_key`` value. No centroid rows are unfrozen, no MoE-block
forward hooks are registered, no MSE term is published, and the
KdOptimizerPlugin falls back to its single ``weight_decay=_wd`` AdamW
group over router-only trainables. The flag-off behaviour is therefore
byte-identical to pre-Direction-E ``main`` and paper-faithful w.r.t.
the Eq. 3 frozen-parameter scope. The deviation is gated entirely by
that one config flag.

Home of the Router-KD Stage-2.5 *merge-repair* concern (Direction E),
extracted from the legacy ``stage5_router_kd.py`` monolith.

Direction E — the strong form of merge-repair (spec §E / tasks §E): in
Stage 2.5, in addition to training the router, also train the *merged
centroid experts* (the kept experts that absorbed others during Stage-2
merging) against a direct per-layer MoE-output MSE target taken from the
teacher. It is config-gated and default-off, so the flag-off path is
byte-identical to pre-Direction-E ``main`` (see the "Default-off
invariant" in the D-merge-repair-grad-flow block above).

Piece A — relocated verbatim (Pattern A, the RK-2/RK-3/RK-4 pattern):
  SEVEN STANDALONE module-level symbols are relocated here character-for-
  character — ``_load_merge_map`` (the Stage-2 merge-map loader),
  ``_merged_centroid_rows`` (post-merge centroid-row identification),
  ``_select_merge_repair_layers`` (live model-agnostic repair-layer
  selection), ``_experts_param_tensors`` (the stacked-expert-weight
  accessor), ``_unfreeze_merged_experts`` (the centroid unfreeze + grad-mask
  hook), the ``_LayerOutputCapture`` CLASS (the MoE-block forward-hook
  capture) and ``_merge_repair_mse`` (the per-layer MoE-output MSE term).
  They are relocated verbatim; the ``stage5_router_kd.py`` monolith
  re-imports them (``# noqa: F401`` block) so ``run()`` and external
  callers/tests (``test_stage5_merge_repair.py``) keep their import paths.

Piece B — the inert hooks (Pattern B): :class:`MergeRepairPlugin` carries
three phase hooks — ``setup_merge_repair`` / ``compute_merge_repair_mse`` /
``teardown_merge_repair`` — that REPRODUCE the inline merge-repair glue
scattered through the monolith ``run()`` (the setup block, the per-batch
MSE term + lazy teacher-capture registration, the hook-removal teardown).
They are INERT at RK-6 — no orchestrator walk or test invokes them — and
exist as the RK-8 wiring surface; RK-8 plugs them into the live Router-KD
plugin sequencer and deletes the monolith ``run()``.

Stage-gated ``is_enabled``: merge-repair is a STAGE-2.5-ONLY concern — the
plugin's :meth:`MergeRepairPlugin.is_enabled` ANDs the stage with the config
flag, returning ``True`` only when this plugin instance is bound to stage
``"stage2p5"`` AND ``stage5_router_kd.merge_repair.enabled`` is true. A
default-constructed plugin (``stage_key="stage5"``) is therefore always
disabled.

Circular-import note (mirror of ``vocab_kd.py`` / ``teacher.py``): this
module imports only from ``...pipeline.*`` / ``..context`` / ``...utils.*`` /
stdlib / torch — NEVER from ``stage5_router_kd`` or
``router_kd.orchestrator`` at any scope (module-top OR function-local). The
monolith re-imports *this* module at load time, so a
``from ..stage5_router_kd import ...`` here would deadlock the import;
nothing in this module does that.

``MergeRepairPlugin`` is registered-but-INERT at RK-6 — RK-8 plugs the hooks
into the live Router-KD plugin sequencer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..context import PipelineContext
from ...utils.model_io import iter_moe_layers

log = logging.getLogger(__name__)


def _load_merge_map(artifacts_dir: Path, override_path: "str | None") -> dict:
    """Load the Stage-2 merge map ``{layer_idx: {new_idx: [orig_ids...]}}``.

    The canonical artifact is ``stage2_pruned/merge_map.json`` written by
    Stage 2 (``stage2_reap_ream._write_merge_json`` / ``save_json_artifact``).
    ``override_path`` lets a caller point elsewhere; relative paths resolve
    against ``artifacts_dir``.

    Keys arrive as strings from JSON; they are normalized to ``int``.
    """
    if override_path:
        path = Path(override_path)
        if not path.is_absolute():
            path = artifacts_dir / path
    else:
        path = artifacts_dir / "stage2_pruned" / "merge_map.json"
    if not path.exists():
        raise RuntimeError(
            f"Stage 2.5 merge_repair: merge map not found at {path}. "
            "merge_repair needs the Stage-2 merge artifact to identify which "
            "experts are merged centroids. Run Stage 2 first, or set "
            "stage5_router_kd.merge_repair.merge_map_path."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict[int, list[int]]] = {}
    for layer_k, groups in raw.items():
        out[int(layer_k)] = {
            int(new_idx): [int(x) for x in members]
            for new_idx, members in groups.items()
        }
    return out


def _merged_centroid_rows(merge_map: dict, layer_idx: int) -> list[int]:
    """Return the post-merge expert row indices that are *merged centroids*.

    A new-index expert is a merged centroid iff the Stage-2 merge map records
    it as having absorbed >1 original expert (member-list length > 1). A
    length-1 entry is an expert kept untouched — it is NOT a repair target.

    Model-agnostic: the new-index keys ARE the post-merge expert rows in the
    fused stacked weight tensors, regardless of architecture.
    """
    groups = merge_map.get(layer_idx, {})
    return sorted(new_idx for new_idx, members in groups.items() if len(members) > 1)


def _select_merge_repair_layers(student: nn.Module, merge_map: dict) -> list:
    """Return ``(MoELayerRef, merged_rows)`` for every layer with ≥1 merged
    centroid, derived live from the model via ``iter_moe_layers``.

    Layer count / structure / expert count are all read from the model — no
    hardcoding. A layer with zero merged centroids contributes nothing.
    """
    selected = []
    for ref in iter_moe_layers(getattr(student, "_orig_mod", student)):
        rows = _merged_centroid_rows(merge_map, ref.layer_idx)
        if not rows:
            continue
        n = ref.num_routed_experts
        bad = [r for r in rows if not (0 <= r < n)]
        if bad:
            raise RuntimeError(
                f"Stage 2.5 merge_repair: layer {ref.layer_idx} merge map names "
                f"centroid row(s) {bad} outside the post-merge expert range "
                f"[0, {n}). The merge map does not match this student — "
                "regenerate Stage 2 or correct merge_map_path."
            )
        selected.append((ref, rows))
    return selected


def _experts_param_tensors(experts_module: nn.Module) -> list:
    """Return the stacked expert weight ``nn.Parameter``s of an experts module.

    Stage 2.5 runs *after* Stage-2 merging and *before* Stage-3 SVD
    factorization, so the experts are always the fused stacked form
    (``gate_up_proj`` / ``down_proj``) with the expert axis leading — which is
    what the gradient mask keys on. Any MoE whose experts module exposes those
    stacked parameters works; nothing here is Qwen-specific. (A post-Stage-3
    ``FactoredExperts`` module is intentionally NOT handled: its factors are
    not plain leading-axis ``nn.Parameter``s, and merge-repair never runs on a
    factored checkpoint — the explicit error below surfaces that misuse.)
    """
    names = ("gate_up_proj", "down_proj")
    out = []
    for name in names:
        t = getattr(experts_module, name, None)
        if isinstance(t, nn.Parameter):
            out.append(t)
    if not out:
        raise RuntimeError(
            "Stage 2.5 merge_repair: could not find the stacked expert weight "
            f"parameters (gate_up_proj / down_proj) on "
            f"{type(experts_module).__name__}; merge_repair cannot unfreeze "
            "merged experts on this architecture (a factored / post-Stage-3 "
            "experts module is not supported)."
        )
    return out


def _unfreeze_merged_experts(student: nn.Module, repair_layers: list) -> dict:
    """Unfreeze the merged-centroid experts for merge-repair.

    The expert weights are stored as a single stacked ``nn.Parameter`` per
    projection (leading axis = expert), so ``requires_grad`` cannot be set per
    row. We instead set ``requires_grad=True`` on the whole stacked tensor and
    register a gradient hook that zeroes every non-centroid expert row — so the
    optimizer updates *only* the merged centroids, exactly as the spec asks.

    Returns ``{id(param): grad-hook-handle}`` so the hooks can be removed and a
    record of which params were touched (for verification / checkpoint scope).
    """
    handles: dict = {}
    for ref, rows in repair_layers:
        row_idx = torch.tensor(sorted(rows), dtype=torch.long)
        for p in _experts_param_tensors(ref.experts_module):
            p.requires_grad_(True)
            # Capture row_idx by default-arg so each closure binds its own.
            # row_idx is built on CPU; move it to the gradient's device inside
            # the hook so the fancy-index works when training on GPU (a CPU
            # index tensor against a CUDA grad raises a device-mismatch error).
            def _mask_grad(grad, _rows=row_idx):
                _rows = _rows.to(grad.device)
                masked = torch.zeros_like(grad)
                masked[_rows] = grad[_rows]
                return masked
            handles[id(p)] = p.register_hook(_mask_grad)
    return handles


class _LayerOutputCapture:
    """Forward-hook capture of every MoE block's output hidden-state.

    Registers a ``register_forward_hook`` on each layer's ``mlp`` module (the
    MoE block). The hook stores that block's output tensor keyed by layer
    index. Works on teacher and student alike — the captured tensor is the
    block output regardless of architecture.

    ``detach`` controls whether captured tensors keep autograd history: the
    teacher capture detaches (it is a fixed target); the student capture keeps
    grad so the MSE term backpropagates into the merged experts + router.

    Shared-expert pass-through note (Qwen3-MoE family). The Qwen3-MoE block
    returns ``shared_expert(x) + sum_i g_i(x) * routed_expert_i(x)``. The hook
    fires on the block as a whole, so the captured tensor is the *sum* of the
    shared-expert and routed-expert paths — not the routed contribution alone.
    Merge-repair unfreezes only the routed-expert centroid rows
    (:func:`_experts_param_tensors` walks ``gate_up_proj`` / ``down_proj`` on
    ``experts_module``, never on the shared expert). Consequence: the
    shared-expert contribution is the SAME tensor at student and teacher iff
    no upstream stage has modified the shared expert, in which case the term
    cancels in the difference and the MSE depends only on the routed-expert
    output gap. If Stage 2 / Stage 3 (or any future stage) ever edits the
    shared expert, that pass-through becomes a non-zero residual baked into
    the MSE — visible at flag-on but invisible at flag-off.
    """

    def __init__(self, model: nn.Module, layer_indices: "set[int]", *, detach: bool):
        self._detach = detach
        self.outputs: dict[int, torch.Tensor] = {}
        self._handles: list = []
        base = getattr(model, "_orig_mod", model)
        wanted = set(layer_indices)
        for ref in iter_moe_layers(base):
            if ref.layer_idx not in wanted:
                continue
            self._handles.append(
                ref.mlp.register_forward_hook(self._make_hook(ref.layer_idx))
            )

    def _make_hook(self, layer_idx: int):
        def _hook(_module, _inp, output):
            # An MoE block returns either the hidden-state tensor directly or
            # a tuple whose first element is it (router-logits may follow).
            tensor = output[0] if isinstance(output, tuple) else output
            self.outputs[layer_idx] = tensor.detach() if self._detach else tensor
        return _hook

    def clear(self) -> None:
        self.outputs.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _merge_repair_mse(
    student_outputs: dict[int, torch.Tensor],
    teacher_outputs: dict[int, torch.Tensor],
    layer_indices: "list[int]",
) -> torch.Tensor:
    """Mean per-layer MSE between student and teacher MoE-block outputs.

    Averaged over the repair layers so the term's scale is independent of how
    many layers had merges (the config ``mse_weight`` is the only knob). The
    teacher tensor is cast to the student's dtype/device before the diff.
    Returns a scalar tensor with grad flowing into the student outputs.

    Averaging convention (double mean). Each per-layer term is
    ``F.mse_loss(s, t)`` — the elementwise squared error averaged over ALL
    elements of the block-output tensor (``batch × seq_len × hidden_size``).
    The per-layer terms are then ``torch.stack(...).mean()``'d across the
    repair layers. The final scalar is therefore a mean-over-elements then
    mean-over-layers; its absolute magnitude scales with how well the routed-
    expert paths already agree but is independent of ``hidden_size``,
    ``seq_len`` and the count of repair layers (each is normalized out).

    ``mse_weight`` tuning guidance. The default ``mse_weight=1.0`` (see
    :meth:`MergeRepairPlugin.setup_merge_repair`) was chosen for plumbing
    parity, not as a calibrated coefficient. The natural scale of the
    vocab-KL Eq. 3 term (``KL(softmax(s_t/τ) || softmax(s_s/τ)) · τ²``)
    differs from a hidden-state MSE by orders of magnitude depending on the
    student/teacher dtype, the activation magnitudes at each block boundary,
    and τ. When enabling Direction E for a real recovery run, ``mse_weight``
    should be tuned (typical sweep range 1e-3 … 1e1) so the MSE term does
    not dominate or vanish next to the vocab-KL term in
    :meth:`VocabKdPlugin.compute_kd_loss`.
    """
    if not layer_indices:
        # Defensive: with no repair layers the MSE term is identically zero.
        any_s = next(iter(student_outputs.values()), None)
        if any_s is None:
            return torch.zeros((), dtype=torch.float32)
        return torch.zeros((), device=any_s.device, dtype=torch.float32)
    terms = []
    for li in layer_indices:
        if li not in student_outputs or li not in teacher_outputs:
            raise RuntimeError(
                f"Stage 2.5 merge_repair: layer {li} output missing from "
                f"{'student' if li not in student_outputs else 'teacher'} "
                "capture — the MoE-block forward hook did not fire. Likely "
                "causes: the teacher/student MoE block layout differs, or "
                "torch.compile inlined the block forward so its submodule "
                "hook no longer runs."
            )
        s = student_outputs[li].to(torch.float32)
        t = teacher_outputs[li].to(device=s.device, dtype=torch.float32)
        if s.shape != t.shape:
            raise RuntimeError(
                f"Stage 2.5 merge_repair: layer {li} student/teacher MoE-output "
                f"shape mismatch {tuple(s.shape)} vs {tuple(t.shape)}."
            )
        terms.append(F.mse_loss(s, t))
    return torch.stack(terms).mean()


class MergeRepairPlugin:
    """Router-KD Stage-2.5 merge-repair plugin (RK-6 — registered-but-INERT).

    Owns the Direction-E merge-repair concern: identifying the merged centroid
    experts from the Stage-2 merge map, unfreezing them with a per-row
    gradient mask, capturing per-layer MoE-block outputs on teacher + student,
    and adding the per-layer MoE-output MSE term to the vocab-KL loss. The
    seven standalone functions/class above (``_load_merge_map`` …
    ``_merge_repair_mse``) are relocated verbatim (the monolith re-imports
    them).

    Merge-repair is a STAGE-2.5-ONLY concern: :meth:`is_enabled` ANDs the
    bound stage with the ``merge_repair.enabled`` config flag. A
    default-constructed plugin (``stage_key="stage5"``) is therefore always
    disabled.

    The three hooks (``setup_merge_repair`` / ``compute_merge_repair_mse`` /
    ``teardown_merge_repair``) REPRODUCE the inline merge-repair ``run()``
    glue; they are INERT at RK-6 — no orchestrator walk or test invokes them.
    RK-8 plugs them into the live Router-KD plugin sequencer.
    """

    name = "merge_repair"
    paper = (
        "Router KD vocab-KL distillation Eq. 3 — arXiv:2603.02217 "
        "(Hyeon & Do); no official code. Concern: Stage-2.5 merge-repair (Direction E, project-original; opt-in). "
        "Calibration D11 (SHARED — see :mod:`stage2.plugins.reap_scoring`). "
        "See module docstring."
    )
    config_key = "stage5_router_kd.merge_repair.enabled"
    # ``config`` / ``student`` / ``artifacts_dir`` drive the one-time setup;
    # ``teacher_logits_cache`` is read for the fail-loud incompatibility guard;
    # ``teacher`` is read for the lazy teacher-capture registration. The
    # per-batch / teardown hooks re-read the setup-published capture state
    # (``merge_repair_layers`` / ``merge_repair_mse_weight`` /
    # ``merge_repair_grad_handles`` / ``merge_repair_student_capture`` /
    # ``merge_repair_teacher_capture``) and the snapshotted
    # ``teacher_layer_outputs`` to build the MSE term.
    reads: tuple[str, ...] = (
        "config", "student", "artifacts_dir", "teacher_logits_cache",
        "merge_repair_layers", "merge_repair_mse_weight", "teacher",
        "merge_repair_teacher_capture", "merge_repair_student_capture",
        "teacher_layer_outputs", "merge_repair_grad_handles",
    )
    # The setup hook publishes the repair-layer list / grad handles / the MSE
    # weight / the student capture; the per-batch hook publishes the lazily
    # registered teacher capture and the MSE term (the slots
    # ``VocabKdPlugin.compute_kd_loss`` reads to combine into ``kd_loss``).
    writes: tuple[str, ...] = (
        "merge_repair_student_capture", "merge_repair_layers",
        "merge_repair_mse_weight", "merge_repair_grad_handles",
        "merge_repair_teacher_capture", "merge_repair_mse_term",
    )
    # Empty: merge-repair needs no separate calibration pass.
    provides: tuple[str, ...] = ()

    def __init__(self, stage_key: str = "stage5") -> None:
        # Bind the plugin to one Router-KD invocation. Merge-repair is a
        # Stage-2.5-only concern; the stage gate in is_enabled() keys on this.
        # Mirrors the _RouterKdStage.__init__ stage_key validation.
        if stage_key not in {"stage2p5", "stage5"}:
            raise ValueError(
                f"MergeRepairPlugin: unsupported stage_key={stage_key!r}; "
                "expected one of ['stage2p5', 'stage5']"
            )
        self._stage_key: str = stage_key

    def is_enabled(self, config: dict) -> bool:
        """True IFF bound to stage 2.5 AND ``merge_repair.enabled`` is set.

        Merge-repair is a STAGE-2.5-ONLY concern (Direction E trains the
        merged centroid experts that the Stage-2 merge produced — Stage 5 has
        no merge to repair). So the gate ANDs the stage with the config flag:
        a plugin bound to any stage other than ``"stage2p5"`` is
        unconditionally disabled, and even at Stage 2.5 it stays off unless
        ``stage5_router_kd.merge_repair.enabled`` is true. A default-
        constructed plugin (``stage_key="stage5"``) is therefore always
        disabled.
        """
        if self._stage_key != "stage2p5":
            return False
        return bool(
            (config.get("stage5_router_kd", {}).get("merge_repair", {}) or {})
            .get("enabled", False)
        )

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def setup_merge_repair(self, ctx: PipelineContext) -> None:
        """One-time setup hook — unfreeze merged centroids + register capture.

        INERT at RK-6: no orchestrator walk or test invokes this hook. RK-8
        calls it once before the epoch loop. The body reproduces the monolith
        ``run()`` inline merge-repair setup block faithfully — dead code at
        RK-6 but RK-8 + unit tests rely on it.

        Reproduces (in monolith order): read the ``merge_repair`` config
        block; fail loud if ``teacher_logits_cache`` is configured (the cache
        holds only vocabulary logits, not the teacher's per-layer MoE-block
        outputs the MSE term needs); load the Stage-2 merge map via
        ``_load_merge_map``; select the repair layers via
        ``_select_merge_repair_layers``; unfreeze the merged centroids via
        ``_unfreeze_merged_experts``; and register the student
        ``_LayerOutputCapture`` (grad-tracked — the teacher capture is
        registered lazily on the first live teacher forward in
        :meth:`compute_merge_repair_mse`). Publishes
        ``merge_repair_student_capture``, ``merge_repair_layers``,
        ``merge_repair_mse_weight`` and ``merge_repair_grad_handles``.
        """
        config = ctx.get("config")
        s5 = config["stage5_router_kd"]
        student = ctx.get("student")
        artifacts_dir = ctx.get("artifacts_dir")
        # A configured teacher-logits cache is read for the incompatibility
        # guard; absent slot is a valid state (no cache → live teacher).
        teacher_logits_cache = (
            ctx.get("teacher_logits_cache")
            if ctx.has("teacher_logits_cache")
            else None
        )

        mr_cfg = s5.get("merge_repair") or {}
        merge_repair_layers: list = []
        merge_repair_mse_weight = 0.0
        merge_repair_grad_handles: dict = {}
        # The per-layer MSE term needs the teacher's per-layer hidden-state
        # output, which a vocab-logits cache does not contain. Fail loud rather
        # than silently degrading to a router-only KD.
        if teacher_logits_cache is not None:
            raise RuntimeError(
                "Stage 2.5 merge_repair.enabled=true is incompatible with "
                "teacher_logits_cache: the cache holds only vocabulary logits, "
                "but merge-repair needs the teacher's per-layer MoE-block "
                "outputs. Remove teacher_logits_cache (run the teacher live) "
                "or disable merge_repair."
            )
        merge_repair_mse_weight = float(mr_cfg.get("mse_weight", 1.0))
        merge_map = _load_merge_map(artifacts_dir, mr_cfg.get("merge_map_path"))
        merge_repair_layers = _select_merge_repair_layers(student, merge_map)
        if not merge_repair_layers:
            log.warning(
                "Stage 2.5 merge_repair.enabled=true but the merge map records "
                "no merged centroids (every kept expert absorbed at most "
                "itself). merge-repair has nothing to train — the MSE term "
                "will be identically zero. Proceeding with router-only KD."
            )
        else:
            merge_repair_grad_handles = _unfreeze_merged_experts(
                student, merge_repair_layers
            )
            n_centroids = sum(len(rows) for _, rows in merge_repair_layers)
            log.info(
                "Stage 2.5 merge_repair: unfroze %d merged centroid experts "
                "across %d layers (mse_weight=%.4f)",
                n_centroids, len(merge_repair_layers), merge_repair_mse_weight,
            )

        # Register the student MoE-output capture (grad-tracked). The teacher
        # capture is registered lazily on the first live teacher forward in
        # compute_merge_repair_mse — the teacher itself is loaded lazily.
        merge_repair_layer_indices = [
            ref.layer_idx for ref, _ in merge_repair_layers
        ]
        if merge_repair_layer_indices:
            student_capture = _LayerOutputCapture(
                student, set(merge_repair_layer_indices), detach=False
            )
            ctx.set("merge_repair_student_capture", student_capture)

        ctx.set("merge_repair_layers", merge_repair_layers)
        ctx.set("merge_repair_mse_weight", merge_repair_mse_weight)
        ctx.set("merge_repair_grad_handles", merge_repair_grad_handles)

    def compute_merge_repair_mse(self, ctx: PipelineContext) -> None:
        """Per-batch hook — the merge-repair MSE term (RK-8 wiring surface).

        INERT at RK-6: no orchestrator walk or test invokes this hook. RK-8
        dispatches it per microbatch ahead of ``VocabKdPlugin.compute_kd_loss``.
        The body reproduces the monolith ``run()`` per-batch merge-repair glue
        faithfully — dead code at RK-6 but RK-8 + unit tests rely on it.

        Reproduces: the lazy teacher ``_LayerOutputCapture`` registration (the
        teacher is loaded lazily, so its detached capture is registered on the
        first live forward and reused), and the per-batch
        ``_merge_repair_mse(...)`` over the student/teacher captured MoE-block
        outputs. Publishes ``merge_repair_mse_term`` and
        ``merge_repair_mse_weight`` — the slots ``VocabKdPlugin.compute_kd_loss``
        reads to combine the term into ``kd_loss``.
        """
        merge_repair_layers = (
            ctx.get("merge_repair_layers")
            if ctx.has("merge_repair_layers")
            else []
        )
        merge_repair_layer_indices = [
            ref.layer_idx for ref, _ in merge_repair_layers
        ]
        merge_repair_mse_weight = (
            float(ctx.get("merge_repair_mse_weight"))
            if ctx.has("merge_repair_mse_weight")
            else 0.0
        )
        if not merge_repair_layer_indices:
            # Merge-repair off (or no merged centroids): no MSE term.
            return

        # Lazy teacher-capture registration: the teacher is loaded lazily, so
        # its detached MoE-output capture is registered on the first live
        # forward and reused every microbatch thereafter. Detached — the
        # teacher output is a fixed MSE target.
        teacher = ctx.get("teacher") if ctx.has("teacher") else None
        if (
            teacher is not None
            and not ctx.has("merge_repair_teacher_capture")
        ):
            teacher_capture = _LayerOutputCapture(
                teacher, set(merge_repair_layer_indices), detach=True
            )
            ctx.set("merge_repair_teacher_capture", teacher_capture)

        # The student capture is grad-tracked; its .outputs hold the current
        # microbatch's MoE-block outputs. The teacher capture's outputs are
        # snapshotted (already detached) into teacher_layer_outputs upstream.
        student_capture = (
            ctx.get("merge_repair_student_capture")
            if ctx.has("merge_repair_student_capture")
            else None
        )
        teacher_layer_outputs = (
            ctx.get("teacher_layer_outputs")
            if ctx.has("teacher_layer_outputs")
            else {}
        )
        mse_term = _merge_repair_mse(
            student_capture.outputs if student_capture is not None else {},
            teacher_layer_outputs,
            merge_repair_layer_indices,
        )

        ctx.set("merge_repair_mse_term", mse_term)
        ctx.set("merge_repair_mse_weight", merge_repair_mse_weight)

    def teardown_merge_repair(self, ctx: PipelineContext) -> None:
        """Teardown hook — remove the merge-repair hooks (RK-8 wiring surface).

        INERT at RK-6: no orchestrator walk or test invokes this hook. RK-8
        calls it once after the epoch loop, before the final save. The body
        reproduces the monolith ``run()`` inline teardown block faithfully —
        dead code at RK-6 but RK-8 + unit tests rely on it.

        Removes the gradient-mask hooks and the forward-capture hooks before
        the final save so no hook handles leak into the exported checkpoint's
        module tree. No-op when merge-repair is off (the containers are empty
        / absent).
        """
        merge_repair_grad_handles = (
            ctx.get("merge_repair_grad_handles")
            if ctx.has("merge_repair_grad_handles")
            else {}
        )
        for h in merge_repair_grad_handles.values():
            h.remove()
        merge_repair_grad_handles.clear()
        student_capture = (
            ctx.get("merge_repair_student_capture")
            if ctx.has("merge_repair_student_capture")
            else None
        )
        if student_capture is not None:
            student_capture.remove()
        teacher_capture = (
            ctx.get("merge_repair_teacher_capture")
            if ctx.has("merge_repair_teacher_capture")
            else None
        )
        if teacher_capture is not None:
            teacher_capture.remove()


__all__ = [
    "MergeRepairPlugin",
    "_load_merge_map",
    "_merged_centroid_rows",
    "_select_merge_repair_layers",
    "_experts_param_tensors",
    "_unfreeze_merged_experts",
    "_LayerOutputCapture",
    "_merge_repair_mse",
]
