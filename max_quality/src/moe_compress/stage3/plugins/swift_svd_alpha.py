"""Swift-SVD+ α-search (S3-4 of the Stage 3 plugin-architecture refactor).

Home of the Swift-SVD+ α-selection and rank-redistribution logic relocated
VERBATIM from the legacy ``stage3_svd.py`` monolith:

* ``_snapshot_originals`` — CPU snapshot of all expert weights (shared by the
  validation α search and Stage 4 EoRA residuals);
* ``_build_wikitext2_validation`` — builds the WikiText-2 validation tensor;
* ``_evaluate_wikitext2_ppl`` — end-to-end WikiText-2 perplexity;
* ``_factor_model_at_ranks`` — factors the full model in-place at candidate
  per-expert ranks (one α candidate);
* ``_restore_fused_experts`` — reverses ``_factor_model_at_ranks`` from the
  CPU snapshot;
* ``_swift_svd_plus_alpha_search_validation`` — paper-exact α selection via
  end-to-end WikiText-2 PPL validation (Swift-SVD 2604.01609 §3.2.2);
* ``_swift_svd_plus_alpha_search`` — spectral-proxy α selection per projection
  type (Swift-SVD+ 2604.01609 Algorithm 2);
* ``_redistribute_ranks_swift_svd_plus`` — given the selected α, computes the
  per-expert rank allocation (budget-conserving within each group).

All eight symbols are byte-identical copies of the monolith bodies; the
monolith re-imports them (``# noqa: F401`` block in ``stage3_svd.py``) so
``run()`` and external callers/tests keep their existing import paths.

Circular-import note (mirror of ``stage3/plugins/d_rank_allocate.py``): this
module imports only stdlib / torch / ``...utils.*`` / ``...pipeline.context``
and the sibling ``.d_rank_allocate`` plugin — NEVER from ``stage3_svd`` at
module scope. ``stage3_svd`` imports *this* module at load time, so a
module-top ``from ...stage3_svd import ...`` here would deadlock the import.

Lazy-import escape: three relocated functions — ``_factor_model_at_ranks``,
``_swift_svd_plus_alpha_search`` and ``_redistribute_ranks_swift_svd_plus`` —
depend on the AA-SVD core (``_cov_lookup``, ``_precompute_eigh``, ``_aa_svd``,
``_aa_svd_precomputed``) which is still monolith-resident (it is S3-5's
relocation target). A module-top ``from ...stage3_svd import ...`` for those
names would deadlock the import cycle described above, so they are imported at
FUNCTION scope as the first statement of each of the 3 functions. Each function
imports ONLY the AA-SVD-core names it actually references. The deferred import
is byte-identical at call time — same object identity, only the name
resolution is deferred to first call. ``_EighDecomp`` is NOT imported: it
appears here solely as a string type annotation (``from __future__ import
annotations``), never evaluated.

The α-cache I/O (the ``_stage3_alpha_result.json`` read/write) stays inline in
the monolith ``run()`` — it is absorbed into the plugin hook at S3-7.

``SwiftSvdAlphaPlugin`` is registered-but-INERT at S3-4 — no walk or test
invokes its ``select_alpha`` hook. S3-7 wires it into the live Stage 3 plugin
sequencer.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ...utils.model_io import MATRIX_NAMES, FactoredExperts, MoELayerRef, build_banks
from ...utils.trackio_log import trackio_log as _trackio_log
from ...pipeline.context import PipelineContext
from .d_rank_allocate import _GroupStats

log = logging.getLogger(__name__)


def _snapshot_originals(
    moe_layers: list[MoELayerRef],
) -> dict[tuple[int, int, str], torch.Tensor]:
    """CPU snapshot of all expert weights.

    Used by (a) validation-based α search (factor → eval → restore) and
    (b) Stage 4 EoRA residual computation. Moved before the α search so
    both consumers share the same snapshot.

    Memory: ~50 GB CPU RAM for Qwen3.6-35B-A3B post-prune (~200 experts ×
    40 layers × 3 matrices × [512,2048] bf16). H200 has 256 GB host RAM;
    combined with A-cov (~68 GB) this leaves ~128 GB headroom.
    """
    originals: dict[tuple[int, int, str], torch.Tensor] = {}
    for ref in moe_layers:
        banks = build_banks(ref)
        for e in range(ref.num_routed_experts):
            for name in MATRIX_NAMES:
                originals[(ref.layer_idx, e, name)] = (
                    banks[name].get(e).detach().cpu().clone()
                )
    return originals


def _build_wikitext2_validation(
    tokenizer,
    n_seqs: int,
    seq_len: int = 2048,
) -> torch.LongTensor:
    """Build a WikiText-2 validation tensor for α search.

    Uses the standard WikiText-2 raw test set: concatenate with EOS
    between documents, chunk to fixed ``seq_len``, return the first
    ``n_seqs`` full-length chunks.

    This mirrors Stage 6's ``_wikitext2_ppl`` tokenization exactly so
    the α-search PPL and Stage 6's final PPL are directly comparable.
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Spec §9 WikiText-2 PPL Protocol (F-iter4-M-1): rows are joined with
    # `"\n\n"` (matching the lm-eval / HF recipe and the imatrix calibration
    # corpus). Stage 3's α-search PPL must use the same protocol as Stage 6
    # so the two perplexities are comparable. Empty rows are PRESERVED — the
    # canonical recipe contributes the literal "\n\n\n\n" boundary tokens
    # to the chunk stream; filtering them changes chunk boundaries vs.
    # Stage 6 and breaks comparability.
    rows = [row.get("text", "") for row in ds]
    joined = "\n\n".join(rows)
    all_ids = tokenizer(joined, add_special_tokens=True)["input_ids"]

    n_full = len(all_ids) // seq_len
    if n_full == 0:
        # Spec §6 Phase B.2 paper-compliance contract: the α-search MUST
        # complete the paper-exact end-to-end PPL grid. Silent degradation
        # to a spectral proxy was D9 (removed). Fail fast.
        raise RuntimeError(
            f"Stage 3 α-search: WikiText-2 yielded no full-length sequences "
            f"(seq_len={seq_len}, total tokens={len(all_ids)}). "
            "Cannot complete the paper-exact PPL grid."
        )
    n_use = min(n_full, n_seqs)
    return torch.tensor(
        all_ids[: n_use * seq_len], dtype=torch.long,
    ).view(n_use, seq_len)


def _evaluate_wikitext2_ppl(
    model, val_tensor: torch.LongTensor, *, device, batch_size: int = 16,
) -> float:
    """Compute WikiText-2 perplexity on pre-tokenized sequences.

    Matches Stage 6's ``_wikitext2_ppl`` methodology: next-token NLL
    averaged over all non-first positions, then exp(mean_NLL).
    """
    if val_tensor.numel() == 0:
        return float("inf")
    model.eval()
    nll_sum = 0.0
    tok_count = 0
    for i in range(0, val_tensor.size(0), batch_size):
        batch = val_tensor[i : i + batch_size]
        if device is not None:
            batch = batch.to(device)
        with torch.no_grad():
            out = model(input_ids=batch, labels=batch)
        # out.loss is the mean NLL over (seq_len - 1) positions per sequence.
        n_tokens = batch.numel() - batch.shape[0]
        nll_sum += float(out.loss.item()) * n_tokens
        tok_count += n_tokens
    if tok_count == 0:
        return float("inf")
    return math.exp(nll_sum / tok_count)


def _factor_model_at_ranks(
    model,
    moe_layers: list[MoELayerRef],
    originals: dict[tuple[int, int, str], torch.Tensor],
    per_expert_ranks: dict[tuple[int, str, int], int],
    base_ranks: dict[tuple[int, str], int],
    A_cov: dict,
    B_acc,
    bcov_spill_dir: Path,
    C_acc,
    ccov_spill_dir: Path | None,
    *,
    device,
    storage_dtype: torch.dtype = torch.float16,
) -> None:
    """Factor all MoE layers in-place at the given per-expert ranks.

    Used by the validation-based α search: for each candidate α, this
    function installs FactoredExperts at the candidate's rank allocation.
    After evaluation, ``_restore_fused_experts`` reverses the swap.

    Covariance is lazy-loaded per layer from spill files and immediately
    unloaded, keeping in-memory footprint bounded to one layer (~5 GB).
    """
    # S3-4 lazy import: the AA-SVD core stays monolith-resident (S3-5 target);
    # a module-top import would deadlock the import cycle (see module docstring).
    from ...stage3_svd import (  # noqa: PLC0415
        _cov_lookup, _precompute_eigh, _aa_svd, _aa_svd_precomputed,
    )
    for ref in moe_layers:
        # Load covariances for this layer.
        B_acc.load_layer_from_disk(ref.layer_idx, bcov_spill_dir)
        if C_acc is not None and ccov_spill_dir is not None:
            C_acc.load_layer_from_disk(ref.layer_idx, ccov_spill_dir)

        # Slot width = max per-expert rank within this layer/matrix.
        ranks_layer = {
            name: max(
                per_expert_ranks.get(
                    (ref.layer_idx, name, e),
                    base_ranks[(ref.layer_idx, name)],
                )
                for e in range(ref.num_routed_experts)
            )
            for name in MATRIX_NAMES
        }

        ex = ref.experts_module
        dtype = ex.gate_up_proj.dtype
        # Offload dense experts to CPU before allocating FactoredExperts
        # to avoid brief double-occupancy.
        ex.to("cpu")
        torch.cuda.empty_cache()

        new_factored = FactoredExperts(
            num_experts=ref.num_routed_experts,
            hidden_dim=ex.gate_up_proj.shape[-1],
            intermediate_dim=ex.gate_up_proj.shape[1] // 2,
            ranks=ranks_layer,
            dtype=dtype,
            device=device,
        )

        for e in range(ref.num_routed_experts):
            # --- Precompute shared eigh for gate_proj / up_proj ---
            B_shared = _cov_lookup(B_acc.covariance, ref.layer_idx, e, "gate_proj")
            A_shared = _cov_lookup(A_cov, ref.layer_idx, e, "gate_proj")
            C_shared = None
            if C_acc is not None:
                C_shared = _cov_lookup(C_acc.covariance, ref.layer_idx, e, "gate_proj")
            gate_up_decomp: _EighDecomp | None = None
            if B_shared is not None:
                try:
                    gate_up_decomp = _precompute_eigh(
                        B_shared, A_shared, C_shared,
                        device=device, storage_dtype=storage_dtype,
                    )
                except ValueError:
                    pass  # falls through to full _aa_svd below

            for name in MATRIX_NAMES:
                W = originals[(ref.layer_idx, e, name)].to(
                    device=device, dtype=torch.float32,
                )
                k = per_expert_ranks.get(
                    (ref.layer_idx, name, e),
                    base_ranks[(ref.layer_idx, name)],
                )
                if name in ("gate_proj", "up_proj") and gate_up_decomp is not None:
                    U_k, V_k, _, k_eff = _aa_svd_precomputed(
                        W, gate_up_decomp, k, device=device,
                    )
                else:
                    A = _cov_lookup(A_cov, ref.layer_idx, e, name)
                    B = _cov_lookup(B_acc.covariance, ref.layer_idx, e, name)
                    C = None
                    if C_acc is not None:
                        C = _cov_lookup(C_acc.covariance, ref.layer_idx, e, name)
                    U_k, V_k, _, k_eff = _aa_svd(
                        W, A, B, k, C=C, device=device,
                        storage_dtype=storage_dtype,
                    )
                new_factored.set_factors(
                    e, name, U_k, V_k, effective_rank=k_eff,
                )

        # Swap in.
        setattr(ref.mlp, "experts", new_factored)
        ref.experts_module = new_factored

        # Free this layer's covariance from memory.
        B_acc.unload_layer(ref.layer_idx)
        if C_acc is not None:
            C_acc.unload_layer(ref.layer_idx)


def _restore_fused_experts(
    model,
    moe_layers: list[MoELayerRef],
    originals: dict[tuple[int, int, str], torch.Tensor],
    *,
    device,
) -> None:
    """Restore original fused experts from the CPU snapshot.

    Reverses ``_factor_model_at_ranks``: replaces each layer's
    FactoredExperts with the original ``Qwen3_5MoeExperts`` fused module
    reconstructed from the ``originals`` dict.

    The fused module is rebuilt manually as a ``SimpleNamespace``-style
    module with the correct ``gate_up_proj`` and ``down_proj`` stacked
    tensors that ``build_banks`` expects.
    """
    for ref in moe_layers:
        fe = ref.experts_module
        if not isinstance(fe, FactoredExperts):
            continue
        dtype = fe.gate_proj_U.dtype
        n = ref.num_routed_experts
        d_int = fe.intermediate_dim
        d_hid = fe.hidden_dim

        # Free FactoredExperts from GPU.
        fe.to("cpu")
        torch.cuda.empty_cache()

        # Rebuild fused storage on GPU from CPU originals.
        gate_up = torch.zeros(n, 2 * d_int, d_hid, dtype=dtype, device=device)
        down = torch.zeros(n, d_hid, d_int, dtype=dtype, device=device)

        for e in range(n):
            gate_w = originals[(ref.layer_idx, e, "gate_proj")].to(
                dtype=dtype, device=device,
            )
            up_w = originals[(ref.layer_idx, e, "up_proj")].to(
                dtype=dtype, device=device,
            )
            down_w = originals[(ref.layer_idx, e, "down_proj")].to(
                dtype=dtype, device=device,
            )
            gate_up[e, :d_int] = gate_w
            gate_up[e, d_int:] = up_w
            down[e] = down_w

        # Reconstruct the fused experts module. We need a module that
        # ``build_banks`` / ``_is_fused_experts`` recognises: it must have
        # ``gate_up_proj`` and ``down_proj`` as Parameters and a
        # ``num_experts`` attribute.
        fused = nn.Module()
        fused.gate_up_proj = nn.Parameter(gate_up, requires_grad=False)
        fused.down_proj = nn.Parameter(down, requires_grad=False)
        fused.num_experts = n
        # Copy the act_fn and forward from the original class if available.
        # Not strictly necessary — we only need the weights accessible via
        # build_banks for the final factoring loop. The forward will be
        # replaced when the final FactoredExperts is installed.
        from transformers.activations import ACT2FN
        fused.act_fn = ACT2FN["silu"]

        setattr(ref.mlp, "experts", fused)
        ref.experts_module = fused


def _swift_svd_plus_alpha_search_validation(
    model,
    tokenizer,
    moe_layers: list[MoELayerRef],
    group_stats: dict[tuple[int, str], _GroupStats],
    base_ranks: dict[tuple[int, str], int],
    alpha_grid: list[float],
    originals: dict[tuple[int, int, str], torch.Tensor],
    A_cov: dict,
    B_acc,
    bcov_spill_dir: Path,
    C_acc,
    ccov_spill_dir: Path | None,
    config: dict,
    *,
    device,
    storage_dtype: torch.dtype = torch.float16,
) -> float:
    """Paper-exact α selection via end-to-end WikiText-2 PPL validation.

    Implements Swift-SVD (2604.01609) §3.2.2:

        "Swift-SVD uses a fixed retention ratio δ=0.5 and 11 scaling
        factors α=[0, 0.1, 0.2, ..., 1] to generate 11 candidate rank
        allocations. For each candidate corresponding to α_i, the optimal
        low-rank approximation of every layer is computed using the
        closed-form solution in (3). The resulting compressed models are
        then evaluated on a validation set, and the candidate that yields
        the best end-to-end performance is selected."

    For each α in ``alpha_grid``:
      1. Compute per-expert rank redistribution (Algorithm 2 blending score)
      2. Factor the full model layer-by-layer via AA-SVD at those ranks
      3. Evaluate WikiText-2 PPL on ``validation_samples`` sequences
      4. Restore original fused experts from CPU snapshot

    Returns the α with the lowest PPL.

    **Cost**: 11 candidates × (~2 min factor + ~0.3 min eval + ~0.5 min
    restore) ≈ ~31 min on H200 for Qwen3.6-35B-A3B.

    **Memory**: CPU RAM holds originals (~50 GB) + A-cov (~68 GB) ≈ ~128 GB.
    H200 has 256 GB host RAM → ~128 GB headroom. VRAM holds the factored
    model (~34 GB) during eval → ~107 GB headroom on 141 GB.
    """
    svd_plus_cfg = config["stage3_svd"]["swift_svd_plus"]
    validation_samples = int(svd_plus_cfg.get("validation_samples", 512))
    validation_batch_size = int(svd_plus_cfg.get("validation_batch_size", 16))

    # Build WikiText-2 validation tensor (same tokenization as Stage 6).
    log.info("Stage 3 α-search: building WikiText-2 validation set "
             "(%d sequences, seq_len=2048)", validation_samples)
    val_tensor = _build_wikitext2_validation(
        tokenizer, n_seqs=validation_samples, seq_len=2048,
    )
    if val_tensor.numel() == 0:
        # Spec §6 Phase B.2 paper-compliance contract: hard-fail rather
        # than silently degrade to a non-paper-compliant α.
        raise RuntimeError(
            "Stage 3 α-search: validation tensor is empty after building. "
            "Spec §6 Phase B.2 mandates the paper-exact end-to-end PPL grid; "
            "the previously-shipped silent spectral-proxy fallback (D9) was "
            "removed for non-compliance."
        )

    log.info("Stage 3 α-search: %d validation sequences (%d tokens)",
             val_tensor.size(0), val_tensor.numel())

    best_alpha = 0.5
    best_ppl = float("inf")
    results: list[tuple[float, float]] = []

    for idx, alpha in enumerate(alpha_grid):
        log.info("Stage 3 α-search: candidate %d/%d (α=%.1f)",
                 idx + 1, len(alpha_grid), alpha)

        # 1. Compute per-expert ranks for this α (single α for all types).
        alpha_by_type = {"all": alpha}
        per_expert_ranks = _redistribute_ranks_swift_svd_plus(
            moe_layers, group_stats, base_ranks, alpha_by_type,
            A_cov=A_cov,
        )

        # 2. Factor the full model at these ranks. Forward storage_dtype so
        # the noise floor in `_aa_svd` matches the main factoring pass — using
        # the default fp16 floor on a bf16-stored B-cov would over-truncate.
        _factor_model_at_ranks(
            model, moe_layers, originals, per_expert_ranks, base_ranks,
            A_cov, B_acc, bcov_spill_dir, C_acc, ccov_spill_dir,
            device=device, storage_dtype=storage_dtype,
        )

        # 3. Evaluate WikiText-2 PPL.
        ppl = _evaluate_wikitext2_ppl(
            model, val_tensor, device=device,
            batch_size=validation_batch_size,
        )
        results.append((alpha, ppl))
        log.info("  α=%.1f → WikiText-2 PPL=%.4f", alpha, ppl)
        _trackio_log({
            "stage3/alpha_search/alpha": alpha,
            "stage3/alpha_search/ppl": ppl,
        })

        # 4. Restore original fused experts for the next candidate.
        _restore_fused_experts(model, moe_layers, originals, device=device)

        if ppl < best_ppl:
            best_ppl = ppl
            best_alpha = alpha

    log.info("Stage 3 α-search complete: best α=%.1f (PPL=%.4f)", best_alpha, best_ppl)
    log.info("  full results: %s",
             ", ".join(f"α={a:.1f}→{p:.4f}" for a, p in results))
    _trackio_log({
        "stage3/alpha_search/best_alpha": best_alpha,
        "stage3/alpha_search/best_ppl": best_ppl,
    })
    return best_alpha


def _swift_svd_plus_alpha_search(
    moe_layers: list,
    group_stats: dict[tuple[int, str], _GroupStats],
    base_ranks: dict[tuple[int, str], int],
    alpha_grid: list[float],
    *,
    per_group_type: bool = True,
    A_cov: dict | None = None,
) -> dict[str, float]:
    """Swift-SVD+ (2604.01609, Algorithm 2): select α per projection type.

    For each candidate α, compute the blending score for every expert within
    each (layer, matrix_type) group:

        s_i = β_i^α · (log(e + ε*_i))^{1-α}

    where:
      - β_i = σ_i² / Σ_j σ_j²  (spectral energy proportion — how much of the
        group's total spectral energy this expert contributes)
      - ε*_i = √(Σ_{j>k̄} σ_j² / Σ_j σ_j²)  (reconstruction error at the
        group's mean rank k̄ — higher = this expert needs more rank)

    Then redistribute the group's total rank budget proportionally to s_i.
    The α that minimises the total weighted reconstruction error across all
    experts in the group wins.

    Returns {matrix_type: best_α} if per_group_type, else {"all": best_α}.
    """
    # S3-4 lazy import: _cov_lookup stays monolith-resident (S3-5 target);
    # a module-top import would deadlock the import cycle (see module docstring).
    from ...stage3_svd import _cov_lookup  # noqa: PLC0415
    import math as _math

    # Collect per-expert singular value spectra, grouped by matrix type.
    # When A_cov is available (D8 fix), compute activation-weighted SVD
    # (SVD of W @ L_A) instead of raw SVD. This gives ε* that reflects
    # actual reconstruction error weighted by input distribution.
    # grouped_svs[name][(layer_idx, expert_idx)] = singular_values tensor
    grouped_svs: dict[str, dict[tuple[int, int], torch.Tensor]] = {
        n: {} for n in MATRIX_NAMES
    }
    for (li, name), gs in group_stats.items():
        banks = build_banks([ref for ref in moe_layers if ref.layer_idx == li][0])
        for e in range(gs.n_experts):
            W = banks[name].get(e).detach().to(torch.float32)
            # D8 fix: activation-weighted singular values when A_cov available.
            A = _cov_lookup(A_cov, li, e, name) if A_cov else None
            if A is not None:
                A_f32 = A.to(torch.float32)
                A_f32 = 0.5 * (A_f32 + A_f32.T)
                eigvals_a, eigvecs_a = torch.linalg.eigh(A_f32)
                keep_a = eigvals_a > eigvals_a.max() * 1e-6
                if keep_a.any():
                    L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a].clamp_min(1e-12).sqrt().unsqueeze(0)
                    M_A = W @ L_A
                    svs = torch.linalg.svdvals(M_A)
                else:
                    svs = torch.linalg.svdvals(W)
            else:
                svs = torch.linalg.svdvals(W)
            grouped_svs[name][(li, e)] = svs

    def _evaluate_alpha(name: str, alpha: float) -> float:
        """Total weighted reconstruction error for this α across all experts
        in the given projection type."""
        group_keys = [(li, n) for (li, n) in base_ranks if n == name]
        total_err = 0.0
        for (li, n) in group_keys:
            gs = group_stats[(li, n)]
            k_group = base_ranks[(li, n)]
            # Collect per-expert scores.
            expert_ids = list(range(gs.n_experts))
            betas: list[float] = []
            epsilons: list[float] = []
            energies: list[float] = []
            for e in expert_ids:
                svs = grouped_svs[n][(li, e)]
                s2 = (svs * svs)
                total_energy = float(s2.sum().clamp_min(1e-30).item())
                energies.append(total_energy)
                # ε*_i at reference rank k_group
                tail = float(s2[k_group:].sum().item()) if k_group < len(s2) else 0.0
                epsilons.append((tail / total_energy) ** 0.5)
            # β_i = energy_i / total_energy_in_group
            group_energy = sum(energies) or 1.0
            betas = [e_val / group_energy for e_val in energies]
            # Blending scores
            scores = []
            for beta, eps in zip(betas, epsilons):
                s = (beta ** alpha) * (_math.log(_math.e + eps) ** (1.0 - alpha))
                scores.append(max(s, 1e-12))
            # Redistribute group rank budget proportionally to scores.
            total_score = sum(scores) or 1.0
            total_group_rank = k_group * gs.n_experts
            per_expert_ranks = [
                max(1, min(min(gs.d_out, gs.d_in) - 1,
                           int(round(total_group_rank * (sc / total_score)))))
                for sc in scores
            ]
            # Evaluate: sum of tail energy at allocated rank per expert.
            for e, k_e in zip(expert_ids, per_expert_ranks):
                svs = grouped_svs[n][(li, e)]
                s2 = svs * svs
                tail = float(s2[k_e:].sum().item()) if k_e < len(s2) else 0.0
                total_err += tail
        return total_err

    if per_group_type:
        best_alphas: dict[str, float] = {}
        for name in MATRIX_NAMES:
            best_alpha = 0.5
            best_err = float("inf")
            for alpha in alpha_grid:
                err = _evaluate_alpha(name, alpha)
                if err < best_err:
                    best_err = err
                    best_alpha = alpha
            best_alphas[name] = best_alpha
            log.info("  Swift-SVD+ %s: best α=%.1f (err=%.4e)", name, best_alpha, best_err)
        return best_alphas
    else:
        best_alpha = 0.5
        best_err = float("inf")
        for alpha in alpha_grid:
            err = sum(_evaluate_alpha(n, alpha) for n in MATRIX_NAMES)
            if err < best_err:
                best_err = err
                best_alpha = alpha
        log.info("  Swift-SVD+ global: best α=%.1f (err=%.4e)", best_alpha, best_err)
        return {"all": best_alpha}


def _redistribute_ranks_swift_svd_plus(
    moe_layers: list,
    group_stats: dict[tuple[int, str], _GroupStats],
    base_ranks: dict[tuple[int, str], int],
    alpha_by_type: dict[str, float],
    *,
    grouped_svs_cache=None,
    A_cov: dict | None = None,
) -> dict[tuple[int, str, int], int]:
    """Given the selected α per type, compute per-expert ranks.

    Returns {(layer_idx, matrix_name, expert_idx): rank}.
    The total rank within each (layer, matrix_type) group is conserved
    (sum of per-expert ranks = base_rank × n_experts).
    """
    # S3-4 lazy import: _cov_lookup stays monolith-resident (S3-5 target);
    # a module-top import would deadlock the import cycle (see module docstring).
    from ...stage3_svd import _cov_lookup  # noqa: PLC0415
    import math as _math

    out: dict[tuple[int, str, int], int] = {}
    for (li, name), gs in group_stats.items():
        k_group = base_ranks[(li, name)]
        alpha = alpha_by_type.get(name, alpha_by_type.get("all", 0.5))

        # Collect per-expert singular values (activation-weighted when A_cov available).
        banks = build_banks([ref for ref in moe_layers if ref.layer_idx == li][0])
        energies: list[float] = []
        epsilons: list[float] = []
        for e in range(gs.n_experts):
            W = banks[name].get(e).detach().to(torch.float32)
            # D8 fix: activation-weighted SVD when A_cov available.
            A = _cov_lookup(A_cov, li, e, name) if A_cov else None
            if A is not None:
                A_f32 = A.to(torch.float32)
                A_f32 = 0.5 * (A_f32 + A_f32.T)
                eigvals_a, eigvecs_a = torch.linalg.eigh(A_f32)
                keep_a = eigvals_a > eigvals_a.max() * 1e-6
                if keep_a.any():
                    L_A = eigvecs_a[:, keep_a] * eigvals_a[keep_a].clamp_min(1e-12).sqrt().unsqueeze(0)
                    svs = torch.linalg.svdvals(W @ L_A)
                else:
                    svs = torch.linalg.svdvals(W)
            else:
                svs = torch.linalg.svdvals(W)
            s2 = svs * svs
            total_e = float(s2.sum().clamp_min(1e-30).item())
            energies.append(total_e)
            tail = float(s2[k_group:].sum().item()) if k_group < len(s2) else 0.0
            epsilons.append((tail / total_e) ** 0.5)

        group_energy = sum(energies) or 1.0
        betas = [e_val / group_energy for e_val in energies]
        scores = [
            max((b ** alpha) * (_math.log(_math.e + eps) ** (1.0 - alpha)), 1e-12)
            for b, eps in zip(betas, epsilons)
        ]
        total_score = sum(scores) or 1.0
        total_group_rank = k_group * gs.n_experts
        cap = min(gs.d_out, gs.d_in) - 1
        # Spec §6 Phase B.2 redistribution rule (paper 2604.01609 Algorithm 2
        # lines 4–9): every expert starts at floor(k̄·δ); the remaining pool
        # `b = k̄·L·(1−δ)` is distributed by score share. δ = 0.5 — paper
        # warns δ = 0 is numerically unstable.
        delta = 0.5
        rank_floor = max(1, int(math.floor(k_group * delta)))
        flexible_pool = k_group * gs.n_experts * (1.0 - delta)

        per_e = [
            max(rank_floor, min(cap, rank_floor + int(math.floor(flexible_pool * (sc / total_score)))))
            for sc in scores
        ]
        # Reconcile rounding residual.
        diff = total_group_rank - sum(per_e)
        if diff != 0:
            order = sorted(range(gs.n_experts),
                           key=lambda i: scores[i], reverse=(diff > 0))
            for idx in order:
                if diff == 0:
                    break
                step = 1 if diff > 0 else -1
                new_val = per_e[idx] + step
                if rank_floor <= new_val <= cap:
                    per_e[idx] = new_val
                    diff -= step
            if diff != 0:
                log.warning(
                    "Swift-SVD+ rank reconciliation: residual drift %+d in group "
                    "(layer=%d, matrix=%s) — all experts at rank cap/floor, "
                    "actual total rank differs from budget by %d.",
                    diff, li, name, abs(diff),
                )

        for e, k_e in enumerate(per_e):
            out[(li, name, e)] = k_e
    return out


class SwiftSvdAlphaPlugin:
    """Stage 3 Swift-SVD+ α-selection plugin (S3-4 — registered-but-INERT).

    Owns the Swift-SVD+ α-selection phase: the paper-exact end-to-end
    WikiText-2 PPL grid (``_swift_svd_plus_alpha_search_validation``), the
    spectral-proxy per-type α search (``_swift_svd_plus_alpha_search``) and the
    per-expert rank redistribution (``_redistribute_ranks_swift_svd_plus``)
    — plus the snapshot / factor / restore / evaluate helpers they drive. The
    phase logic lives in the module-level functions relocated verbatim from the
    monolith (Swift-SVD+ paper 2604.01609, §3.2.2 / Algorithm 2).

    S3-4 wires this class into the plugin registry as metadata only — no walk
    or test invokes ``select_alpha``. S3-7 plugs the hook into the live Stage 3
    plugin sequencer.
    """

    name = "swift_svd_alpha"
    paper = (
        "Swift-SVD+ α selection — blending score s_i = β_i^α · "
        "(log(e + ε*_i))^(1−α) (paper 2604.01609, §3.2.2 / Algorithm 2)."
    )
    config_key = "stage3_svd.swift_svd_plus.alpha_grid"
    reads: tuple[str, ...] = (
        "model", "tokenizer", "moe_layers", "group_stats", "ranks",
        "originals", "A_cov", "B_acc", "bcov_spill_dir", "C_acc",
        "ccov_spill_dir", "config", "device",
    )
    writes: tuple[str, ...] = ("alpha_by_type", "per_expert_ranks")
    provides: tuple[str, ...] = ()

    def is_enabled(self, config: dict) -> bool:
        """Always True — Swift-SVD+ α selection is UNCONDITIONAL.

        Every Stage 3 run resolves an ``alpha_by_type`` / ``per_expert_ranks``
        pair. An ``alpha_grid`` of length ≤ 1 does NOT disable the phase — it
        simply yields the uniform path (``alpha_by_type = None``, every expert
        keeps the group rank). ``config_key`` only parametrises the grid; it
        never gates the plugin as a whole.
        """
        return True

    def contribute_artifact(self, ctx: Any) -> dict:
        return {}

    def select_alpha(self, ctx: PipelineContext) -> None:
        """Phase hook — Swift-SVD+ α selection (S3-7 wiring surface).

        INERT at S3-4: no orchestrator walk or test invokes this hook. S3-7
        replaces the Stage 3 orchestrator body with the plugin sequencer and
        dispatches this hook in place of the monolith's inline α-dispatch. The
        body reproduces the ``run()`` α-dispatch: pick the validation or
        spectral-proxy path off ``alpha_grid`` / ``validation_samples`` and
        redistribute per-expert ranks. The α-cache read/write stays in the
        monolith ``run()`` for now — it is absorbed here at S3-7.

        Dead code at S3-4; kept faithful to the monolith dispatch but S3-7
        validates and finalises it.
        """
        # Required slots — direct get(): a missing one is a wiring bug and
        # SHOULD raise. Optional slots are has()-guarded (get() raises KeyError
        # on an unset slot).
        model = ctx.get("model")
        moe_layers = ctx.get("moe_layers")
        group_stats = ctx.get("group_stats")
        ranks = ctx.get("ranks")
        config = ctx.get("config")
        A_cov = ctx.get("A_cov") if ctx.has("A_cov") else None

        s3 = config.get("stage3_svd", {})
        svd_plus_cfg = s3.get("swift_svd_plus", {})
        alpha_grid = svd_plus_cfg.get("alpha_grid")
        per_group_type = svd_plus_cfg.get("per_group_type", True)
        validation_samples = int(svd_plus_cfg.get("validation_samples", 0))

        if alpha_grid and len(alpha_grid) > 1:
            if validation_samples > 0:
                # Paper-exact: global α via WikiText-2 PPL validation (§3.2.2).
                best_global_alpha = _swift_svd_plus_alpha_search_validation(
                    model,
                    ctx.get("tokenizer"),
                    moe_layers,
                    group_stats,
                    ranks,
                    alpha_grid,
                    ctx.get("originals"),
                    A_cov,
                    ctx.get("B_acc"),
                    ctx.get("bcov_spill_dir"),
                    ctx.get("C_acc") if ctx.has("C_acc") else None,
                    ctx.get("ccov_spill_dir") if ctx.has("ccov_spill_dir") else None,
                    config,
                    device=ctx.get("device"),
                )
                if per_group_type:
                    alpha_by_type = _swift_svd_plus_alpha_search(
                        moe_layers, group_stats, ranks, alpha_grid,
                        per_group_type=True, A_cov=A_cov,
                    )
                else:
                    alpha_by_type = {"all": best_global_alpha}
            else:
                # Fallback: spectral proxy only (no forward passes).
                alpha_by_type = _swift_svd_plus_alpha_search(
                    moe_layers, group_stats, ranks, alpha_grid,
                    per_group_type=per_group_type, A_cov=A_cov,
                )
            per_expert_ranks = _redistribute_ranks_swift_svd_plus(
                moe_layers, group_stats, ranks, alpha_by_type,
                grouped_svs_cache=None, A_cov=A_cov,
            )
        else:
            # alpha_grid length ≤ 1 — uniform path: every expert keeps the
            # group rank, no redistribution.
            alpha_by_type = None
            per_expert_ranks = None

        ctx.set("alpha_by_type", alpha_by_type)
        ctx.set("per_expert_ranks", per_expert_ranks)
