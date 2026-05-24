"""Stage 1 — Super-Expert detection + GRAPE budgets (plugin architecture).

Papers (one paper per plugin; canonical citations live in the per-plugin
module docstrings):

- arXiv:2507.23279 "Unveiling Super Experts in MoE LLMs" (Su et al., ICLR
  2026) — Super-Expert detection (ma_detection, three_way_and, sink_token).
- arXiv:2603.18492 "AIMER: Calibration-Free Task-Agnostic MoE Pruning"
  (Liu et al.) — AIMER weight-importance scoring (aimer).
- arXiv:1905.00414 "Similarity of Neural Network Representations Revisited"
  (Kornblith et al., ICML 2019) — CKA primitive (cka_distance).
- arXiv:2604.06542 "Does a Global Perspective Help Prune Sparse MoEs
  Elegantly?" (GRAPE; Zhang et al.) — entropy-aware greedy merge with
  restart (grape_merge); CKA-distance consumer (cka_distance).

Pipeline shape — Stage 1 is unified two-pass calibration + four-source
candidate generation + ablation filter + GRAPE merge:

  Pass 1 (Algorithm 1 Stage 1, paper 2507.23279):
    Build the MA-formation layer set ``L`` via
    :mod:`stage1.plugins.ma_detection`.

  Pass 2 (Algorithm 1 Stage 2 + project candidate detectors):
    Single forward pass over ~256 calibration samples instrumenting all
    MoE layers simultaneously, collecting per-(layer, expert):

      - Max activation magnitude at down_proj output (for three-way AND,
        AIMER, magnitude top-K).
      - Expert output representations (reservoir-sampled, cap = 256
        tokens/expert) for CKA pairwise similarity.
      - Sink-token routing aggregates (mean router score on sink vs
        normal tokens, freq on sink).

    The four candidate detectors run on the post-pass accumulators:
    three_way_and (paper Eq. 6 + L-filter), aimer (paper Eq. 4),
    sink_token (project-original detector keyed on paper Figs. 6/20/21),
    magnitude_topk (project-original, K = 2 × top-routing).
    De-duplication by ``(layer, expert)``; provenance carries the
    union of detectors that flagged each candidate.

  Ablation filter (project-original; see
  :mod:`stage1.plugins.ablation_filter` D-causal-ablation-validation):
    Per-candidate ΔNLL on a held-out slice; final blacklist =
    ``{(l, e) | ΔNLL > threshold}``.

  CKA + GRAPE merge:
    :mod:`stage1.plugins.cka_distance` builds per-layer pairwise CKA
    distance matrices; :mod:`stage1.plugins.grape_merge` runs
    arXiv:2604.06542 Algorithm 1 (entropy-aware greedy merge with
    restart) on those matrices to produce per-layer budgets ``N'_l``.

Sampling parameters (project-specified):

  - MA-formation detection pass: ``phase_a_batch_size = 32``. Tracks max
    magnitudes only, so batch-size invariant.
  - Main instrumentation pass: ``phase_b_batch_size = 8``. Every
    accumulator (``DownProjMaxAccumulator``, ``ExpertOutputAccumulator``
    reservoir sampling, ``SinkTokenRoutingAccumulator`` vectorized
    reduction) handles arbitrary ``B``; the prior ``bs=1`` was inherited
    from a per-token routing-instrumentation path that the vectorization
    eliminated. Cuts forward-pass count from 1024 to 128.
  - Ablation filter: ``ablation_filter_batch_size = 8`` — see
    :mod:`stage1.plugins.ablation_filter` git archaeology for the bs=32→8
    drop after the 2026-05-10 H200 job 6a00caf0 OOM on
    ``ForCausalLMLoss`` bf16→fp32 logits upcast.
  - ``num_calibration_samples = 1024`` (down from 4000): saturates
    per-layer max and the 256-token reservoir while staying within the
    <5 % Frobenius drift threshold reported in arXiv:2603.18492's
    calibration sensitivity figure.

Why L matters (referenced by three_way_and, aimer, magnitude_topk,
sink_token): the paper documents that some experts produce extreme
down_proj output magnitudes outside the MA-formation layers — these are
called "outlier experts" (Table 7: L1E8, L47E48, L47E100 for
Qwen3-30B-A3B; see Appendix C). Tables 6 and 7 are internally
inconsistent for the first outlier expert (Table 6: "Layer 47 Expert 8";
Table 7: "Layer 1 Expert 8"); the Table-6 entry "Layer 47 Expert 8" is
almost certainly a typo for "Layer 1 Expert 8", so this stage follows
Table 7's L1E8 reading. These outlier experts do not contribute to MA
formation and are not SEs. Not all outlier experts are excluded by the
L-filter: L1E8 sits in Layer 1, which IS an MA-formation layer
(l ∈ L); Table 7 lists it as an outlier expert that is not classified
as an SE, implying it fails the magnitude thresholds rather than being
excluded by the L-filter (spec inference; paper does not explicitly
classify why L1E8 fails the SE criterion). L47E48 and L47E100 sit
outside L and are excluded by the L-filter. The ``l ∈ L`` constraint
ensures that late-layer outlier experts outside L could not be
blacklisted even if their magnitudes were large enough to satisfy the
P99.5 and 0.1·a_max thresholds. Appendix C establishes that outlier
experts lack the mechanistic significance of SEs but does not assert
they would or would not pass the numerical thresholds.

Properties of L: MA formation in MoE models typically begins in the
first 1-3 decoder layers and then stabilises — Mixtral exhibits this in
a single layer (paper §3.2.2 / Table 2: Mixtral-8x7B-Instruct SE at
"Layer 1 Expert 3"), Qwen3-30B-A3B in three consecutive early layers.
The MA pattern, once established, propagates stably across all
subsequent layers via residual connections, so ``L`` is a small set of
early layers (not the full layer stack). Note: this three-layer
observation applies to Qwen3-30B-A3B (the paper's subject model); the
pipeline's target model (Qwen3.6-35B-A3B) has a different architecture
and its ``L`` will be determined empirically at runtime.

Empirical scale of SEs (paper 2507.23279 Table 1): fewer than 0.5 % of
all experts across the MoE models studied — 0.05 % for Qwen3-30B-A3B,
0.06 % for DeepSeek-R1, 0.11 % for DeepSeek-V2-Lite-Chat, 0.39 % for
Mixtral-8x7B-Instruct-v0.1. The three-way AND source alone reproduces
the paper's canonical SE set on Qwen3-30B-A3B (Table 2: L1E68, L2E92,
L3E82); the project broadens the candidate pool with the other three
sources (AIMER, sink-token, magnitude top-K) to catch architecture-
shifted SEs that the static three-way AND threshold misses on Qwen3.6.

Blacklist output (``stage1_blacklist.json``): the ablation-validated SE
blacklist — ``(layer, expert)`` index pairs whose ablation-pass ΔNLL
exceeded ``ablation_filter_threshold``. The candidate pool is recorded
separately under ``three_way_and.candidates``, ``aimer.candidates``,
``sink_token.candidates``, ``magnitude_topk.candidates`` for audit; the
per-candidate ΔNLL is in the companion ``stage1_ablation_filter.json``.

Shared experts (``mlp.shared_expert``) are **not in the blacklist** and
are never processed by Stage 1. They live in a separate model
attribute, distinct from the routed ``mlp.experts`` list, and are
architecturally invisible to ``iter_moe_layers``, GRAPE, and REAM. No
explicit exclusion is needed.

Naming-history note: legacy code identifiers carry "Phase A / B / C / D
/ E / F" prefixes (log strings, Trackio keys, internal variable names).
The current plugin architecture has no phase taxonomy; new prose drops
the labels. Existing identifiers preserved for dashboard back-compat.
"""
from .orchestrator import run
from .stage import STAGE1

__all__ = ["run", "STAGE1"]
