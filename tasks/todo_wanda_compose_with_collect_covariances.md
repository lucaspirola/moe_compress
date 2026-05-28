# TODO: Compose Wanda intra-expert calibration with `_collect_covariances`

## Context
`stage3/plugins/wanda_intra_expert_score.py` currently runs its own per-layer
calibration sweep via `instrument_experts`, doubling Stage 3 calibration time
when enabled. The callback shape (`{"input": ..., "intermediate": ...}`)
matches what `_collect_covariances` already uses.

## Plan
1. Extend `_collect_covariances` to accept an optional `extra_callbacks: dict[str, list[callable]] | None = None`
2. Register the Wanda accumulator's `input`/`intermediate` callbacks alongside the existing cov callbacks
3. Drop the standalone per-layer pass in `WandaIntraExpertScorePlugin.collect_wanda_scores`
4. Re-run `D-zero-extra-forward` removal + the existing 24 Wanda tests

## Cost (current state)
~2× Stage 3 calibration wall-clock when `stage3.wanda_intra_expert.enabled=True`.

## Tracking
Surfaced by Wanda reviewer 2026-05-28; not blocking initial Wanda plugin merge.
