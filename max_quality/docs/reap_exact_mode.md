# REAP-exact mode

A screening configuration for the Strategy A pipeline that produces a fast,
honest teacher-vs-student signal **without** running stages 2.5, 3, 4, 5.
Stage 2 is configured to behave as **pure pruning** (no merging, no
distillation, no heal), and the compressed model is handed directly to the
stage 6alt thermometer.

Status: realized. First one-shot mode using calibration-v2 signals.

---

## When to use

- You want a quick directional read on **how a REAP-score-based prune alone
  performs at the target ratio** before committing to the full pipeline.
- You are screening many ablation rows where running the full pipeline
  (~hours per row) would be prohibitive.
- You are debugging Stage 2's pure-prune path and need a clean isolation
  (no Stage 2.5 router KD obscuring the result).

Do **not** use it as a final-quality artifact. Use
`qwen36_35b_a3b_30pct.yaml` for production runs.

---

## What it changes vs the production config

### Stage 2 — pure prune

| Key | Production (`30pct.yaml`) | REAP-exact |
|---|---|---|
| `skip_merge_percentile` | absent (defaults to OFF sentinel 100.0) | `0.0` (mask all merges) |
| `expert_distill_steps` | `0` | `0` (explicit) |
| `merge_heal_enabled` | `false` | `false` (explicit) |
| `cost_asymmetric` | `false` | `false` (explicit) |

`skip_merge_percentile: 0.0` is the lever: the 0th percentile is the
minimum finite cost; every entry strictly above the 0th-percentile finite
cost is set to `+inf`, so the assignment solver assigns nothing and
non-centroid experts are pure-dropped, not merged into a centroid.

### Stage 6 evaluator

`stage6_validate.mode: thermometer` — uses the cheap forward-pass BPT
signal (~$0.22/row) instead of the full WikiText-PPL + lm-eval + HumanEval
+ MATH-500 suite (~$50-120/row). The mode is also forced by the
orchestrator from `pipeline.evaluator` (see below) so misconfiguration
cannot accidentally launch the 2-hour suite.

### New `pipeline:` top-level section

These keys are read **only** by `run_pipeline.py`. Stage modules do not
see them.

```yaml
pipeline:
  skip_intermediate_stages: true   # skip 2.5 / 3 / 4 / 5
  evaluator: stage6alt             # 'stage6' or 'stage6alt'
```

`evaluator: stage6alt` is the default for REAP-exact (cheap screening).
Override to `stage6` if you want a single-row, full-suite teacher-vs-
student damage report.

---

## What stages run

| Stage | Normal pipeline | REAP-exact |
|---|---|---|
| 1   GRAPE + SE detection | yes | yes |
| 2   REAP + REAM         | yes | yes (pure-prune posture) |
| 2.5 Post-merge router KD | yes | **skipped** |
| 3   SVD                 | yes | **skipped** |
| 4   EoRA                | yes | **skipped** |
| 5   Router KD           | yes | **skipped** |
| 6   Validation          | full or thermometer | thermometer (default) or full |

The stage 6 loader's candidate fallback was extended to look for
`stage2_pruned/` as a third option (after `stage5_final/` and
`stage2p5_final/`), since REAP-exact never produces those intermediate
artifacts.

---

## How the orchestrator implements the skip

`run_pipeline.main()` reads the `pipeline:` section once at startup:

1. Validates `pipeline.evaluator ∈ {"stage6", "stage6alt"}`.
2. When `skip_intermediate_stages: true`, forces
   `config["stage6_validate"]["mode"]` to `full` or `thermometer` based
   on the evaluator (overriding any value in the YAML).
3. Suppresses the Stage 2.5 invocation by setting the local `skip_stage25`
   flag.
4. Each of the `stop < N` / `start <= N <= stop` guards for stages 3, 4,
   5 (and the `stop < 6` gate) gains an `and not _skip_intermediate`
   condition, so the body is unreachable in REAP-exact mode.
5. The `start <= 6 <= stop` block is **not** modified — REAP-exact
   reaches it normally and the existing `mode` dispatch picks
   `stage6_validate.run` vs `stage6alt_thermometer.run`.

No new CLI flag was introduced. The mode is YAML-only.

---

## Invocation

```bash
python -m moe_compress.run_pipeline \
    --config configs/qwen36_35b_a3b_reap_exact.yaml \
    --artifacts-dir ./artifacts/reap_exact
```

The required calibration sidecar (REAP scores + optional iMatrix) must be
present. Build it with:

```bash
python max_quality/scripts/build_self_traces_calib_vllm.py \
    --capture-reap-scores [--capture-imatrix] ...
```

(See the calibration-v2 V1+V2 writers — `1fd9cc4` + `36f08af`.)

---

## Tests

- `max_quality/tests/test_reap_exact_config.py` — pure YAML schema check.
- `max_quality/tests/test_run_pipeline_reap_exact.py` — orchestration
  smoke: stages 2.5/3/4/5 stay skipped; stage6alt fires by default;
  evaluator=stage6 routes to stage6_validate; bad evaluator raises.
- `max_quality/tests/test_run_pipeline_normal_mode_regression.py` —
  baseline: without `pipeline:` (or with `skip_intermediate_stages:
  false`), all six stages still run.

---

## Limitations

- This is a **screening tool**, not a final-quality artifact. The
  literature on REAP shows pure-prune is usually beaten by REAP+merge or
  REAP+heal at the same compression ratio.
- The thermometer BPT metric is directional only — the absolute number
  is not leaderboard-comparable. Use `evaluator: stage6` for one
  rigorous run on the winning row of a sweep.
- The skip is hard: there is no resume path that picks up at Stage 3
  from a REAP-exact `stage2_pruned/`. To run the full pipeline against
  a REAP-exact-style Stage 2, build a sibling config with the four
  pure-prune keys flipped on but `pipeline.skip_intermediate_stages:
  false`.
