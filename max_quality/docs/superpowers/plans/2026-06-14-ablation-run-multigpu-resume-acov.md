# Ablation Run: per-arm resume + multi-GPU + acov whitening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Make `run_reap_ream_35pct.py` actually execute the steps we agreed for the 2×H200 run: (1) reap arm resumes from its HF-backed `stage2p5_final` (skipping Stage 2+2.5), ream arm runs Stage 2→2.5→…; (2) all merged multi-GPU opt-in features are switched ON across every stage that supports them; (3) the Stage-4 EoRA whitening (`whitening_cov`) is parameterizable so the acov A-vs-A\* A/B can drive the arms. NEITHER arm runs Stage 1.

**Architecture:** All changes are confined to the runner `src/moe_compress/run_reap_ream_35pct.py` plus a new small helper for the HF seed, and a multi-GPU YAML overlay injected by `build_arm_config`. No stage-plugin code changes (the opt-in knobs already exist and are default-OFF; we only flip them in the per-arm config). Byte-identity of every stage's default path is unaffected because we touch only this ablation runner + its emitted config.

**Tech stack:** Python, `huggingface_hub.snapshot_download`, the existing `run_pipeline` subprocess contract.

---

## Verified facts (do not re-derive; spot-check the cites if editing near them)

Opt-in knobs (key path → resolver → default):
- Stage-2 profile DP: `stage2_reap_ream.profile_dp.{enabled,replicas,shards_per_model}` — `stage2/profile_dp.py:76-117`, default OFF. **Auto-disables** if a layer-input reservoir consumer is active (`expert_distill_steps>0` OR `cost_alignment="output"` OR `merge_step="mergemoe"`).
- Stage-3 cov DP: `multi_gpu.cov_replicas` (+`multi_gpu.shard_models`, `multi_gpu.cov_window_size`) — `stage3/orchestrator.py:79-104`, default 1.
- Stage-3 α-grid: `multi_gpu.alpha_workers` — `stage3/orchestrator.py:127-142`, default 1. (No effect unless `len(alpha_grid)>1`.)
- Stage-3 per-expert SVD: `multi_gpu.factor_workers` — `stage3/orchestrator.py:107-125`, default 1.
- Stage-4 EoRA concurrency: `multi_gpu.eora_workers` — `stage4/orchestrator.py:69-85`, default 1.
- Stage-2.5 + Stage-5 DDP: `stage5_router_kd.ddp.{enabled,world_size,backend}` — `router_kd/ddp_config.py:28-66`; **both 2.5 and 5 read `stage5_router_kd` regardless of stage_key** (`router_kd/orchestrator.py:200`). `world_size` must divide `stage5_router_kd.batch_size`. DDP **raises** if `world_size>=2` AND Stage-2.5 `merge_repair.enabled=true`.
- acov whitening: `stage4_eora.whitening_cov` ∈ {anchor(default),shift,anchored_adaptive} — `stage4/plugins/eora_inputs.py:363`. `shift`/`anchored_adaptive` **require** `stage3_svd.persist_shift_covariance: true` (`stage3/orchestrator.py:969`) or Stage 4 raises.
- Stage-6 eval-shard: `stage6_validate.eval_shard.*` lives ONLY in `stage6/`, NOT `stage6alt/`. **This ablation evals via `stage6alt` thermometer (`pipeline.evaluator=stage6alt`), so eval-shard does NOT apply — do NOT set it; the eval stays single-GPU.**

Resume/seed mechanics:
- `run_pipeline --resume-from-stage 3` loads `artifacts_dir/stage2p5_final/` from **disk** (`run_pipeline.py:516-535`); falls back to `stage2_pruned/` with a warning if absent. No in-process Hub download — the entrypoint must place the dir.
- reap checkpoint on HF: **`pirola/reap-s234-stage2p5-final`** (23 files, 52.1 GB: 11 safetensors shards + index + config + `compressed_metadata.json`, plus 4 `_shared/` metadata files). A resume@3 needs `stage2p5_final/{config.json, model-*.safetensors, model.safetensors.index.json, compressed_metadata.json}`.

Current runner shape:
- `ARMS = [("reap-s234","faithful_prune"),("ream-s234","merge")]`; `ARM_STAGE_WINDOWS=((2,2),(3,6))` applied to BOTH arms in `run_one_arm` (`run_reap_ream_35pct.py:489-513`).
- `build_arm_config` (`:290-349`) emits the per-arm YAML; `_pipeline_argv` (`:437-451`) builds the subprocess argv.
- Base config `configs/qwen36_35b_a3b_reap_faithful.yaml`: `model.device_map: auto`; none of the multi-GPU keys set.

---

## Task 1: Per-arm spec (windows + optional HF seed) replaces the global `ARM_STAGE_WINDOWS`

**Files:** Modify `src/moe_compress/run_reap_ream_35pct.py`. Test: `tests/test_run_reap_ream_35pct.py` (create if absent).

Replace the flat `ARMS` tuples + global `ARM_STAGE_WINDOWS` with a small `ArmSpec` dataclass carrying: `arm_id`, `method`, `seed_hub_repo: str | None`, `stage_windows: tuple[tuple[int,int],...]`.
- `reap-s234`: `method="faithful_prune"`, `seed_hub_repo="pirola/reap-s234-stage2p5-final"`, `stage_windows=((3,6),)` (Stage 3→6 ONLY).
- `ream-s234`: `method="merge"`, `seed_hub_repo=None`, `stage_windows=((2,2),(3,6))` (unchanged behavior).

- [ ] Step 1 — failing test: `test_arm_specs_reap_resumes_post_2p5_ream_runs_2p5` asserting the reap spec has `seed_hub_repo` set and windows `((3,6),)`, ream has `None` and `((2,2),(3,6))`.
- [ ] Step 2 — run it, verify it fails (AttributeError / no ArmSpec).
- [ ] Step 3 — implement the dataclass + the two specs; keep `--only` filtering by `arm_id`.
- [ ] Step 4 — verify pass.
- [ ] Step 5 — commit `feat(ablation): per-arm stage windows + HF seed spec`.

## Task 2: HF seed helper — place `stage2p5_final/` on disk for a seeded arm

**Files:** Modify the runner; new fn `_seed_stage2p5_from_hub(repo, arm_dir)`. Test: same test file.

Behavior: `snapshot_download(repo_id=repo, local_dir=<tmp>)`, then materialize `arm_dir/stage2p5_final/` containing the 4 required file kinds (config.json, model-*.safetensors, model.safetensors.index.json, compressed_metadata.json). Handle BOTH possible repo layouts (files at repo root, or under a `stage2p5_final/` subdir) — detect which by probing for `compressed_metadata.json`. If the repo also carries `_shared/` metadata, copy those into `shared_dir` only if missing (do not clobber a locally-seeded `_shared/`). **Idempotent**: if `arm_dir/stage2p5_final/compressed_metadata.json` already exists, skip the download. **Verify CONTENT** (per [[feedback_verify_content_not_filesize]]): after placing, assert `compressed_metadata.json` parses and the index lists all shards that exist on disk — raise loudly otherwise (a half-download must FAIL, not silently fall back to `stage2_pruned`).

- [ ] Step 1 — failing test using a fake snapshot dir (monkeypatch the download fn to populate a tmp dir with a tiny stub checkpoint in each candidate layout) → assert `arm_dir/stage2p5_final/compressed_metadata.json` ends up present and the idempotent re-call does not re-download.
- [ ] Step 2 — verify fail.
- [ ] Step 3 — implement (use a small injectable `_downloader=snapshot_download` seam so the test never hits the network).
- [ ] Step 4 — verify pass.
- [ ] Step 5 — commit `feat(ablation): HF stage2p5_final seed helper (content-verified, idempotent)`.

## Task 3: `run_one_arm` honors the per-arm spec

**Files:** runner `run_one_arm`. Test: same file.

- For a `seed_hub_repo` arm: call `_seed_stage2p5_from_hub` BEFORE the stage loop, then run ONLY the `(3,6)` window — skip the Stage-2/2.5 subprocess entirely. Keep the existing post-condition guard (`stage2p5_final/` must exist before resume@3) — it now passes because the seed placed it.
- For a non-seeded arm: run all windows in `stage_windows` in order (the existing two-subprocess behavior), keeping the `stage2p5_final/` existence guard between them.
- Generalize the body to iterate `spec.stage_windows` instead of the hardcoded `ARM_STAGE_WINDOWS[0]/[1]`.
- `seed_stage1_artifacts(...)` still runs for BOTH arms (the survivor guard reads `_shared/` even for reap; harmless and already present).

- [ ] Step 1 — failing test: monkeypatch `subprocess.run` to record argv + create the expected output dirs; assert reap issues ONE pipeline call with `--resume-from-stage 3 --stop-after-stage 6` and ZERO `--resume-from-stage 2` calls, and that the seed helper was invoked; ream issues the `2→2` then `3→6` pair and NO seed.
- [ ] Step 2 — verify fail.
- [ ] Step 3 — implement.
- [ ] Step 4 — verify pass.
- [ ] Step 5 — commit `feat(ablation): run_one_arm drives per-arm windows + reap HF resume`.

## Task 4: Multi-GPU overlay injected by `build_arm_config`

**Files:** runner `build_arm_config` + a new `--num-gpus` CLI arg (default: detect via `torch.cuda.device_count()`, fall back 1). Test: same file.

When `num_gpus >= 2`, inject (idempotent `setdefault`/explicit set):
- `multi_gpu.cov_replicas = num_gpus`, `multi_gpu.factor_workers = num_gpus`, `multi_gpu.alpha_workers = num_gpus`.
- `multi_gpu.eora_workers = num_gpus`.
- `stage2_reap_ream.profile_dp.enabled = True`, `profile_dp.replicas = "auto"`. (Will auto-disable inside the plugin if a reservoir consumer is active — acceptable; log it.)
- `stage5_router_kd.ddp = {enabled: True, world_size: num_gpus, backend: "nccl"}`.
- Do NOT set any `stage6_validate.eval_shard.*` (stage6alt path — would no-op; setting it would be misleading).
- Leave `model.device_map` as base (`auto`) — DP cov/DDP replicas pin their own GPU via `CUDA_VISIBLE_DEVICES`; the per-process model placement is the plugin's job. (If the implementer's read of `run_dp_covariance_collection` / `_spawn_ddp_workers` shows they require a specific parent `device_map`, set it here and note why — see Task 4 verification.)

DDP divisibility guard: after injection, assert `stage5_router_kd.batch_size % world_size == 0`; if `batch_size` is absent (paper_dials_only default), resolve the effective batch the same way the plugin does and assert, else raise a clear error telling the operator to set a divisible batch. (Reuse/extend `assert_paper_recipe_safety` or add `assert_ddp_batch_divisible`.)

**Task 4 verification (implementer MUST confirm with file:line, not assume):**
1. `multi_gpu.*` is read at the TOP-LEVEL config (not under a stage section) — confirm `stage3/orchestrator.py` + `stage4/orchestrator.py` read `config.get("multi_gpu", {})`.
2. `stage4/orchestrator.py:69-85` truly reads `eora_workers` from `multi_gpu` (not `stage4_eora`).
3. Whether ream's by-the-book merge sets `merge_step="mergemoe"` (→ profile_dp auto-disables for ream). Report it; it's acceptable either way but must be stated.
4. `stage5_router_kd.batch_size` effective value under `paper_dials_only`, to make the divisibility guard correct.

- [ ] Step 1 — failing test: `build_arm_config(..., num_gpus=2)` sets every key above to the expected values, sets NO `eval_shard`, and `num_gpus=1` injects NONE of them (1-GPU path unchanged).
- [ ] Step 2 — verify fail.
- [ ] Step 3 — implement + the divisibility guard + a unit test for the guard (raises on an odd batch with world_size=2).
- [ ] Step 4 — verify pass.
- [ ] Step 5 — commit `feat(ablation): multi-GPU overlay (DP cov/SVD, EoRA threads, RKD DDP) gated on num_gpus>=2`.

**Reviewer notes folded in (plan-review, APPROVED):**
- M1: the divisibility guard can only validate the AS-WRITTEN config — `rkd_paper_recipe` runs inside `router_kd.run()` (after this runner emits the config) but does NOT touch `batch_size`, so the pre-check on `stage5_router_kd.batch_size` is valid. Base config sets `batch_size: 2`, `world_size` will be 2 ⇒ `2 % 2 == 0` OK. Document the "as-written only" limitation in the guard docstring; handle an absent `batch_size` (`.get`) instead of letting `int(s5["batch_size"])` KeyError.
- M2: Stage-2 profile DP is silently disabled on a Stage-2 RESUME (`profile_dp.py:39-44`). If the ream arm crashes mid-Stage-2 and resumes, the injected `profile_dp.enabled=true` is overridden to serial with a loud log. Acceptable — note it so the operator isn't surprised.

## Task 5: `whitening_cov` runner flag + persist_shift wiring

**Files:** runner: new `--whitening-cov {anchor,shift,anchored_adaptive}` (default `anchor`). `build_arm_config` sets `stage4_eora.whitening_cov` and, when ≠ anchor, `stage3_svd.persist_shift_covariance = True`. Test: same file.

This makes the acov A/B a matter of running the arm/eval twice with `--whitening-cov anchor` vs `shift` — the main ablation arms take whatever the A/B concludes (default stays `anchor` = historical/safe).

- [ ] Step 1 — failing test: `--whitening-cov shift` ⇒ config has `stage4_eora.whitening_cov=="shift"` AND `stage3_svd.persist_shift_covariance is True`; default `anchor` ⇒ whitening anchor AND persist_shift NOT set (byte-identical historical path).
- [ ] Step 2 — verify fail.
- [ ] Step 3 — implement.
- [ ] Step 4 — verify pass.
- [ ] Step 5 — commit `feat(ablation): --whitening-cov flag wires acov shift-cov + persist`.

## Task 6: Full-suite + final review

- [ ] Run `pytest -k "reap_ream or run_reap"` + the broader ablation/integration tests; paste summaries.
- [ ] Final whole-file read of the runner for seam bugs (per [[feedback_run_full_suite_and_final_review]]): the `--only` filter, idempotent re-runs (a completed arm short-circuits), and that a seeded reap re-run does not re-download.
- [ ] No golden artifacts touched (this runner emits configs only; confirm `git status` shows no golden churn).

---

## Out of scope / explicitly NOT done
- No Stage-1 for either arm (`_shared/` stubbed by `gen_shared.py`; resume≥2 gates Stage-1 off — `run_pipeline.py:182`).
- No `eval_shard` (stage6alt thermometer eval — feature doesn't cover it).
- No stage-plugin code changes; the multi-GPU features are flipped via config only.
- Live ≥2-GPU correctness is validated separately (the `validate_cov_*` scripts + watching arm-1 early stages), not in this runner's unit tests.
